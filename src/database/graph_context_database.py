"""
Graph-structured context database.

Extends ContextDatabase with a typed entity-relation index over the same
key-value store. Each value can optionally carry:

    {
        "db_content": "...",                        # existing field
        "entity":     "kitchen",                    # primary entity (optional)
        "entities":   ["kitchen", "stove"],         # extra mentioned entities (optional)
        "relations":  [
            {"source": "kitchen", "type": "contains", "target": "stove"},
            {"source": "kitchen", "type": "contains", "target": "kettle"},
        ],
    }

If `source` is omitted in a relation, it defaults to the block's `entity`.
Blocks with neither `entity` nor `relations` behave exactly like
MemoryContextDatabase entries (no graph indexing).

The new method `query_subgraph(focus, hops, budget_chars, edge_types)`
performs a budgeted BFS from a focus entity (or db_index) and returns the
neighbour subgraph as structured data. Rendering to text is done by the
caller (MemoryAgentMixin).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from typing import Iterable, Optional

from .context_database import ContextDatabase

logger = logging.getLogger(__name__)


class GraphContextDatabase(ContextDatabase):
    """In-memory graph-indexed context store.

    Implements ContextDatabase verbatim (store / retrieve / delete /
    list_keys / clear) and adds typed-edge indexing on top. Edges are
    derived from the value['entity'] and value['relations'] fields at
    store time. delete() removes the node and all incident edges.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}
        # entity name -> set of db_indices that "own" this entity
        self._entity_to_keys: dict[str, set[str]] = defaultdict(set)
        # entity name -> list of (rel_type, target_entity, source_key)
        self._adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        # entity name -> list of (rel_type, source_entity, source_key)
        self._reverse_adjacency: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        # full ordered edge log: (src, rel, tgt, source_key)
        self._edges: list[tuple[str, str, str, str]] = []
        logger.info("[GraphCtxDB] Using in-memory graph storage")

    # ------------------------------------------------------------------
    # ContextDatabase interface
    # ------------------------------------------------------------------

    def store(self, key: str, value: dict) -> None:
        # Remove any prior incident edges for this key (re-store = update)
        if key in self._store:
            self._remove_edges_for_key(key)

        self._store[key] = value
        self._index_entities_and_edges(key, value)
        logger.debug(f"[GraphCtxDB] Stored key: {key}")

    def retrieve(self, key: str) -> dict:
        if key not in self._store:
            raise KeyError(f"Key not found: {key}")
        return self._store[key]

    def delete(self, key: str) -> None:
        if key in self._store:
            self._remove_edges_for_key(key)
            del self._store[key]
            logger.debug(f"[GraphCtxDB] Deleted key: {key}")

    def list_keys(self, prefix: Optional[str] = None) -> list[str]:
        if prefix is None:
            return list(self._store.keys())
        return [k for k in self._store.keys() if k.startswith(prefix)]

    def clear(self) -> None:
        count = len(self._store)
        self._store.clear()
        self._entity_to_keys.clear()
        self._adjacency.clear()
        self._reverse_adjacency.clear()
        self._edges.clear()
        logger.info(f"[GraphCtxDB] Cleared {count} entries")

    # ------------------------------------------------------------------
    # Graph-specific methods
    # ------------------------------------------------------------------

    def list_entities(self) -> list[str]:
        """Return all entities currently indexed (sorted for determinism)."""
        return sorted(self._entity_to_keys.keys())

    def has_entity(self, entity: str) -> bool:
        return entity in self._entity_to_keys

    def add_edge(
        self,
        src: str,
        rel_type: str,
        tgt: str,
        source_key: str = "",
    ) -> None:
        """Manually add an edge outside of the store() flow.

        Useful for cold-start heuristics that link existing nodes after
        compression (the mixin can call this when it detects similarity
        between newly-stored db_indices).
        """
        self._edges.append((src, rel_type, tgt, source_key))
        self._adjacency[src].append((rel_type, tgt, source_key))
        self._reverse_adjacency[tgt].append((rel_type, src, source_key))

    def query_subgraph(
        self,
        focus: str,
        hops: int = 1,
        budget_chars: int = 4000,
        edge_types: Optional[Iterable[str]] = None,
        include_reverse: bool = True,
    ) -> dict:
        """BFS from a focus entity (or db_index) and return a budgeted subgraph.

        Args:
            focus: entity name OR db_index. If a db_index, we resolve to
                its block's primary entity (value['entity']) when present;
                otherwise we BFS from the db_index itself by treating it
                as a synthetic entity (so callers can always navigate).
            hops: BFS depth limit (>=0). 0 = focus only, no neighbours.
            budget_chars: cap on total db_content chars returned across all
                visited nodes. Nodes are visited BFS-order; per-node content
                is truncated as the remaining budget shrinks.
            edge_types: if set, only follow edges of these types.
            include_reverse: if True, also follow incoming edges (treats
                the graph as undirected for traversal). Edges in the result
                still carry their original direction.

        Returns:
            {
              "focus": <resolved focus entity or db_index>,
              "entities": [
                  {"entity": str, "db_indices": [str, ...],
                   "content_preview": str, "depth": int},
                  ...
              ],
              "edges": [(src, rel_type, tgt, source_key), ...],
              "total_chars": int,
              "truncated": bool,
              "missing": bool,           # True if focus had no node and no edges
            }
        """
        edge_filter: Optional[set[str]] = set(edge_types) if edge_types else None

        # Resolve focus: prefer entity; fall back to db_index -> entity lookup
        focus_entity = focus
        if focus not in self._entity_to_keys and focus in self._store:
            stored = self._store[focus]
            primary = stored.get("entity")
            if isinstance(primary, str) and primary:
                focus_entity = primary

        if (
            focus_entity not in self._entity_to_keys
            and focus_entity not in self._adjacency
            and focus_entity not in self._reverse_adjacency
        ):
            return {
                "focus": focus_entity,
                "entities": [],
                "edges": [],
                "total_chars": 0,
                "truncated": False,
                "missing": True,
            }

        # BFS
        visited: dict[str, int] = {focus_entity: 0}  # entity -> depth
        order: list[str] = [focus_entity]
        queue: deque[tuple[str, int]] = deque([(focus_entity, 0)])
        collected_edges: list[tuple[str, str, str, str]] = []
        seen_edges: set[tuple[str, str, str, str]] = set()

        while queue:
            node, depth = queue.popleft()
            if depth >= hops:
                continue
            outgoing = self._adjacency.get(node, [])
            incoming = self._reverse_adjacency.get(node, []) if include_reverse else []

            for rel_type, target, source_key in outgoing:
                if edge_filter is not None and rel_type not in edge_filter:
                    continue
                edge_key = (node, rel_type, target, source_key)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    collected_edges.append(edge_key)
                if target not in visited:
                    visited[target] = depth + 1
                    order.append(target)
                    queue.append((target, depth + 1))

            for rel_type, source, source_key in incoming:
                if edge_filter is not None and rel_type not in edge_filter:
                    continue
                edge_key = (source, rel_type, node, source_key)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    collected_edges.append(edge_key)
                if source not in visited:
                    visited[source] = depth + 1
                    order.append(source)
                    queue.append((source, depth + 1))

        # Render entities with content under budget_chars
        entities_out: list[dict] = []
        remaining = budget_chars
        truncated = False
        for ent in order:
            keys = sorted(self._entity_to_keys.get(ent, set()))
            preview, used, was_truncated = self._collect_preview(keys, remaining)
            if was_truncated:
                truncated = True
            remaining = max(0, remaining - used)
            entities_out.append({
                "entity": ent,
                "db_indices": keys,
                "content_preview": preview,
                "depth": visited[ent],
            })

        return {
            "focus": focus_entity,
            "entities": entities_out,
            "edges": collected_edges,
            "total_chars": budget_chars - remaining,
            "truncated": truncated,
            "missing": False,
        }

    def get_stats(self) -> dict:
        total_size = sum(len(json.dumps(v, default=str)) for v in self._store.values())
        return {
            "backend": "graph",
            "entry_count": len(self._store),
            "entity_count": len(self._entity_to_keys),
            "edge_count": len(self._edges),
            "total_size_bytes": total_size,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _index_entities_and_edges(self, key: str, value: dict) -> None:
        if not isinstance(value, dict):
            return

        primary_entity: Optional[str] = None
        ent = value.get("entity")
        if isinstance(ent, str) and ent.strip():
            primary_entity = ent.strip()
            self._entity_to_keys[primary_entity].add(key)

        extra = value.get("entities")
        if isinstance(extra, list):
            for e in extra:
                if isinstance(e, str) and e.strip():
                    self._entity_to_keys[e.strip()].add(key)

        rels = value.get("relations")
        if not isinstance(rels, list):
            return

        for rel in rels:
            if not isinstance(rel, dict):
                continue
            src = rel.get("source", primary_entity)
            rel_type = rel.get("type")
            tgt = rel.get("target")
            if not (isinstance(src, str) and src.strip()):
                continue
            if not (isinstance(rel_type, str) and rel_type.strip()):
                continue
            if not (isinstance(tgt, str) and tgt.strip()):
                continue
            src, rel_type, tgt = src.strip(), rel_type.strip(), tgt.strip()
            self._edges.append((src, rel_type, tgt, key))
            self._adjacency[src].append((rel_type, tgt, key))
            self._reverse_adjacency[tgt].append((rel_type, src, key))

    def _remove_edges_for_key(self, key: str) -> None:
        """Remove every edge that was indexed from this key, plus its entity ownership."""
        # Remove ownership of entities (only this key's contribution; entity is kept
        # alive if other keys also reference it)
        prior = self._store.get(key, {})
        for ent_field in ("entity", "entities"):
            v = prior.get(ent_field)
            if isinstance(v, str):
                v = [v]
            if isinstance(v, list):
                for e in v:
                    if isinstance(e, str) and e in self._entity_to_keys:
                        self._entity_to_keys[e].discard(key)
                        if not self._entity_to_keys[e]:
                            del self._entity_to_keys[e]

        # Remove edges sourced by this key
        kept_edges = []
        removed = []
        for edge in self._edges:
            if edge[3] == key:
                removed.append(edge)
            else:
                kept_edges.append(edge)
        self._edges = kept_edges

        # Rebuild adjacency lists by replaying surviving edges. Cheap enough
        # for the sizes we care about (tens to low thousands of nodes).
        self._adjacency.clear()
        self._reverse_adjacency.clear()
        for src, rel_type, tgt, source_key in self._edges:
            self._adjacency[src].append((rel_type, tgt, source_key))
            self._reverse_adjacency[tgt].append((rel_type, src, source_key))

    def _collect_preview(
        self,
        keys: list[str],
        budget_chars: int,
    ) -> tuple[str, int, bool]:
        """Concatenate db_content for a list of keys under a char budget.

        Returns (preview_text, chars_used, truncated).
        """
        if budget_chars <= 0 or not keys:
            return "", 0, bool(keys)

        parts: list[str] = []
        used = 0
        truncated = False
        per_key_budget = max(200, budget_chars // max(1, len(keys)))
        for k in keys:
            value = self._store.get(k)
            if not isinstance(value, dict):
                continue
            content = value.get("db_content", "")
            if not isinstance(content, str):
                content = str(content)
            allow = min(per_key_budget, budget_chars - used)
            if allow <= 0:
                truncated = True
                break
            if len(content) > allow:
                snippet = content[:allow] + "..."
                truncated = True
            else:
                snippet = content
            piece = f"[{k}] {snippet}"
            parts.append(piece)
            used += len(piece)
            if used >= budget_chars:
                truncated = True
                break

        return "\n".join(parts), used, truncated
