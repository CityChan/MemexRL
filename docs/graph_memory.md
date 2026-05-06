# Graph Memory for MemexRL

> Status: implemented (v0). Tests in `tests/test_graph_memory.py`. No training
> code changed yet — this is purely an additive extension to the agent memory
> stack and a new context-database backend.

## TL;DR

We add a new memory mode `graph_db` that turns MemexRL's flat indexed memory
into a typed-edge graph. Compression now optionally attaches `entity` /
`entities` / `relations` to each `db_block`; a new `GraphContextDatabase`
backend indexes these into a graph, and a new memory tool
`QueryGraph(focus, hops, budget, edge_types)` lets the agent retrieve a
focus-centred subgraph instead of (or in addition to) a single block.

Everything is **strictly additive**:

- Existing modes (`lossless_db`, `lossy`, `rag`, `none`) are unchanged
  byte-for-byte (validated by regression tests).
- `db_block` graph fields are all optional. A `graph_db` block with no
  `entity` and no `relations` behaves identically to a `lossless_db` block.
- The `GraphContextDatabase` implements the existing `ContextDatabase`
  interface; you can drop it in anywhere a `MemoryContextDatabase` was used.

## Why

This work merges two threads:

1. **MemexRL** has the right *learning substrate*: an agent that calls
   `CompressExperience` and `ReadExperience` as tools, trained end-to-end
   with RL (Slime/GRPO) and a reward shaper that penalises context overflow
   and redundancy. But its storage is a flat key→value map, so cross-cutting
   evidence (e.g. an entity that appears in subtask A and is needed in
   subtask B) is hard to recall by anything other than the exact key the
   model wrote down at compression time.

2. **ContextGraph** (the sister project at `../context-graph`) had the right
   *representation* — a typed-edge graph over context units with four
   policy-controlled operations (Merge / AddEdge / Select / Prune) and a
   focus-centred `Reconstruct(G, c, B)`. But it placed the graph **inside
   the prompt** and rewarded the policy directly for graph-shape signals.
   That created two failure modes we observed empirically:

   - **Reward hacking on graph shaping.** With a small task-reward signal
     (LLM-judge EM on multi-hop QA) and a constant graph-shape bonus, the
     policy learned to maximise the bonus while letting task reward decay.
   - **Rollout/recompute drift.** Mutating graph state inside the prompt
     each turn breaks vLLM prefix-cache reuse and inflates the
     `kl_loss` between the rollout and the FSDP recompute pass.

`graph_db` resolves both: the graph lives **outside** the prompt (in
`GraphContextDatabase`), and the policy interacts with it through bounded,
deterministic tool calls (`CompressExperience` to write, `QueryGraph` /
`ReadExperience` to read). Reward shaping continues to use the existing
`MemoryEfficiencyShaper` signals (compression frequency, overflow penalty,
redundancy penalty); no new graph-shape bonus is introduced.

## Architecture

```
+-------------------------------------+        agent emits tool call
|  policy (Qwen3-30B-A3B / 4B / etc.) | ─────────────────────────────────┐
+-------------------------------------+                                  │
              ▲                                                          │
              │ context status                                           ▼
              │ + observation                                       +-------------+
              │                                                     |  agent loop |
+-------------------------------------+                             +-------------+
|  MemoryAgentMixin (graph_db mode)   | ◀───── execute_memory_tool ──────┤
|                                     |                                  │
|  - is_memory_tool(name)             |                                  │
|  - _execute_compress (parses        |                                  │
|      entity / entities / relations) |                                  │
|  - _execute_retrieve (db_index)     |                                  │
|  - _execute_graph_query (QueryGraph)|                                  │
+-------------------------------------+                                  │
              │ store(key, value-with-graph-fields)                      │
              │ retrieve(key)                                            │
              │ query_subgraph(focus, hops, budget, edge_types)          │
              ▼                                                          │
+-------------------------------------+                                  │
|  GraphContextDatabase               |                                  │
|  - _store: key -> value             |                                  │
|  - _entity_to_keys: entity -> {key} |                                  │
|  - _adjacency: ent -> [(rel,tgt,k)] |                                  │
|  - _reverse_adjacency: ent -> [...] |                                  │
|  - _edges: [(src, rel, tgt, key)]   |                                  │
+-------------------------------------+                                  │
              ▲                                                          │
              └───── tools available alongside env tools ────────────────┘
                     (file_editor, execute_bash, search, ...)
```

### `db_block` schema (graph_db mode)

```jsonc
{
  "db_index":   "obs_kitchen_001",        // required, unchanged
  "db_content": "Kitchen contains ...",    // required, unchanged
  "entity":     "kitchen",                 // OPTIONAL: primary subject of this node
  "entities":   ["kitchen", "stove"],      // OPTIONAL: extra entities mentioned
  "relations": [                           // OPTIONAL: typed edges originating here
    {"type": "contains",   "target": "stove"},
    {"type": "contains",   "target": "kettle"},
    {"source": "alice", "type": "knows", "target": "bob"}  // explicit override
  ]
}
```

Edge types are **untyped strings chosen by the policy** — i.e. the schema is
open. This mirrors A-MEM / Mem0 and matches our use-case (the appropriate
edge labels for ALFWorld differ from those for HotpotQA). For tasks that
need a closed schema, the prompt or a post-processor can constrain types.

If `entity` is absent on a relation, the relation's `source` defaults to
the block's `entity`. A block can also declare relations whose source is a
*different* entity (useful when one observation describes a relation
between two third parties, e.g. "Alice told Bob …").

### `QueryGraph` tool

```
QueryGraph(focus: str, hops: int = 1, budget: int = 2000,
           edge_types: list[str] | None = None) -> text
```

- BFS from `focus` (entity name preferred; db_index also accepted and
  resolved through the stored block's `entity` field).
- Traversal is undirected: incoming edges are followed too, but the
  rendered edges keep their original direction.
- `hops` capped at 4. `budget` clamped to [200, 8000] characters.
- Returns a structured text block (entities + edges + per-node previews)
  that the agent can scan, then optionally call
  `ReadExperience(db_index)` for the full content of any node.

`QueryGraph` is the moral analogue of ContextGraph's `Reconstruct(G, c, B)`,
but as an external query rather than an in-prompt mutation.

### `GraphContextDatabase`

In-memory backend. SQLite/Redis ports are deferred to v1; if needed they
just wrap the same indices in persistent storage. Key methods:

- `store / retrieve / delete / list_keys / clear`: inherited
  `ContextDatabase` interface, unchanged semantics.
- `add_edge(src, rel, tgt, source_key="")`: manual edge addition outside
  of `store`. Useful for cold-start heuristics that link existing nodes
  (e.g. a similarity-based linker that runs after compression).
- `list_entities()`: for `[Context Status]` injection.
- `query_subgraph(...)`: BFS query described above.

`delete(key)` correctly removes both the node and all edges sourced from
that key, and only releases entity ownership when no other key references
that entity.

## Why this is novel relative to prior work

| System | Memory shape | Typed cross-edges | RL-trained memory ops |
|---|---|---|---|
| ReAct / ReAct+              | linear history     | ❌  | ❌ |
| MemGPT                       | tiered text         | ❌  | ❌ (prompted) |
| FoldAgent / Context-Folding  | tree of subtasks    | ❌  | ✅ |
| A-MEM                        | graph of notes      | ✅  | ❌ (prompted) |
| Mem0                         | vector + graph      | ✅  | ❌ (prompted) |
| **MemexRL (v0)**             | indexed flat KV     | partial (via index) | ✅ |
| **MemexRL + graph_db (this)**| **typed graph + budgeted query** | **✅** | **✅** |

The unique cell is *RL-trained query-and-write policy over a typed-edge
agent memory graph.* That is the contribution.

## Backwards compatibility

- All existing call sites pass `compression_mode` as one of
  `none / lossless_db / lossy / rag`. Those continue to work bit-identically.
- The change to `_do_lossless_compress`'s block tuple shape (now
  `(db_index, db_content, graph_fields_dict)`) is internal; the new
  `graph_fields_dict` is empty for non-graph mode and the stored value dict
  is unchanged.
- `is_memory_tool` adds `"QueryGraph"`. If a non-graph-mode agent receives
  a `QueryGraph` call from a confused policy, `_execute_graph_query`
  returns a clean validation error rather than crashing.

## What still needs doing

These are the next concrete steps. None of them is implemented yet.

1. **Cold-start prompted run.** Wire `compression_mode="graph_db"` and
   `get_memory_tools_prompt_graph()` into the existing
   `ToolAgentWithMemory` factory in `src/agents/tool_agent.py`, and
   sanity-check on one ALFWorld episode and one HotpotQA episode with a
   fixed instruction-tuned model (no RL). Confirm the model emits well-formed
   `entity` / `relations`, that `QueryGraph` returns sensible subgraphs, and
   that `[Context Status]` correctly lists entities.
2. **HotpotQA / 2WikiMQA / MuSiQue env wrappers.** MemexRL currently ships
   with ALFWorld. Add `src/environments/multihop_qa/` with the same Wikipedia
   retriever the sister `context-graph` project uses, so we can A/B the
   `graph_db` mode against `lossless_db` on multi-hop QA where the entity
   structure is naturally exposed.
3. **Reward shaping audit.** Confirm `MemoryEfficiencyShaper` does not
   need a new term for graph quality. The hypothesis (and the entire reason
   for this redesign) is that task reward + existing overflow/redundancy
   penalties suffice — graph-shape bonuses caused the v1 ContextGraph
   reward-hacking failure and are deliberately *not* introduced here.
4. **Slime training pass.** Once 1–3 are settled, run a single-task
   end-to-end Slime/GRPO training pass with `graph_db` and compare to
   `lossless_db` on the same task / model / seeds.
5. **Persistence backends.** A `SQLiteGraphContextDatabase` is mechanical
   to add (mirror the existing SQLite backend, plus a small `edges` table).
   Defer until persistence is actually needed; in-memory is correct for
   per-trajectory training.

## Ablation plan

Three runs per benchmark, same seed and step budget:

| Mode | Memory backend | What we measure |
|---|---|---|
| `lossless_db`    | flat KV        | baseline; tests that any indexed memory beats no memory |
| `rag`            | BM25 chunks    | baseline; tests that key-based retrieval beats topical retrieval |
| `graph_db`       | typed graph    | does typed-edge structure + `QueryGraph` add value? |

Per run, log: task EM, average compression count, average retrieval count,
average chars per retrieval, redundant-tool-call rate, context-overflow rate.
The headline number is task EM at fixed compute; the secondary story is
whether `graph_db` reduces redundant tool calls without sacrificing EM.

## File-level changes summary

- `src/database/graph_context_database.py` (new, ~280 lines)
- `src/database/__init__.py` (new, exports)
- `src/database/context_database.py` (factory: `backend="graph"` → `GraphContextDatabase`)
- `src/agents/memory/mixin.py`
  - `init_memory`: support `compression_mode="graph_db"`
  - `is_memory_tool`: include `"QueryGraph"`
  - `execute_memory_tool`: dispatch `QueryGraph` to `_execute_graph_query`
  - `_execute_compress`: accept optional `entity`/`entities`/`relations`
    on each db_block in graph mode, dispatch through `_parse_graph_fields`
  - `_do_lossless_compress`: now stores graph fields when present
  - `_execute_graph_query`: new
  - `_parse_graph_fields`: new (validation helper)
  - `_render_subgraph`: new (formatting helper)
  - `update_from_env` / `get_context_status`: include graph mode in
    status injection; show `[Indexed entities: …]`
- `src/agents/memory/prompts.py`
  - new `_GRAPH_*` templates and `get_memory_tools_prompt_graph()`
- `src/agents/memory/__init__.py` (export `get_memory_tools_prompt_graph`)
- `tests/test_graph_memory.py` (new, 24 tests, all pass)
