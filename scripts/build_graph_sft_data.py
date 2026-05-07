"""
SFT warm-up data generator for graph_db memory mode.

Converts existing lossless_db trajectories into graph_db format by extracting
`entity` / `relations` for each `db_block` written in CompressExperience calls.
The output is a JSONL of trajectories suitable for SFT fine-tuning of an
instruction-tuned base model BEFORE running RL training.

Why this matters:
    Cold-start risk is the largest single variable for graph_db RL training.
    A base model with no exposure to the graph schema will emit malformed
    entity/relations JSON and fail to build a useful graph, leaving RL to
    bootstrap from an unlearnable initial policy. SFT warm-up gives the model
    a working prior over how to write graph blocks.

Workflow:
    1. Take an input JSONL of lossless_db trajectories (each a Trajectory.to_dict()).
    2. For each step that contains a CompressExperience tool call, parse its
       db_blocks. For each block, call an extractor (mock or LLM) to produce
       {entity, entities, relations} fields.
    3. Rewrite the model_response so the CompressExperience call carries
       graph fields, and the ParseResult tool_calls reflect the new arguments.
    4. Emit the rewritten trajectories to an output JSONL.

Extractors:
    --extractor mock      Deterministic stub. Picks the first non-stopword in
                          db_content as `entity`, leaves relations empty.
                          Useful for unit tests and pipeline smoke checks.
    --extractor openai    Calls an OpenAI-compatible API (env: OPENAI_API_KEY,
                          OPENAI_BASE_URL). Prompts the model to emit a strict
                          JSON object {entity, entities, relations}. Bad JSON
                          falls back to mock extraction for that block.

Usage:
    python scripts/build_graph_sft_data.py \\
        --input data/lossless_traj.jsonl \\
        --output data/graph_traj.jsonl \\
        --extractor mock \\
        --edge-schema-name alfworld

CLI is intentionally minimal; extend as needed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.agents.memory.prompts import ALFWORLD_EDGE_SCHEMA, HOTPOTQA_EDGE_SCHEMA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "but", "in", "on", "at", "to", "for",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "i", "you", "he",
    "she", "we", "they", "them", "his", "her", "their",
})


@dataclass
class GraphFields:
    entity: Optional[str] = None
    entities: list[str] = None  # type: ignore
    relations: list[dict] = None  # type: ignore

    def to_dict(self) -> dict:
        out: dict = {}
        if self.entity:
            out["entity"] = self.entity
        if self.entities:
            out["entities"] = self.entities
        if self.relations:
            out["relations"] = self.relations
        return out


class Extractor:
    """Base interface for entity/relation extractors."""

    def extract(self, db_content: str, db_index: str) -> GraphFields:
        raise NotImplementedError


class MockExtractor(Extractor):
    """Deterministic, dependency-free extractor for tests and smoke runs.

    Picks the first non-stopword token (alphanumeric only, len >= 3) in
    db_content as `entity`. Leaves entities/relations empty. The output is
    syntactically valid graph_db; whether it carries USEFUL structure is
    the role of a real LLM extractor.
    """

    _TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")

    def extract(self, db_content: str, db_index: str) -> GraphFields:
        for tok in self._TOKEN_RE.findall(db_content or ""):
            low = tok.lower()
            if len(low) >= 3 and low not in _STOPWORDS:
                return GraphFields(entity=low)
        # Fallback: derive entity from db_index suffix
        return GraphFields(entity=db_index)


class OpenAIExtractor(Extractor):
    """Calls an OpenAI-compatible API to extract graph fields.

    Env vars:
        OPENAI_API_KEY   (required)
        OPENAI_BASE_URL  (optional; defaults to https://api.openai.com/v1)
        OPENAI_MODEL     (optional; defaults to gpt-4o-mini)

    On any extraction failure (network error, malformed JSON), falls back to
    MockExtractor for that block so the pipeline never aborts mid-conversion.
    """

    def __init__(self, edge_schema: Optional[list[str]] = None):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. Run `pip install openai` "
                "or use --extractor mock for an offline pass."
            ) from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        base_url = os.environ.get("OPENAI_BASE_URL")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._edge_schema = edge_schema
        self._fallback = MockExtractor()

    def _system_prompt(self) -> str:
        edge_clause = ""
        if self._edge_schema:
            edge_clause = (
                "\nALLOWED EDGE TYPES (use ONLY these, lowercased):\n  "
                + ", ".join(sorted(self._edge_schema))
                + "\nEdges with other types will be discarded.\n"
            )
        return (
            "You extract a small knowledge-graph fragment from a single "
            "memory block. Reply with a JSON object having exactly these "
            "keys (any may be empty/null):\n"
            "  - entity: string, the primary subject of this block\n"
            "  - entities: list of additional entities mentioned\n"
            "  - relations: list of {source?, type, target} edges\n"
            "Output JSON only, no commentary.\n"
            + edge_clause
        )

    def extract(self, db_content: str, db_index: str) -> GraphFields:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": f"db_index: {db_index}\n\ndb_content:\n{db_content}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or ""
            parsed = json.loads(raw)
            return _coerce_graph_fields(parsed, fallback_entity=db_index)
        except Exception as e:  # noqa: BLE001 — log and fall back
            logger.warning("OpenAI extractor failed for %s: %s — falling back to mock", db_index, e)
            return self._fallback.extract(db_content, db_index)


def _coerce_graph_fields(obj: Any, fallback_entity: str) -> GraphFields:
    """Validate / coerce LLM output into a clean GraphFields."""
    if not isinstance(obj, dict):
        return GraphFields(entity=fallback_entity)
    out = GraphFields()
    ent = obj.get("entity")
    if isinstance(ent, str) and ent.strip():
        out.entity = ent.strip()
    extras = obj.get("entities")
    if isinstance(extras, list):
        cleaned = [e.strip() for e in extras if isinstance(e, str) and e.strip()]
        if cleaned:
            out.entities = cleaned
    rels = obj.get("relations")
    if isinstance(rels, list):
        cleaned_rels: list[dict] = []
        for r in rels:
            if not isinstance(r, dict):
                continue
            rt = r.get("type")
            tgt = r.get("target")
            if not (isinstance(rt, str) and rt.strip() and isinstance(tgt, str) and tgt.strip()):
                continue
            entry = {"type": rt.strip().lower(), "target": tgt.strip()}
            src = r.get("source")
            if isinstance(src, str) and src.strip():
                entry["source"] = src.strip()
            cleaned_rels.append(entry)
        if cleaned_rels:
            out.relations = cleaned_rels
    if out.entity is None:
        out.entity = fallback_entity
    return out


# ---------------------------------------------------------------------------
# Trajectory rewriting
# ---------------------------------------------------------------------------

def rewrite_trajectory(traj: dict, extractor: Extractor) -> dict:
    """Return a copy of ``traj`` with each CompressExperience tool call
    rewritten to carry graph fields on its db_blocks.

    Only ``ParseResult.tool_calls.arguments`` is rewritten. ``model_response``
    text is left as-is, on the assumption that downstream SFT formatters
    re-render the response from the structured tool call. If your pipeline
    instead uses model_response directly, also pass --rewrite-response.
    """
    new_traj = json.loads(json.dumps(traj))  # deep copy
    n_blocks_rewritten = 0
    for step in new_traj.get("steps", []):
        pr = step.get("parse_result")
        if not pr:
            continue
        for tc in pr.get("tool_calls", []):
            if tc.get("name") != "CompressExperience":
                continue
            args = tc.get("arguments", {})
            if not isinstance(args, dict):
                continue
            db_blocks = args.get("db_blocks")
            # db_blocks may be a JSON string, a list, or absent (lossy mode).
            if isinstance(db_blocks, str):
                try:
                    parsed_blocks = json.loads(db_blocks)
                except json.JSONDecodeError:
                    continue
            else:
                parsed_blocks = db_blocks
            if not isinstance(parsed_blocks, list):
                continue
            for blk in parsed_blocks:
                if not isinstance(blk, dict):
                    continue
                idx = blk.get("db_index", "")
                content = blk.get("db_content", "")
                if not isinstance(idx, str) or not idx.strip():
                    continue
                gf = extractor.extract(content if isinstance(content, str) else "", idx)
                gf_dict = gf.to_dict()
                if gf_dict:
                    blk.update(gf_dict)
                    n_blocks_rewritten += 1
            # Write back in the same format we read it.
            if isinstance(db_blocks, str):
                args["db_blocks"] = json.dumps(parsed_blocks)
            else:
                args["db_blocks"] = parsed_blocks
    new_traj.setdefault("info", {})["graph_sft_rewritten"] = True
    new_traj["info"]["graph_sft_blocks_rewritten"] = n_blocks_rewritten
    return new_traj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: str, items: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def _resolve_edge_schema(name: Optional[str]) -> Optional[list[str]]:
    if not name:
        return None
    name = name.lower()
    if name in ("alfworld", "alfw"):
        return list(ALFWORLD_EDGE_SCHEMA)
    if name in ("hotpotqa", "hotpot"):
        return list(HOTPOTQA_EDGE_SCHEMA)
    if name == "open":
        return None
    raise ValueError(f"Unknown edge_schema name: {name}. Choose alfworld, hotpotqa, or open.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL of trajectories")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--extractor", choices=("mock", "openai"), default="mock")
    parser.add_argument("--edge-schema-name", default=None,
                        help="alfworld | hotpotqa | open. Used by openai extractor.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    edge_schema = _resolve_edge_schema(args.edge_schema_name)
    if args.extractor == "mock":
        extractor: Extractor = MockExtractor()
    else:
        extractor = OpenAIExtractor(edge_schema=edge_schema)

    trajectories = list(_read_jsonl(args.input))
    rewritten = [rewrite_trajectory(t, extractor) for t in trajectories]
    n = _write_jsonl(args.output, rewritten)
    logger.info("Wrote %d rewritten trajectories to %s", n, args.output)


if __name__ == "__main__":
    main()
