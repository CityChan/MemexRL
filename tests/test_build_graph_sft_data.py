"""
Unit tests for the graph_db SFT warm-up data generator.

Covers the deterministic mock extractor + trajectory rewriting + the JSONL
end-to-end pipeline (no LLM call). The OpenAI path is exercised separately
in environments where the SDK + key are available.

Run with:
    python -m unittest tests.test_build_graph_sft_data -v
"""
import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.build_graph_sft_data import (
    MockExtractor,
    rewrite_trajectory,
    _coerce_graph_fields,
    main as cli_main,
)


def _trajectory_with_compress(db_blocks_field) -> dict:
    """Build a minimal trajectory dict that has one CompressExperience step.

    db_blocks_field is passed through verbatim into the tool call args, so we
    can test both list and JSON-string shapes.
    """
    return {
        "uid": "u",
        "name": "agent",
        "task": None,
        "reward": 0.0,
        "info": {},
        "steps": [
            {
                "chat_completions": [],
                "observation": None,
                "action": None,
                "model_response": "...",
                "model_output": None,
                "info": {},
                "reward": 0.0,
                "done": False,
                "mc_return": 0.0,
                "is_compression_boundary": True,
                "chat_completions_before_compression": [],
                "compression_summary": "",
                "chat_completions_at_generation": [],
                "num_retrievals_in_step": 0,
                "num_compressions_in_step": 1,
                "retrieval_indices": [],
                "context_length": 5000,
                "parse_result": {
                    "tool_calls": [
                        {
                            "name": "CompressExperience",
                            "arguments": {
                                "summary": "ix",
                                "db_blocks": db_blocks_field,
                            },
                        }
                    ],
                    "format_errors": [],
                    "had_tool_attempt": True,
                },
            }
        ],
    }


class TestMockExtractor(unittest.TestCase):

    def test_picks_first_non_stopword(self):
        ex = MockExtractor()
        gf = ex.extract("the kitchen contains a stove", db_index="x")
        self.assertEqual(gf.entity, "kitchen")
        self.assertIsNone(gf.relations)

    def test_falls_back_to_db_index_when_empty(self):
        ex = MockExtractor()
        gf = ex.extract("", db_index="ctx_obs_001")
        self.assertEqual(gf.entity, "ctx_obs_001")

    def test_skips_short_tokens_and_stopwords(self):
        ex = MockExtractor()
        gf = ex.extract("a an of to from forest", db_index="x")
        self.assertEqual(gf.entity, "forest")


class TestCoerceGraphFields(unittest.TestCase):

    def test_lowercases_and_filters_relations(self):
        out = _coerce_graph_fields({
            "entity": "kitchen",
            "entities": ["kitchen", "stove"],
            "relations": [
                {"type": "Contains", "target": "stove"},
                {"type": "  ", "target": "broken"},                   # dropped
                {"type": "near", "target": "fridge", "source": "stove"},
                "not a dict",                                          # dropped
            ],
        }, fallback_entity="x")
        self.assertEqual(out.entity, "kitchen")
        self.assertEqual(out.entities, ["kitchen", "stove"])
        self.assertEqual(out.relations, [
            {"type": "contains", "target": "stove"},
            {"type": "near", "target": "fridge", "source": "stove"},
        ])

    def test_uses_fallback_when_no_entity_present(self):
        out = _coerce_graph_fields({}, fallback_entity="ctx_x")
        self.assertEqual(out.entity, "ctx_x")

    def test_non_dict_input_falls_back(self):
        out = _coerce_graph_fields("garbage", fallback_entity="ctx_x")
        self.assertEqual(out.entity, "ctx_x")


class TestRewriteTrajectory(unittest.TestCase):

    def test_rewrites_db_blocks_as_list(self):
        traj = _trajectory_with_compress([
            {"db_index": "k1", "db_content": "the kitchen contains a stove"}
        ])
        out = rewrite_trajectory(traj, MockExtractor())
        blk = out["steps"][0]["parse_result"]["tool_calls"][0]["arguments"]["db_blocks"][0]
        self.assertEqual(blk["entity"], "kitchen")
        self.assertEqual(out["info"]["graph_sft_blocks_rewritten"], 1)
        self.assertTrue(out["info"]["graph_sft_rewritten"])

    def test_rewrites_db_blocks_as_json_string(self):
        # When db_blocks comes in as a JSON string, output should remain a JSON
        # string with the graph fields injected.
        original_blocks = [{"db_index": "k1", "db_content": "the kitchen contains a stove"}]
        traj = _trajectory_with_compress(json.dumps(original_blocks))
        out = rewrite_trajectory(traj, MockExtractor())
        blocks_str = out["steps"][0]["parse_result"]["tool_calls"][0]["arguments"]["db_blocks"]
        self.assertIsInstance(blocks_str, str)
        parsed = json.loads(blocks_str)
        self.assertEqual(parsed[0]["entity"], "kitchen")

    def test_skips_non_compress_tool_calls(self):
        # A trajectory with only env tool calls should be left untouched.
        traj = _trajectory_with_compress([
            {"db_index": "k1", "db_content": "kitchen"}
        ])
        # Replace the compress call with an env tool call
        traj["steps"][0]["parse_result"]["tool_calls"][0] = {
            "name": "execute_action", "arguments": {"action": "look"},
        }
        out = rewrite_trajectory(traj, MockExtractor())
        self.assertEqual(out["info"]["graph_sft_blocks_rewritten"], 0)

    def test_does_not_mutate_input(self):
        original = _trajectory_with_compress([
            {"db_index": "k1", "db_content": "kitchen"}
        ])
        snapshot = json.dumps(original, sort_keys=True)
        rewrite_trajectory(original, MockExtractor())
        self.assertEqual(json.dumps(original, sort_keys=True), snapshot,
                         "rewrite_trajectory must not mutate its input")

    def test_handles_missing_db_blocks_field(self):
        # lossy / rag mode: no db_blocks. Must not crash.
        traj = _trajectory_with_compress(None)
        del traj["steps"][0]["parse_result"]["tool_calls"][0]["arguments"]["db_blocks"]
        out = rewrite_trajectory(traj, MockExtractor())
        self.assertEqual(out["info"]["graph_sft_blocks_rewritten"], 0)


class TestEndToEndJSONL(unittest.TestCase):

    def test_cli_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in.jsonl")
            out_path = os.path.join(td, "out.jsonl")
            traj = _trajectory_with_compress([
                {"db_index": "k1", "db_content": "kitchen contains a stove"},
                {"db_index": "k2", "db_content": "stove is empty"},
            ])
            with open(in_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(traj) + "\n")
            cli_main(["--input", in_path, "--output", out_path, "--extractor", "mock"])
            with open(out_path, "r", encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
            self.assertEqual(len(lines), 1)
            blocks = lines[0]["steps"][0]["parse_result"]["tool_calls"][0]["arguments"]["db_blocks"]
            self.assertEqual(blocks[0]["entity"], "kitchen")
            self.assertEqual(blocks[1]["entity"], "stove")


if __name__ == "__main__":
    unittest.main()
