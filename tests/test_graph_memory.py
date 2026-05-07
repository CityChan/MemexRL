"""
Unit tests for GraphContextDatabase and MemoryAgentMixin graph_db mode.

Run with:
    python -m pytest tests/test_graph_memory.py -v
or standalone:
    python tests/test_graph_memory.py
"""
import os
import sys
import unittest

# Make `src` importable when run from repo root or this directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.database.graph_context_database import GraphContextDatabase
from src.database.context_database import create_context_database
from src.agents.memory.mixin import MemoryAgentMixin
from src.agents.memory.types import MemoryToolResult


# ---------------------------------------------------------------------------
# GraphContextDatabase
# ---------------------------------------------------------------------------

class TestGraphContextDatabase(unittest.TestCase):

    def setUp(self):
        self.db = GraphContextDatabase()

    def test_implements_context_database_interface(self):
        # store / retrieve / delete / list_keys / clear
        self.db.store("a", {"db_content": "alpha", "entity": "A"})
        self.assertEqual(self.db.retrieve("a")["db_content"], "alpha")
        self.assertIn("a", self.db.list_keys())
        self.db.delete("a")
        with self.assertRaises(KeyError):
            self.db.retrieve("a")

    def test_factory_returns_graph_backend(self):
        db = create_context_database(backend="graph")
        self.assertIsInstance(db, GraphContextDatabase)

    def test_entity_indexing(self):
        self.db.store("k1", {"db_content": "...", "entity": "kitchen"})
        self.db.store("k2", {"db_content": "...", "entity": "stove",
                              "entities": ["kitchen"]})  # k2 also mentions kitchen
        # kitchen is owned by both k1 and k2 (k2 via 'entities' extra list)
        self.assertEqual(set(self.db.list_entities()), {"kitchen", "stove"})
        self.assertIn("kitchen", self.db._entity_to_keys)
        self.assertEqual(self.db._entity_to_keys["kitchen"], {"k1", "k2"})

    def test_relations_build_edges(self):
        self.db.store("k1", {
            "db_content": "...",
            "entity": "kitchen",
            "relations": [
                {"type": "contains", "target": "stove"},
                {"type": "contains", "target": "kettle"},
            ],
        })
        # adjacency
        self.assertEqual(len(self.db._adjacency["kitchen"]), 2)
        self.assertEqual(self.db._reverse_adjacency["stove"][0][1], "kitchen")
        self.assertEqual(self.db._reverse_adjacency["kettle"][0][1], "kitchen")

    def test_query_subgraph_bfs_hops(self):
        # kitchen --contains--> stove --located_on--> counter
        self.db.store("k1", {"db_content": "kitchen body", "entity": "kitchen",
                              "relations": [{"type": "contains", "target": "stove"}]})
        self.db.store("k2", {"db_content": "stove body", "entity": "stove",
                              "relations": [{"type": "located_on", "target": "counter"}]})
        self.db.store("k3", {"db_content": "counter body", "entity": "counter"})

        # hops=1 from kitchen: visits kitchen + stove only (counter is 2 hops away)
        r1 = self.db.query_subgraph("kitchen", hops=1)
        ents1 = {e["entity"] for e in r1["entities"]}
        self.assertEqual(ents1, {"kitchen", "stove"})
        self.assertFalse(r1["missing"])

        # hops=2: visits kitchen + stove + counter
        r2 = self.db.query_subgraph("kitchen", hops=2)
        ents2 = {e["entity"] for e in r2["entities"]}
        self.assertEqual(ents2, {"kitchen", "stove", "counter"})

        # depths are correct
        depth = {e["entity"]: e["depth"] for e in r2["entities"]}
        self.assertEqual(depth["kitchen"], 0)
        self.assertEqual(depth["stove"], 1)
        self.assertEqual(depth["counter"], 2)

    def test_query_subgraph_includes_reverse_edges(self):
        # B --rel--> A. Querying focus=A should still find B via reverse adjacency.
        self.db.store("b", {"db_content": "b body", "entity": "B",
                             "relations": [{"type": "points_to", "target": "A"}]})
        self.db.store("a", {"db_content": "a body", "entity": "A"})
        r = self.db.query_subgraph("A", hops=1)
        self.assertIn("B", {e["entity"] for e in r["entities"]})

    def test_query_subgraph_edge_type_filter(self):
        self.db.store("k1", {
            "db_content": "...", "entity": "X",
            "relations": [
                {"type": "contains", "target": "Y"},
                {"type": "near", "target": "Z"},
            ],
        })
        self.db.store("k2", {"db_content": "...", "entity": "Y"})
        self.db.store("k3", {"db_content": "...", "entity": "Z"})

        r = self.db.query_subgraph("X", hops=1, edge_types=["contains"])
        ents = {e["entity"] for e in r["entities"]}
        # Should find Y but NOT Z (edge filtered out)
        self.assertIn("Y", ents)
        self.assertNotIn("Z", ents)

    def test_query_subgraph_budget_truncates(self):
        big = "x" * 5000
        self.db.store("k1", {"db_content": big, "entity": "X",
                              "relations": [{"type": "rel", "target": "Y"}]})
        self.db.store("k2", {"db_content": big, "entity": "Y"})
        r = self.db.query_subgraph("X", hops=1, budget_chars=500)
        self.assertTrue(r["truncated"])
        self.assertLessEqual(r["total_chars"], 500)

    def test_query_subgraph_focus_by_db_index(self):
        # Resolve focus through db_index -> stored entity
        self.db.store("kitchen_obs_1", {"db_content": "...", "entity": "kitchen",
                                         "relations": [{"type": "contains", "target": "stove"}]})
        self.db.store("stove_obs_1", {"db_content": "...", "entity": "stove"})
        r = self.db.query_subgraph("kitchen_obs_1", hops=1)
        self.assertEqual(r["focus"], "kitchen")
        self.assertIn("stove", {e["entity"] for e in r["entities"]})

    def test_query_subgraph_missing_focus(self):
        r = self.db.query_subgraph("nonexistent", hops=2)
        self.assertTrue(r["missing"])
        self.assertEqual(r["entities"], [])

    def test_delete_removes_incident_edges(self):
        self.db.store("k1", {"db_content": "...", "entity": "A",
                              "relations": [{"type": "rel", "target": "B"}]})
        self.db.store("k2", {"db_content": "...", "entity": "B"})
        self.assertEqual(len(self.db._edges), 1)
        self.db.delete("k1")
        self.assertEqual(len(self.db._edges), 0)
        # B is still in entity index because k2 owns it
        self.assertIn("B", self.db._entity_to_keys)
        self.assertNotIn("A", self.db._entity_to_keys)

    def test_re_store_replaces_edges(self):
        self.db.store("k1", {"db_content": "...", "entity": "A",
                              "relations": [{"type": "rel_old", "target": "B"}]})
        self.db.store("k1", {"db_content": "...", "entity": "A",
                              "relations": [{"type": "rel_new", "target": "C"}]})
        # Only the new edge should remain
        self.assertEqual(len(self.db._edges), 1)
        self.assertEqual(self.db._edges[0][1], "rel_new")
        self.assertEqual(self.db._edges[0][2], "C")

    def test_explicit_relation_source_overrides_entity(self):
        # A block can declare relations whose source is a different entity
        self.db.store("k1", {
            "db_content": "...",
            "entity": "narrator",
            "relations": [{"source": "alice", "type": "knows", "target": "bob"}],
        })
        self.assertIn("alice", self.db._adjacency)
        self.assertEqual(self.db._adjacency["alice"][0][1], "bob")
        self.assertNotIn("narrator", self.db._adjacency)

    def test_blocks_without_graph_fields_behave_like_kv(self):
        # If you store without entity/relations, no graph indexing happens.
        self.db.store("plain_block", {"db_content": "no entities here"})
        self.assertEqual(self.db.list_entities(), [])
        self.assertEqual(self.db._edges, [])
        self.assertEqual(self.db.retrieve("plain_block")["db_content"], "no entities here")

    def test_get_stats(self):
        self.db.store("k1", {"db_content": "...", "entity": "X",
                              "relations": [{"type": "rel", "target": "Y"}]})
        s = self.db.get_stats()
        self.assertEqual(s["backend"], "graph")
        self.assertEqual(s["entry_count"], 1)
        self.assertEqual(s["edge_count"], 1)
        self.assertGreaterEqual(s["entity_count"], 1)
        self.assertIsNone(s["edge_schema"])
        self.assertEqual(s["dropped_edge_count"], 0)

    def test_edge_type_lowercase_canonicalization(self):
        # "Contains" / "CONTAINS" / "contains" must merge into a single edge type.
        self.db.store("k1", {
            "db_content": "...", "entity": "X",
            "relations": [
                {"type": "Contains", "target": "Y"},
                {"type": "CONTAINS", "target": "Z"},
                {"type": "contains", "target": "W"},
            ],
        })
        types = {e[1] for e in self.db._edges}
        self.assertEqual(types, {"contains"})

    def test_closed_edge_schema_drops_unknown_types(self):
        db = GraphContextDatabase(edge_schema=["contains", "located_in"])
        db.store("k1", {
            "db_content": "...", "entity": "kitchen",
            "relations": [
                {"type": "contains", "target": "stove"},     # kept
                {"type": "near", "target": "fridge"},          # dropped
                {"type": "Located_in", "target": "house"},     # kept (case-folded)
            ],
        })
        edge_types = {e[1] for e in db._edges}
        self.assertEqual(edge_types, {"contains", "located_in"})
        self.assertEqual(db._dropped_edge_count, 1)
        self.assertEqual(db.get_stats()["dropped_edge_count"], 1)

    def test_closed_schema_filters_query_edge_types_with_case_insensitivity(self):
        db = GraphContextDatabase(edge_schema=["contains", "near"])
        db.store("k1", {"db_content": "...", "entity": "X",
                         "relations": [
                             {"type": "contains", "target": "Y"},
                             {"type": "near", "target": "Z"},
                         ]})
        # Caller passes uppercased filter — should still match
        r = db.query_subgraph("X", hops=1, edge_types=["CONTAINS"])
        ents = {e["entity"] for e in r["entities"]}
        self.assertIn("Y", ents)
        self.assertNotIn("Z", ents)

    def test_add_edge_respects_schema_and_indexes_endpoints(self):
        db = GraphContextDatabase(edge_schema=["link"])
        # Allowed type → indexed; endpoints appear in entity index
        self.assertTrue(db.add_edge("A", "Link", "B", source_key=""))
        self.assertIn("A", db._entity_to_keys)
        self.assertIn("B", db._entity_to_keys)
        self.assertEqual(db._edges[0][1], "link")
        # Disallowed type → rejected, dropped count increments
        self.assertFalse(db.add_edge("A", "unknown", "C"))
        self.assertEqual(db._dropped_edge_count, 1)
        self.assertNotIn("C", db._entity_to_keys)


# ---------------------------------------------------------------------------
# MemoryAgentMixin (graph_db mode) - minimal harness
# ---------------------------------------------------------------------------

class _StubTokenManager:
    def get_working_tokens(self, messages):
        return sum(len(str(m.get("content", ""))) for m in messages) // 4

    def get_total_tokens(self, messages):
        return self.get_working_tokens(messages)


class _GraphAgentHarness(MemoryAgentMixin):
    """Minimal harness around the mixin (no super class) for unit testing."""

    def __init__(self):
        self.token_manager = _StubTokenManager()
        self.messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "task description"},
            {"role": "assistant", "content": "I will explore."},
            {"role": "user", "content": "obs: you see a kitchen"},
            {"role": "assistant", "content": "I will look around."},
            {"role": "user", "content": "obs: stove and kettle"},
        ]
        self.step = 0
        self.init_memory(compression_mode="graph_db")


class TestMemoryAgentMixinGraphMode(unittest.TestCase):

    def setUp(self):
        self.agent = _GraphAgentHarness()

    def test_init_creates_graph_backend(self):
        self.assertEqual(self.agent.compression_mode, "graph_db")
        self.assertIsInstance(self.agent.context_db, GraphContextDatabase)

    def test_query_graph_is_recognized(self):
        self.assertTrue(self.agent.is_memory_tool("QueryGraph"))
        self.assertTrue(self.agent.is_memory_tool("CompressExperience"))
        self.assertTrue(self.agent.is_memory_tool("ReadExperience"))
        self.assertFalse(self.agent.is_memory_tool("file_editor"))

    def test_compress_with_entity_and_relations(self):
        params = {
            "summary": "Index map:\n- k1 [entity=kitchen] - kitchen layout\nStatus: explore done.",
            "db_blocks": [
                {
                    "db_index": "k1",
                    "db_content": "Kitchen with stove and kettle.",
                    "entity": "kitchen",
                    "relations": [
                        {"type": "contains", "target": "stove"},
                        {"type": "contains", "target": "kettle"},
                    ],
                },
            ],
        }
        result = self.agent.execute_memory_tool("CompressExperience", params)
        self.assertTrue(result.success, msg=result.message)
        self.assertEqual(result.indices, ["k1"])
        # graph indexed
        self.assertIn("kitchen", self.agent.context_db.list_entities())
        self.assertEqual(len(self.agent.context_db._edges), 2)

    def test_query_graph_round_trip(self):
        # First compress with structure
        self.agent.execute_memory_tool("CompressExperience", {
            "summary": "ix",
            "db_blocks": [
                {"db_index": "k1", "db_content": "kitchen body",
                 "entity": "kitchen",
                 "relations": [{"type": "contains", "target": "stove"}]},
            ],
        })
        # Then add another node via a second compression
        # Repopulate messages so _execute_compress doesn't bail with len<=2
        self.agent.messages = [
            self.agent.messages[0],  # system
            self.agent.messages[1],  # task
            {"role": "assistant", "content": "..."},
            {"role": "user", "content": "obs2"},
        ]
        self.agent.execute_memory_tool("CompressExperience", {
            "summary": "ix2",
            "db_blocks": [
                {"db_index": "k2", "db_content": "stove body", "entity": "stove"},
            ],
        })
        # Now query
        result = self.agent.execute_memory_tool("QueryGraph", {
            "focus": "kitchen", "hops": 1,
        })
        self.assertTrue(result.success, msg=result.message)
        self.assertIn("kitchen", result.message)
        self.assertIn("stove", result.message)
        self.assertIn("--contains-->", result.message)

    def test_query_graph_missing_focus_returns_error_with_known_entities(self):
        self.agent.execute_memory_tool("CompressExperience", {
            "summary": "ix",
            "db_blocks": [
                {"db_index": "k1", "db_content": "...", "entity": "kitchen"},
            ],
        })
        result = self.agent.execute_memory_tool("QueryGraph", {"focus": "bedroom"})
        self.assertFalse(result.success)
        self.assertIn("kitchen", result.message)

    def test_query_graph_rejects_non_graph_mode(self):
        # Construct an agent in lossless_db mode and confirm QueryGraph errors out
        class _LosslessHarness(MemoryAgentMixin):
            def __init__(self):
                self.token_manager = _StubTokenManager()
                self.messages = []
                self.step = 0
                self.init_memory(compression_mode="lossless_db")
        a = _LosslessHarness()
        result = a.execute_memory_tool("QueryGraph", {"focus": "x"})
        self.assertFalse(result.success)
        self.assertIn("graph_db", result.message)

    def test_compress_invalid_relation_returns_validation_error(self):
        # relation without 'target' should fail validation cleanly
        result = self.agent.execute_memory_tool("CompressExperience", {
            "summary": "ix",
            "db_blocks": [
                {"db_index": "k1", "db_content": "...",
                 "entity": "kitchen",
                 "relations": [{"type": "contains"}]},  # missing target
            ],
        })
        self.assertFalse(result.success)
        self.assertIn("target", result.message)

    def test_compress_without_graph_fields_still_works(self):
        # graph_db mode tolerates blocks without entity/relations.
        result = self.agent.execute_memory_tool("CompressExperience", {
            "summary": "ix",
            "db_blocks": [
                {"db_index": "k1", "db_content": "plain content"},
            ],
        })
        self.assertTrue(result.success, msg=result.message)
        self.assertEqual(self.agent.context_db.list_entities(), [])

    def test_graph_mode_forces_disable_retrieve_false(self):
        # disable_retrieve=True is meaningless in graph_db (QueryGraph is the
        # primary retrieval surface). init_memory must override it so prompt
        # and runtime stay consistent.
        class _Harness(MemoryAgentMixin):
            def __init__(self):
                self.token_manager = _StubTokenManager()
                self.messages = [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "t"},
                    {"role": "assistant", "content": "a"},
                    {"role": "user", "content": "o"},
                ]
                self.step = 0
                self.init_memory(compression_mode="graph_db", disable_retrieve=True)

        a = _Harness()
        self.assertFalse(a.disable_retrieve)
        # And QueryGraph + ReadExperience actually work end-to-end
        a.execute_memory_tool("CompressExperience", {
            "summary": "ix",
            "db_blocks": [{"db_index": "k1", "db_content": "...", "entity": "kitchen"}],
        })
        r = a.execute_memory_tool("QueryGraph", {"focus": "kitchen"})
        self.assertTrue(r.success, msg=r.message)
        r2 = a.execute_memory_tool("ReadExperience", {"db_index": "k1"})
        self.assertTrue(r2.success, msg=r2.message)

    def test_lossless_mode_preserves_disable_retrieve(self):
        # Sanity: the override only applies to graph_db, not other modes.
        class _Harness(MemoryAgentMixin):
            def __init__(self):
                self.token_manager = _StubTokenManager()
                self.messages = []
                self.step = 0
                self.init_memory(compression_mode="lossless_db", disable_retrieve=True)
        a = _Harness()
        self.assertTrue(a.disable_retrieve)

    def test_init_memory_with_edge_schema_propagates_to_db(self):
        class _Harness(MemoryAgentMixin):
            def __init__(self):
                self.token_manager = _StubTokenManager()
                self.messages = []
                self.step = 0
                self.init_memory(
                    compression_mode="graph_db",
                    edge_schema=["contains", "near"],
                )
        a = _Harness()
        self.assertEqual(a.context_db._edge_schema, {"contains", "near"})

    def test_graph_prompt_includes_schema_when_set(self):
        from src.agents.memory.prompts import get_memory_tools_prompt_graph
        prompt = get_memory_tools_prompt_graph(
            tool_call_format="xml",
            edge_schema=["contains", "located_in"],
        )
        self.assertIn("EDGE TYPE VOCABULARY", prompt)
        self.assertIn("contains", prompt)
        self.assertIn("located_in", prompt)
        # Without schema, no vocabulary clause
        prompt_open = get_memory_tools_prompt_graph(tool_call_format="xml")
        self.assertNotIn("EDGE TYPE VOCABULARY", prompt_open)

    def test_context_status_lists_entities_in_graph_mode(self):
        self.agent.execute_memory_tool("CompressExperience", {
            "summary": "ix",
            "db_blocks": [
                {"db_index": "k1", "db_content": "...", "entity": "kitchen"},
            ],
        })
        status = self.agent.get_context_status()
        self.assertIn("Indexed entities", status)
        self.assertIn("kitchen", status)


if __name__ == "__main__":
    unittest.main()
