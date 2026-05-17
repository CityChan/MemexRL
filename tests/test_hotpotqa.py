"""
Unit tests for HotpotQA env + agent factory.

Uses a handcrafted in-memory example so the test does not depend on the
HotpotQA dataset being downloaded.

Run with:
    python -m unittest tests.test_hotpotqa -v
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.environments.hotpotqa import HotpotQAEnv, get_hotpotqa_tools
from src.agents.hotpotqa import HotpotQAAgent, HotpotQAAgentWithMemory
from src.data.hotpotqa import (
    _normalize_example,
    exact_match_score,
    f1_score,
)


# A minimal HotpotQA-shaped example. Bridge question: "Who founded the
# company that owns Wikipedia?" -> "Jimmy Wales".
SAMPLE_RAW = {
    "_id": "test_001",
    "question": "Who founded the company that owns Wikipedia?",
    "answer": "Jimmy Wales",
    "type": "bridge",
    "level": "medium",
    "supporting_facts": [["Wikipedia", 1], ["Wikimedia Foundation", 0]],
    "context": [
        ["Wikipedia", [
            "Wikipedia is a free online encyclopedia.",
            "It is owned by the Wikimedia Foundation.",
            "It launched on January 15, 2001.",
        ]],
        ["Wikimedia Foundation", [
            "The Wikimedia Foundation was co-founded by Jimmy Wales in 2003.",
            "It is a nonprofit organization based in San Francisco.",
        ]],
        ["Linux", [
            "Linux is a family of Unix-like operating systems.",
        ]],
        ["Python (programming language)", [
            "Python is an interpreted, high-level programming language.",
        ]],
    ],
}


class TestHotpotQADataNormalization(unittest.TestCase):

    def test_normalize_extracts_passages_and_supporting_facts(self):
        task = _normalize_example(SAMPLE_RAW)
        self.assertEqual(task["question"], SAMPLE_RAW["question"])
        self.assertEqual(task["answer"], "Jimmy Wales")
        self.assertEqual(len(task["passages"]), 4)
        gold = [p for p in task["passages"] if p["is_gold"]]
        self.assertEqual({p["title"] for p in gold},
                         {"Wikipedia", "Wikimedia Foundation"})
        self.assertEqual(len(task["supporting_facts"]), 2)

    def test_em_and_f1_against_gold(self):
        self.assertEqual(exact_match_score("Jimmy Wales", "Jimmy Wales"), 1.0)
        # Normalization handles articles/case/punctuation
        self.assertEqual(exact_match_score("the Jimmy Wales.", "jimmy wales"), 1.0)
        # Wrong answer
        self.assertEqual(exact_match_score("Larry Sanger", "Jimmy Wales"), 0.0)
        # F1 partial credit
        self.assertGreater(f1_score("Jimmy", "Jimmy Wales"), 0.0)
        self.assertLess(f1_score("Jimmy", "Jimmy Wales"), 1.0)


class TestHotpotQAEnv(unittest.TestCase):

    def setUp(self):
        self.task = _normalize_example(SAMPLE_RAW)
        self.env = HotpotQAEnv(task=self.task)

    def test_reset_returns_question_and_titles(self):
        obs, info = self.env.reset()
        self.assertEqual(obs["question"], self.task["question"])
        self.assertEqual(set(obs["passage_titles"]),
                         {p["title"] for p in self.task["passages"]})
        self.assertIn("Question:", obs["task_description"])
        self.assertIn("Wikipedia", obs["task_description"])
        self.assertEqual(info["task_id"], "test_001")

    def test_read_passage_known_title(self):
        self.env.reset()
        obs, reward, done, info = self.env.step({
            "function": {"name": "read_passage",
                         "arguments": {"title": "Wikimedia Foundation"}},
        })
        self.assertIn("Jimmy Wales", obs["observation"])
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertIn("Wikimedia Foundation", self.env.read_titles)
        # interaction_history records is_gold=True
        self.assertTrue(self.env.interaction_history[-1]["is_gold"])

    def test_read_passage_unknown_title_returns_friendly_error(self):
        self.env.reset()
        obs, _, done, _ = self.env.step({
            "function": {"name": "read_passage",
                         "arguments": {"title": "Does Not Exist"}},
        })
        self.assertIn("No passage with title", obs["observation"])
        # Should list available titles
        self.assertIn("Wikipedia", obs["observation"])
        self.assertFalse(done)

    def test_list_passages(self):
        self.env.reset()
        obs, _, _, _ = self.env.step({
            "function": {"name": "list_passages", "arguments": {}},
        })
        self.assertIn("Wikipedia", obs["observation"])
        self.assertIn("Wikimedia Foundation", obs["observation"])

    def test_finish_records_answer_and_reward(self):
        self.env.reset()
        # Read gold passages
        self.env.step({"function": {"name": "read_passage",
                                      "arguments": {"title": "Wikipedia"}}})
        self.env.step({"function": {"name": "read_passage",
                                      "arguments": {"title": "Wikimedia Foundation"}}})
        obs, _, done, _ = self.env.step({
            "function": {"name": "finish",
                         "arguments": {"answer": "Jimmy Wales"}},
        })
        self.assertTrue(done)
        self.assertEqual(self.env.final_response, "Jimmy Wales")
        self.assertEqual(self.env.compute_final_reward(), 1.0)

    def test_finish_with_wrong_answer(self):
        self.env.reset()
        self.env.step({"function": {"name": "finish",
                                      "arguments": {"answer": "Larry Sanger"}}})
        self.assertEqual(self.env.compute_final_reward(), 0.0)

    def test_max_steps_terminates(self):
        env = HotpotQAEnv(task=self.task, max_steps=2)
        env.reset()
        env.step({"function": {"name": "list_passages", "arguments": {}}})
        _, _, done, _ = env.step({"function": {"name": "list_passages", "arguments": {}}})
        self.assertTrue(done)

    def test_unknown_tool_returns_error_does_not_terminate(self):
        self.env.reset()
        obs, _, done, _ = self.env.step({
            "function": {"name": "foo_tool", "arguments": {}},
        })
        self.assertIn("Unknown tool", obs["observation"])
        self.assertFalse(done)

    def test_from_dict_round_trip(self):
        env = HotpotQAEnv.from_dict(self.task)
        env.reset()
        self.assertEqual(env.task["task_id"], "test_001")

    def test_get_tools_returns_three_tools(self):
        tools = get_hotpotqa_tools()
        names = {t["function"]["name"] for t in tools}
        self.assertEqual(names, {"read_passage", "list_passages", "finish"})


class TestHotpotQAAgentFactory(unittest.TestCase):

    def test_baseline_agent_has_no_memory_prompt(self):
        agent = HotpotQAAgent(tool_call_format="xml")
        sp = agent.system_prompt
        self.assertIn("research agent", sp)
        self.assertIn("read_passage", sp)
        self.assertIn("finish", sp)
        # No memory tools
        self.assertNotIn("CompressExperience", sp)
        self.assertNotIn("QueryGraph", sp)

    def test_lossless_db_agent_has_compress_read_only(self):
        agent = HotpotQAAgentWithMemory(
            tool_call_format="xml", compression_mode="lossless_db",
        )
        sp = agent.system_prompt
        self.assertIn("CompressExperience", sp)
        self.assertIn("ReadExperience", sp)
        self.assertNotIn("QueryGraph", sp)
        # No HotpotQA-specific graph guidance in non-graph mode
        self.assertNotIn("MEMORY MANAGEMENT (graph_db) FOR HotpotQA", sp)

    def test_graph_db_agent_has_graph_tools_and_hotpotqa_guidance(self):
        agent = HotpotQAAgentWithMemory(
            tool_call_format="xml", compression_mode="graph_db",
        )
        sp = agent.system_prompt
        # Graph mode tools
        self.assertIn("CompressExperience", sp)
        self.assertIn("ReadExperience", sp)
        self.assertIn("QueryGraph", sp)
        self.assertIn("entity", sp)
        self.assertIn("relations", sp)
        # HotpotQA-specific hint about title-as-entity
        self.assertIn("MEMORY MANAGEMENT (graph_db) FOR HotpotQA", sp)
        self.assertIn("Wikipedia titles", sp)
        # Backend
        from src.database.graph_context_database import GraphContextDatabase
        self.assertIsInstance(agent.context_db, GraphContextDatabase)


if __name__ == "__main__":
    unittest.main()
