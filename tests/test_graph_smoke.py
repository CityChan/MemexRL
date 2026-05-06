"""
End-to-end (offline) smoke test for graph_db memory mode.

Exercises the full agent-factory path WITHOUT spinning up vLLM, Slime,
TextWorld, or any other heavy dependency:

- Construct ToolAgentWithMemory(compression_mode="graph_db", ...)
- Verify the system prompt embeds the graph-mode preamble + tool descs
- Drive the agent via the same call sequence the engine uses:
    update_from_env(observation)
      -> update_from_model(model_response_with_tool_call)
        -> agent.is_memory_tool(...) + agent.execute_memory_tool(...)
      -> update_from_env(tool_result)
- Round-trip: CompressExperience writes graph -> QueryGraph reads subgraph.
- Verify the GraphContextDatabase ends up with the expected nodes + edges.

Run with:
    python -m unittest tests.test_graph_smoke -v
or:
    python tests/test_graph_smoke.py
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.agents.tool_agent import ToolAgentWithMemory
from src.database.graph_context_database import GraphContextDatabase


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubTokenManager:
    """Minimal token_manager so MemoryAgentMixin._estimate_working_tokens works."""

    def get_working_tokens(self, messages):
        return sum(len(str(m.get("content", ""))) for m in messages) // 4

    def get_total_tokens(self, messages):
        return self.get_working_tokens(messages)


# A minimal task-specific base prompt; real envs (ALFWorld / HotpotQA) supply
# their own. We just need *something* the factory can append to.
_TEST_SYSTEM_PROMPT_BASE = (
    "You are a test agent. Solve the task with the available tools.\n"
    "Available env tool: noop()."
)


def _build_compress_xml(summary: str, db_blocks_json: str) -> str:
    """Emit a CompressExperience tool call in XML format (what the model would output)."""
    return (
        "I have learned about the kitchen layout; compressing now.\n"
        f"<function=CompressExperience>\n"
        f"<parameter=summary>{summary}</parameter>\n"
        f"<parameter=db_blocks>{db_blocks_json}</parameter>\n"
        f"</function>"
    )


def _build_query_graph_xml(focus: str, hops: int = 1) -> str:
    return (
        "I need to recall what is near the kitchen.\n"
        f"<function=QueryGraph>\n"
        f"<parameter=focus>{focus}</parameter>\n"
        f"<parameter=hops>{hops}</parameter>\n"
        f"</function>"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGraphDbAgentSmoke(unittest.TestCase):

    def _make_agent(self, **overrides):
        kwargs = dict(
            system_prompt=_TEST_SYSTEM_PROMPT_BASE,
            tool_call_format="xml",
            agent_name="graph_smoke_agent",
            compression_mode="graph_db",
            context_length_threshold=4000,
        )
        kwargs.update(overrides)
        agent = ToolAgentWithMemory(**kwargs)
        agent.token_manager = _StubTokenManager()
        return agent

    def test_factory_uses_graph_backend_and_prompt(self):
        agent = self._make_agent()

        # Backend
        self.assertEqual(agent.compression_mode, "graph_db")
        self.assertIsInstance(agent.context_db, GraphContextDatabase)

        # System prompt contains: base + graph-mode intro + CompressExperience
        # description (mentions entity/relations) + QueryGraph description.
        sp = agent.system_prompt
        self.assertIn("test agent", sp)  # base preserved
        self.assertIn("THREE OBJECTIVES (graph memory)", sp)
        self.assertIn("CompressExperience", sp)
        self.assertIn("entity", sp)
        self.assertIn("relations", sp)
        self.assertIn("QueryGraph", sp)
        self.assertIn("focus", sp)

        # The first message in the conversation IS the system prompt.
        self.assertEqual(agent.messages[0]["role"], "system")
        self.assertIs(agent.messages[0]["content"], sp)

    def test_full_compress_then_query_round_trip(self):
        agent = self._make_agent()

        # ---- Step 1: First observation (task description) -------------------
        # update_from_env with no current_step (first call) treats this as the
        # task description.
        first_obs = "Task: explore the kitchen and report what you find."
        agent.update_from_env(observation=first_obs, reward=0.0, done=False, info={})
        # Should have system prompt + first observation
        self.assertEqual(len(agent.messages), 2)
        self.assertEqual(agent.messages[1]["role"], "user")
        self.assertIn("Task:", agent.messages[1]["content"])

        # ---- Step 2: Model emits a normal env action (text turn) ------------
        agent.update_from_model("I will look around. <function=noop></function>")
        # Engine would now call env.step(); we simulate the env returning more text.
        agent.update_from_env(
            observation="You see a stove, a kettle on the counter, and a fridge.",
            reward=0.0, done=False, info={},
        )

        # ---- Step 3: Model emits a CompressExperience with graph fields -----
        db_blocks = (
            '['
            '{"db_index":"obs_kitchen_1","db_content":"Kitchen contains stove, kettle, fridge.",'
            '"entity":"kitchen",'
            '"relations":[{"type":"contains","target":"stove"},'
                         '{"type":"contains","target":"kettle"},'
                         '{"type":"contains","target":"fridge"}]},'
            '{"db_index":"obs_kettle_1","db_content":"Stainless-steel kettle on the counter.",'
            '"entity":"kettle",'
            '"relations":[{"type":"located_in","target":"kitchen"},'
                         '{"type":"near","target":"stove"}]}'
            ']'
        )
        summary = (
            "Index map:\\n"
            "- obs_kitchen_1 [entity=kitchen] - kitchen contents\\n"
            "- obs_kettle_1 [entity=kettle] - kettle state\\n"
            "Status: kitchen explored."
        )
        compress_response = _build_compress_xml(summary, db_blocks)
        parse_result = agent.update_from_model(compress_response)

        # Parser should have extracted the CompressExperience call
        self.assertEqual(len(parse_result.tool_calls), 1, msg=parse_result.to_dict())
        tc = parse_result.tool_calls[0]
        self.assertEqual(tc.name, "CompressExperience")

        # Mimic the engine: dispatch memory tool through the agent
        self.assertTrue(agent.is_memory_tool(tc.name))
        result = agent.execute_memory_tool(tc.name, tc.arguments)
        self.assertTrue(result.success, msg=result.message)
        self.assertEqual(set(result.indices), {"obs_kitchen_1", "obs_kettle_1"})

        # Graph state in the database
        db = agent.context_db
        ents = set(db.list_entities())
        # primary entities + targets that don't have their own block are also in the index
        # via _adjacency, but only "owned" entities show in list_entities().
        self.assertIn("kitchen", ents)
        self.assertIn("kettle", ents)
        # Edge count: 3 from kitchen + 2 from kettle = 5
        self.assertEqual(len(db._edges), 5)

        # Compression replaced messages with [system, task, summary]
        self.assertEqual(len(agent.messages), 3)
        self.assertIn("SUMMARY OF YOUR PREVIOUS CONTEXT", agent.messages[2]["content"])

        # ---- Step 4: env returns the memory tool result back as observation -
        agent.update_from_env(observation=result.message, reward=0.0, done=False, info={})

        # The injected [Context Status] should now mention indexed entities
        last_obs = agent.messages[-1]["content"]
        self.assertIn("Indexed entities", last_obs)
        self.assertIn("kitchen", last_obs)

        # ---- Step 5: Model emits QueryGraph(focus="kitchen", hops=2) --------
        query_response = _build_query_graph_xml(focus="kitchen", hops=2)
        parse_result = agent.update_from_model(query_response)
        self.assertEqual(len(parse_result.tool_calls), 1, msg=parse_result.to_dict())
        qc = parse_result.tool_calls[0]
        self.assertEqual(qc.name, "QueryGraph")

        self.assertTrue(agent.is_memory_tool(qc.name))
        qr = agent.execute_memory_tool(qc.name, qc.arguments)
        self.assertTrue(qr.success, msg=qr.message)

        # Subgraph response should mention kitchen (focus) + kettle (neighbour)
        # + edges (--contains--> / --located_in--> / --near-->)
        self.assertIn("kitchen", qr.message)
        self.assertIn("kettle", qr.message)
        self.assertIn("--contains-->", qr.message)
        self.assertIn("Subgraph centered on \"kitchen\"", qr.message)

    def test_graph_mode_query_graph_recognized_after_construction(self):
        """Sanity-check that the agent considers QueryGraph a memory tool
        when constructed via the public factory (not just the bare mixin)."""
        agent = self._make_agent()
        self.assertTrue(agent.is_memory_tool("QueryGraph"))
        self.assertTrue(agent.is_memory_tool("CompressExperience"))
        self.assertTrue(agent.is_memory_tool("ReadExperience"))
        self.assertFalse(agent.is_memory_tool("noop"))

    def test_lossless_db_factory_unchanged(self):
        """Regression: lossless_db agents must NOT advertise QueryGraph
        in the prompt and must reject QueryGraph calls cleanly."""
        agent = self._make_agent(compression_mode="lossless_db")
        agent.token_manager = _StubTokenManager()

        # Prompt: should NOT contain graph-mode intro.
        sp = agent.system_prompt
        self.assertNotIn("THREE OBJECTIVES (graph memory)", sp)
        # But should still contain CompressExperience / ReadExperience descs.
        self.assertIn("CompressExperience", sp)
        self.assertIn("ReadExperience", sp)

        # QueryGraph still registered in is_memory_tool (mixin-level), but
        # _execute_graph_query rejects it because mode != graph_db.
        result = agent.execute_memory_tool("QueryGraph", {"focus": "x"})
        self.assertFalse(result.success)
        self.assertIn("graph_db", result.message)

    def test_graph_mode_lists_entities_in_context_status(self):
        agent = self._make_agent()
        # Seed a node
        result = agent.execute_memory_tool("CompressExperience", {
            "summary": "ix",
            "db_blocks": [
                {"db_index": "k1", "db_content": "...", "entity": "kitchen"},
            ],
        })
        # The above CompressExperience call needs >2 messages to operate;
        # if it failed for that reason, populate fake history first and retry.
        if not result.success and "Not enough messages" in result.message:
            agent.messages.extend([
                {"role": "assistant", "content": "..."},
                {"role": "user", "content": "obs"},
            ])
            result = agent.execute_memory_tool("CompressExperience", {
                "summary": "ix",
                "db_blocks": [
                    {"db_index": "k1", "db_content": "...", "entity": "kitchen"},
                ],
            })
        self.assertTrue(result.success, msg=result.message)

        status = agent.get_context_status()
        self.assertIn("Indexed entities", status)
        self.assertIn("kitchen", status)


if __name__ == "__main__":
    unittest.main()
