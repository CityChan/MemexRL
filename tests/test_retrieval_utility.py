"""
Unit tests for the retrieval utility bonus in MemoryEfficiencyShaper.

The utility signal rewards retrievals (ReadExperience / QueryGraph) whose
key (db_index or focus entity) appears downstream — either inside a
non-memory tool call's arguments or in the model's response text — within
util_window_steps.

Run with:
    python -m unittest tests.test_retrieval_utility -v
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.agents.agent import Step, Trajectory
from src.parser.tool_parser import ParseResult
from src.tools.tool_base import ToolCall
from src.rewards.shapers.memory_efficiency_shaper import (
    MemoryEfficiencyShaper,
    MEMORY_TOOLS,
    RETRIEVAL_TOOLS,
)


def _step(model_response: str = "", tool_calls=None) -> Step:
    pr = ParseResult(
        tool_calls=tool_calls or [],
        format_errors=[],
        had_tool_attempt=bool(tool_calls),
    )
    return Step(
        chat_completions=[],
        observation=None,
        action=None,
        model_response=model_response,
        parse_result=pr,
    )


def _traj(steps) -> Trajectory:
    return Trajectory(name="test", task=None, steps=steps)


def _shaper(**overrides) -> MemoryEfficiencyShaper:
    cfg = {
        'lambda_ctx': 0.0,
        'lambda_red': 0.0,
        'lambda_format': 0.0,
        'lambda_util': 1.0,
        'enable_context_penalty': False,
        'enable_redundancy_penalty': False,
        'enable_format_penalty': False,
        'enable_utility_reward': True,
        'util_window_steps': 5,
    }
    cfg.update(overrides)
    return MemoryEfficiencyShaper(cfg)


class TestRetrievalToolsConstant(unittest.TestCase):
    def test_query_graph_in_memory_tools(self):
        # QueryGraph must be excluded from redundancy & utility-downstream
        # calculations, otherwise QueryGraph -> QueryGraph would farm bonus.
        self.assertIn("QueryGraph", MEMORY_TOOLS)
        self.assertIn("QueryGraph", RETRIEVAL_TOOLS)
        self.assertNotIn("CompressExperience", RETRIEVAL_TOOLS)


class TestRetrievalUtilityReward(unittest.TestCase):

    def test_read_followed_by_env_action_with_key_counts_as_used(self):
        steps = [
            _step(tool_calls=[ToolCall("ReadExperience", {"db_index": "ctx_kitchen_001"})]),
            _step(tool_calls=[ToolCall("execute_action",
                                        {"action": "go to ctx_kitchen_001 zone"})]),
        ]
        shaped, info = _shaper().shape(0.0, _traj(steps), env=None)
        u = info['penalties']['retrieval_utility']
        self.assertEqual(u['total_retrievals'], 1)
        self.assertEqual(u['used_retrievals'], 1)
        self.assertAlmostEqual(u['utility_ratio'], 1.0)
        self.assertGreater(shaped, 0.0)

    def test_query_graph_followed_by_response_mentioning_focus_counts_as_used(self):
        steps = [
            _step(tool_calls=[ToolCall("QueryGraph", {"focus": "kitchen", "hops": 2})]),
            _step(model_response="ok the kitchen has a stove. let me go check.",
                  tool_calls=[ToolCall("execute_action", {"action": "go north"})]),
        ]
        _, info = _shaper().shape(0.0, _traj(steps), env=None)
        u = info['penalties']['retrieval_utility']
        self.assertEqual(u['used_retrievals'], 1)

    def test_unused_retrieval_does_not_score(self):
        # ReadExperience(ctx_X) but neither subsequent args nor response
        # contains "ctx_X"
        steps = [
            _step(tool_calls=[ToolCall("ReadExperience", {"db_index": "ctx_X"})]),
            _step(model_response="thinking about something else",
                  tool_calls=[ToolCall("execute_action", {"action": "look around"})]),
        ]
        _, info = _shaper().shape(0.0, _traj(steps), env=None)
        u = info['penalties']['retrieval_utility']
        self.assertEqual(u['total_retrievals'], 1)
        self.assertEqual(u['used_retrievals'], 0)
        self.assertEqual(u['utility_ratio'], 0.0)

    def test_downstream_memory_tool_call_does_not_count(self):
        # Mention in another QueryGraph/ReadExperience must NOT count —
        # otherwise policy can farm by chaining memory tools.
        steps = [
            _step(tool_calls=[ToolCall("QueryGraph", {"focus": "kitchen"})]),
            _step(tool_calls=[ToolCall("QueryGraph", {"focus": "kitchen"})]),
            _step(tool_calls=[ToolCall("ReadExperience", {"db_index": "kitchen_blob"})]),
        ]
        _, info = _shaper().shape(0.0, _traj(steps), env=None)
        u = info['penalties']['retrieval_utility']
        # Three retrievals, NONE has a non-memory consumer
        self.assertEqual(u['total_retrievals'], 3)
        self.assertEqual(u['used_retrievals'], 0)

    def test_window_limit_excludes_late_use(self):
        # Use a small window: utility expires before the env action
        steps = [
            _step(tool_calls=[ToolCall("ReadExperience", {"db_index": "ctx_X"})]),
            _step(tool_calls=[ToolCall("execute_action", {"action": "noop"})]),
            _step(tool_calls=[ToolCall("execute_action", {"action": "noop"})]),
            _step(tool_calls=[ToolCall("execute_action", {"action": "use ctx_X"})]),
        ]
        # Window=2 means only steps 1 and 2 are checked → no match
        _, info = _shaper().shape(0.0, _traj(steps), env=None,
                                  )  # default util_window_steps=5; override below
        u_default = info['penalties']['retrieval_utility']
        self.assertEqual(u_default['used_retrievals'], 1)

        _, info_small = _shaper(util_window_steps=2).shape(0.0, _traj(steps), env=None)
        u_small = info_small['penalties']['retrieval_utility']
        self.assertEqual(u_small['used_retrievals'], 0)

    def test_disabled_by_default_in_baseline_config(self):
        # Default config (enable_utility_reward absent) must NOT affect reward.
        cfg = {'enable_utility_reward': False}
        shaper = MemoryEfficiencyShaper(cfg)
        steps = [
            _step(tool_calls=[ToolCall("ReadExperience", {"db_index": "X"})]),
            _step(tool_calls=[ToolCall("env_tool", {"x": "X"})]),
        ]
        shaped, info = shaper.shape(0.0, _traj(steps), env=None)
        self.assertNotIn('retrieval_utility', info['penalties'])
        self.assertEqual(shaped, 0.0)

    def test_ratio_normalization_caps_bonus(self):
        # Many retrievals, mostly unused → ratio < 1, bonus < lambda_util
        retrieval_step = _step(tool_calls=[ToolCall("ReadExperience", {"db_index": "X"})])
        useless_step = _step(tool_calls=[ToolCall("env_tool", {"foo": "bar"})])
        steps = [retrieval_step, useless_step,
                 retrieval_step, useless_step,
                 _step(tool_calls=[ToolCall("ReadExperience", {"db_index": "USED_KEY"})]),
                 _step(tool_calls=[ToolCall("env_tool", {"input": "USED_KEY here"})])]
        _, info = _shaper(lambda_util=1.0).shape(0.0, _traj(steps), env=None)
        u = info['penalties']['retrieval_utility']
        # 3 retrievals, 1 used → ratio 1/3
        self.assertEqual(u['total_retrievals'], 3)
        self.assertEqual(u['used_retrievals'], 1)
        self.assertAlmostEqual(u['utility_ratio'], 1.0 / 3.0)
        self.assertAlmostEqual(u['retrieval_utility_bonus'], 1.0 / 3.0)

    def test_no_retrievals_no_bonus(self):
        steps = [_step(tool_calls=[ToolCall("env_tool", {"x": 1})])]
        _, info = _shaper().shape(0.0, _traj(steps), env=None)
        u = info['penalties']['retrieval_utility']
        self.assertEqual(u['total_retrievals'], 0)
        self.assertEqual(u['retrieval_utility_bonus'], 0.0)

    def test_empty_or_missing_key_skipped(self):
        # ReadExperience with missing db_index → not counted as a retrieval.
        steps = [
            _step(tool_calls=[ToolCall("ReadExperience", {})]),
            _step(tool_calls=[ToolCall("ReadExperience", {"db_index": ""})]),
            _step(tool_calls=[ToolCall("env_tool", {"x": 1})]),
        ]
        _, info = _shaper().shape(0.0, _traj(steps), env=None)
        u = info['penalties']['retrieval_utility']
        self.assertEqual(u['total_retrievals'], 0)


if __name__ == "__main__":
    unittest.main()
