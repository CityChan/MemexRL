"""
HotpotQA environment (distractor setting).

Each episode:
- One HotpotQA question + 10 candidate Wikipedia paragraphs (2 gold + 8
  distractors), each tagged by article title.
- Tools the agent can call:
    read_passage(title)  -> returns the passage text for that title, or
                            an error listing available titles
    finish(answer)       -> submits the final answer; ends episode
- Reward (computed by `compute_final_reward`):
    EM (1.0/0.0) against gold answer, with F1 surfaced in info.

The agent learns *which* titles to read to answer the question. With
graph_db memory, paragraph titles map naturally onto entity names and
cross-paragraph mentions can be encoded as relations.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Optional

from src.environments.base.base_env import BaseEnv


HOTPOTQA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_passage",
            "description": (
                "Read the full Wikipedia passage with the given title from the "
                "candidate set for this question. Use list_passages() if you do "
                "not yet know which titles are available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Exact Wikipedia article title (case-sensitive).",
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_passages",
            "description": "List all candidate passage titles for this question.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit the final answer to the question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Final natural-language answer.",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]


def get_hotpotqa_tools() -> list:
    return HOTPOTQA_TOOLS.copy()


def _format_passage(title: str, sentences: list[str]) -> str:
    body = " ".join(sentences) if sentences else "(empty)"
    return f"Title: {title}\n{body}"


class HotpotQAEnv(BaseEnv):
    """Single-question HotpotQA distractor-setting environment."""

    def __init__(
        self,
        task: Optional[dict] = None,
        reward_fn: Optional[Callable] = None,
        max_steps: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task = task or {}
        self.reward_fn = reward_fn
        self.max_steps = max_steps

        # State
        self.current_step = 0
        self.done = False
        self.final_response: Optional[str] = None
        self.interaction_history: list[dict] = []
        self.read_titles: set[str] = set()

        # Pre-index passages by title for O(1) lookup
        self._title_to_passage: dict[str, dict] = {}
        for p in self.task.get("passages", []):
            t = p.get("title")
            if isinstance(t, str):
                self._title_to_passage[t] = p

    # ------------------------------------------------------------------
    # BaseEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[dict, dict]:
        self.current_step = 0
        self.done = False
        self.final_response = None
        self.interaction_history = []
        self.read_titles = set()

        question = self.task.get("question", "")
        titles = sorted(self._title_to_passage.keys())

        observation = {
            "question": question,
            "passage_titles": titles,
            "task_description": self._build_task_description(question, titles),
        }
        info = {
            "task_id": self.task.get("task_id", ""),
            "type": self.task.get("type", ""),
            "level": self.task.get("level", ""),
            "max_steps": self.max_steps,
            "tools_json": get_hotpotqa_tools(),
        }
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        self.current_step += 1
        func_info = action.get("function", {}) if isinstance(action, dict) else {}
        tool_name = func_info.get("name", "")
        tool_args = func_info.get("arguments", {})
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {}

        if tool_name == "finish":
            self.final_response = str(tool_args.get("answer", ""))
            self.done = True
            return (
                {"observation": f"[Final answer recorded: {self.final_response}]"},
                0.0,
                True,
                self._get_info(),
            )

        if tool_name == "list_passages":
            titles = sorted(self._title_to_passage.keys())
            obs = "Candidate passage titles:\n" + "\n".join(f"- {t}" for t in titles)
            self.interaction_history.append(
                {"step": self.current_step, "tool": "list_passages"}
            )
            return ({"observation": obs}, 0.0, self._check_done(), self._get_info())

        if tool_name == "read_passage":
            title = str(tool_args.get("title", "")).strip()
            passage = self._title_to_passage.get(title)
            if passage is None:
                # Friendly error listing available titles
                available = sorted(self._title_to_passage.keys())
                obs = (
                    f"[No passage with title \"{title}\". Available titles ({len(available)}):]\n"
                    + "\n".join(f"- {t}" for t in available)
                )
            else:
                obs = _format_passage(title, passage["sentences"])
                self.read_titles.add(title)
                self.interaction_history.append({
                    "step": self.current_step,
                    "tool": "read_passage",
                    "title": title,
                    "is_gold": passage.get("is_gold", False),
                })
            return ({"observation": obs}, 0.0, self._check_done(), self._get_info())

        # Unknown tool
        return (
            {"observation": f"[Unknown tool '{tool_name}'. Use read_passage, list_passages, or finish.]"},
            0.0,
            self._check_done(),
            self._get_info(),
        )

    def compute_final_reward(self) -> float:
        # Prefer caller-supplied reward_fn (for shaping); fall back to plain EM.
        if self.reward_fn is not None:
            task_info = self.task.copy()
            task_info["interaction_history"] = self.interaction_history
            task_info["read_titles"] = sorted(self.read_titles)
            try:
                out = self.reward_fn(task_info=task_info, action=self.final_response or "")
                if hasattr(out, "reward"):
                    return out.reward
                return float(out)
            except Exception:
                pass

        from src.data.hotpotqa import exact_match_score
        gold = self.task.get("answer", "")
        pred = self.final_response or ""
        return float(exact_match_score(pred, gold))

    def _check_done(self) -> bool:
        if self.current_step >= self.max_steps:
            self.done = True
        return self.done

    def _get_info(self) -> dict:
        return {
            "task_id": self.task.get("task_id", ""),
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "done": self.done,
            "n_titles_read": len(self.read_titles),
        }

    def get_tools(self) -> list[dict]:
        return get_hotpotqa_tools()

    @staticmethod
    def from_dict(info: dict) -> "HotpotQAEnv":
        # info should already be a normalized HotpotQA task dict
        # (see src.data.hotpotqa._normalize_example)
        return HotpotQAEnv(
            task=info,
            reward_fn=info.get("reward_fn"),
            max_steps=info.get("max_steps", 30),
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_task_description(question: str, titles: list[str]) -> str:
        title_list = "\n".join(f"- {t}" for t in titles)
        return (
            "Question: " + question + "\n\n"
            f"You have {len(titles)} candidate Wikipedia passages. Use "
            "`read_passage(title=...)` to read any of them, or `list_passages()` "
            "to recall the titles. When you can answer the question, call "
            "`finish(answer=...)`.\n\n"
            "Candidate titles:\n" + title_list
        )
