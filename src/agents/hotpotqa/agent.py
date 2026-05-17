"""
HotpotQA agent (base + memory-enabled variant).

The agent ships with two memory configurations of interest:

    HotpotQAAgentWithMemory(compression_mode="lossless_db")  # baseline
    HotpotQAAgentWithMemory(compression_mode="graph_db")     # this work

In graph_db mode, the suggested entity vocabulary is the set of paragraph
titles (which are Wikipedia article titles) plus answer candidates. Edges
naturally express "<title_A> --mentions--> <entity>" links across passages,
which is exactly the structure that lets QueryGraph(focus=<entity>)
recall every passage that touches an entity in one call.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from src.agents.tool_agent import ToolAgent, ToolAgentWithMemory


REASONING_INSTRUCTION = """
<IMPORTANT>
- Provide a brief reasoning step BEFORE every tool call.
</IMPORTANT>
"""

HOTPOTQA_SYSTEM_PROMPT_BASE = """You are a careful research agent answering a multi-hop question.

# Task
You are given a question and a small set of candidate Wikipedia passages
(typically 10: 2 of them contain the answer, the rest are distractors).
Read passages selectively, gather the evidence you need, then submit a
final answer.

# Available Tools

1. **read_passage** - Read one passage by its exact Wikipedia title.
2. **list_passages** - List all candidate passage titles for the question.
3. **finish** - Submit the final answer.

# Tips
- The question type is either "bridge" (entity B connects facts about A and C)
  or "comparison" (compare two entities). For bridge, identify the bridge
  entity from one passage, then look it up in another.
- The answer is usually a short phrase (a name, year, place, or number).
- Do not write more than the answer itself in `finish(answer=...)`.

##############################################################################
#                         MANDATORY REQUIREMENTS                              #
##############################################################################

>>> REQUIREMENT 1: YOU MUST CALL A TOOL IN EVERY RESPONSE <<<
- Every response MUST contain a tool call.
- Plain text responses without a tool call will be REJECTED.

>>> REQUIREMENT 2: USE finish ONLY ONCE YOU CAN ANSWER <<<
- Only call `finish` once you have enough evidence.
- Calling finish without supporting reads is risky.
""" + REASONING_INSTRUCTION


HOTPOTQA_TOOL_CALL_FORMAT_XML = """# Tool Call Format
Use the following XML format to call tools:

<function=tool_name>
<parameter=param_name>param_value</parameter>
</function>

Examples:
<function=read_passage>
<parameter=title>Albert Einstein</parameter>
</function>

<function=list_passages>
</function>

<function=finish>
<parameter=answer>1879</parameter>
</function>
"""


HOTPOTQA_TOOL_CALL_FORMAT_QWEN = """# Tool Call Format
Use the following JSON format inside <tool_call> tags:

<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1"}}
</tool_call>

Examples:
<tool_call>
{"name": "read_passage", "arguments": {"title": "Albert Einstein"}}
</tool_call>

<tool_call>
{"name": "list_passages", "arguments": {}}
</tool_call>

<tool_call>
{"name": "finish", "arguments": {"answer": "1879"}}
</tool_call>
"""


# Memory guidance for graph_db mode: hint that titles are good entities.
HOTPOTQA_GRAPH_MEMORY_GUIDANCE = """
##############################################################################
#                MEMORY MANAGEMENT (graph_db) FOR HotpotQA                   #
##############################################################################

When you call CompressExperience in this task, the most useful entity names
are usually:
- The Wikipedia titles of passages you read (one entity per passage you keep).
- The "bridge" entity that links the two gold passages (for bridge questions).
- The candidate answers when comparing two entities (for comparison questions).

Useful relation types:
- "mentions"      : passage_title -> any other entity it discusses
- "answer_of"     : entity -> question (use sparingly, once you have the answer)
- "compared_with" : entity -> other entity (for comparison questions)

After compressing, prefer QueryGraph(focus=<entity>, hops=1) to recall
every passage that touches a given entity in one call, instead of
re-reading passages.
"""


def _format_hotpotqa_observation(observation: Any) -> str:
    if isinstance(observation, dict):
        if "task_description" in observation:
            return observation["task_description"]
        if "observation" in observation:
            return str(observation["observation"])
        return json.dumps(observation, ensure_ascii=False, indent=2)
    return str(observation)


def _get_hotpotqa_system_prompt(tool_call_format: str) -> str:
    if tool_call_format == "qwen":
        return HOTPOTQA_SYSTEM_PROMPT_BASE + HOTPOTQA_TOOL_CALL_FORMAT_QWEN
    return HOTPOTQA_SYSTEM_PROMPT_BASE + HOTPOTQA_TOOL_CALL_FORMAT_XML


def _get_tool_format_suffix(tool_call_format: str) -> str:
    if tool_call_format == "qwen":
        return HOTPOTQA_TOOL_CALL_FORMAT_QWEN
    return HOTPOTQA_TOOL_CALL_FORMAT_XML


class HotpotQAAgent(ToolAgent):
    """HotpotQA agent without memory."""

    def __init__(
        self,
        tool_call_format: str = "xml",
        model_name: Optional[str] = None,
    ):
        if model_name and "qwen" in model_name.lower():
            tool_call_format = "qwen"
        system_prompt = _get_hotpotqa_system_prompt(tool_call_format)
        super().__init__(
            system_prompt=system_prompt,
            tool_call_format=tool_call_format,
            agent_name="hotpotqa_agent",
            model_name=model_name,
            observation_formatter=_format_hotpotqa_observation,
        )


class HotpotQAAgentWithMemory(ToolAgentWithMemory):
    """HotpotQA agent with the unified memory tool stack.

    Set compression_mode in {"lossless_db", "graph_db", "rag", "lossy"}.
    For graph_db, an additional task-specific guidance section is appended
    to the tool format suffix to nudge the policy toward sensible entity
    names (paragraph titles).
    """

    def __init__(
        self,
        tool_call_format: str = "xml",
        model_name: Optional[str] = None,
        compression_mode: str = "graph_db",
        context_db: Any = None,
        db_path: str = "hotpotqa_experience.sqlite",
        context_length_threshold: int = 8000,
        auto_compress_prompt: bool = True,
        disable_retrieve: bool = False,
        max_summary_tokens: int = 0,
    ):
        if model_name and "qwen" in model_name.lower():
            tool_call_format = "qwen"

        tool_format_suffix = _get_tool_format_suffix(tool_call_format)
        if compression_mode == "graph_db":
            full_tool_format_suffix = HOTPOTQA_GRAPH_MEMORY_GUIDANCE + tool_format_suffix
        else:
            full_tool_format_suffix = tool_format_suffix

        super().__init__(
            system_prompt=HOTPOTQA_SYSTEM_PROMPT_BASE,
            tool_call_format=tool_call_format,
            agent_name="hotpotqa_agent",
            model_name=model_name,
            observation_formatter=_format_hotpotqa_observation,
            compression_mode=compression_mode,
            context_db=context_db,
            db_path=db_path,
            context_length_threshold=context_length_threshold,
            auto_compress_prompt=auto_compress_prompt,
            disable_retrieve=disable_retrieve,
            max_summary_tokens=max_summary_tokens,
            tool_format_suffix=full_tool_format_suffix,
        )
