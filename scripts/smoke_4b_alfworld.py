"""
End-to-end smoke test: drive one ALFWorld episode with Qwen3-4B and the
real MemexRL agent. Single GPU, no training, no Slime / Ray / Megatron.

Why:
    Cheapest way to validate that on a Vista GH200 idev node, with a
    bare-metal Slime install:
      - PyTorch + 4B model load actually drives the GPU
      - ALFWorld + TextWorld import & game files work on aarch64
      - The MemexRL agent loop (update_from_env -> generate -> parse ->
        is_memory_tool / env.step -> update_from_env) runs end-to-end
      - Compression / retrieval / QueryGraph tools fire correctly when
        the model emits them

Stages (each fail-tolerant):
    1. Imports + CUDA
    2. Load 4B model + tokenizer (transformers, bfloat16)
    3. Locate an ALFWorld game file under $ALFWORLD_DATA
    4. Construct ALFWorldEnv + ALFWorldAgentWithMemory
    5. Run up to MAX_STEPS turns:
         observation -> agent.messages -> model.generate -> response
         agent.update_from_model(response) -> parse_result
         for each tool_call:
             if memory tool: agent.execute_memory_tool(...)
             else:           env.step(wrap_action(tc))
    6. Dump memory state + interaction history

Usage:
    # Defaults assume the install_slime_vista.sh layout
    python scripts/smoke_4b_alfworld.py

    # Switch memory mode
    python scripts/smoke_4b_alfworld.py --mode graph_db

    # Override the model / data paths
    python scripts/smoke_4b_alfworld.py \\
        --model /work/.../models/Qwen3-4B-Thinking-2507 \\
        --alfworld-data $HOME/.alfworld

    # More steps, see model talk longer
    python scripts/smoke_4b_alfworld.py --max-steps 20
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import traceback
from typing import Any, Optional


def _stage(name: str):
    print(f"\n\033[1;36m==== {name} ====\033[0m")


def _ok(msg: str):
    print(f"  \033[1;32m[OK]\033[0m {msg}")


def _fail(msg: str):
    print(f"  \033[1;31m[FAIL]\033[0m {msg}")


def _info(msg: str):
    print(f"  \033[1;33m[..]\033[0m {msg}")


# ---------------------------------------------------------------------------
# Stage 1: CUDA + imports
# ---------------------------------------------------------------------------

def stage_imports() -> bool:
    _stage("Stage 1: imports + CUDA")
    try:
        import torch
        if not torch.cuda.is_available():
            _fail("torch.cuda.is_available() = False")
            return False
        _ok(f"torch {torch.__version__}  cuda {torch.version.cuda}  GPU {torch.cuda.get_device_name(0)}")
        # Need MemexRL on PYTHONPATH
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from src.environments.alfworld.env import ALFWorldEnv  # noqa: F401
        from src.agents.alfworld.agent import ALFWorldAgentWithMemory  # noqa: F401
        _ok("MemexRL src + ALFWorld imports OK")
        return True
    except Exception as e:  # noqa: BLE001
        _fail(f"imports crashed: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Stage 2: model load
# ---------------------------------------------------------------------------

def stage_load_model(model_path: str):
    _stage(f"Stage 2: load {os.path.basename(model_path)}")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not os.path.exists(model_path):
            _fail(f"model dir not found: {model_path}")
            _fail("  download with:")
            _fail(f"    huggingface-cli download Qwen/Qwen3-4B-Thinking-2507 --local-dir {model_path}")
            return None, None
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _ok(f"tokenizer loaded in {time.time()-t0:.1f}s")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        model.eval()
        _ok(f"model loaded in {time.time()-t0:.1f}s, "
            f"GPU mem {torch.cuda.memory_allocated()/1024**3:.1f} GB")
        return model, tok
    except Exception as e:  # noqa: BLE001
        _fail(f"load crashed: {e}")
        traceback.print_exc()
        return None, None


# ---------------------------------------------------------------------------
# Stage 3: find ALFWorld game file
# ---------------------------------------------------------------------------

def stage_find_game(alfworld_data: str) -> Optional[str]:
    _stage("Stage 3: find an ALFWorld game file")
    if not os.path.exists(alfworld_data):
        _fail(f"$ALFWORLD_DATA={alfworld_data} does not exist")
        _fail("  install + download with:")
        _fail("    pip install --user alfworld textworld")
        _fail("    alfworld-download -f")
        return None
    # ALFWorld download lays out games under .../json_2.1.1/{train,valid_seen,...}
    # Each game is a directory containing game.tw-pddl
    candidates = glob.glob(os.path.join(alfworld_data, "**", "*.tw-pddl"), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(alfworld_data, "**", "game.*"), recursive=True)
    if not candidates:
        _fail(f"no .tw-pddl game files found under {alfworld_data}")
        _fail("  did you run `alfworld-download -f`?")
        return None
    game = sorted(candidates)[0]
    _ok(f"using {game}")
    _ok(f"  ({len(candidates)} games available)")
    return game


# ---------------------------------------------------------------------------
# Stage 4: construct env + agent
# ---------------------------------------------------------------------------

def stage_make_env_agent(game_file: str, mode: str, model_name: str):
    _stage(f"Stage 4: build env + agent (compression_mode={mode})")
    try:
        from src.environments.alfworld.env import ALFWorldEnv
        from src.agents.alfworld.agent import ALFWorldAgentWithMemory

        env = ALFWorldEnv(
            task={"game_file": game_file},
            max_steps=20,
            use_process_lock=False,
        )
        _ok(f"env: {type(env).__name__}")

        agent = ALFWorldAgentWithMemory(
            tool_call_format="qwen",   # Qwen3 model => qwen format
            model_name=model_name,
            compression_mode=mode,
            context_length_threshold=8000,
            auto_compress_prompt=True,
        )
        _ok(f"agent: {type(agent).__name__}  compression_mode={agent.compression_mode}")
        _ok(f"system prompt len: {len(agent._base_system_prompt or '')} chars; "
            f"with memory tools: {sum(len(m['content']) for m in agent.messages)} chars in initial messages")
        return env, agent
    except Exception as e:  # noqa: BLE001
        _fail(f"construction crashed: {e}")
        traceback.print_exc()
        return None, None


# ---------------------------------------------------------------------------
# Stage 5: run episode
# ---------------------------------------------------------------------------

def _generate(
    model,
    tok,
    messages: list[dict],
    max_new: int = 2048,
    enable_thinking: bool = True,
) -> str:
    import torch
    # Qwen3-Thinking models default to producing a long <think>...</think>
    # block before any actual tool call. With small max_new this gets cut
    # off mid-thinking and we never see a tool call. Two knobs here:
    #   - max_new: large enough (>=2048) for thinking + answer
    #   - enable_thinking=False: skip the <think> block entirely (Qwen3
    #     chat templates support this via the `enable_thinking` kwarg)
    template_kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
    )
    try:
        text = tok.apply_chat_template(messages, **template_kwargs, enable_thinking=enable_thinking)
    except TypeError:
        # Older / non-Qwen3 templates ignore the kwarg
        text = tok.apply_chat_template(messages, **template_kwargs)
    inputs = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    new = out[0, inputs.input_ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True)


def _wrap_env_action(tool_call) -> dict:
    """Map a parsed tool_call into the dict shape ALFWorldEnv.step expects."""
    return {
        "function": {
            "name": tool_call.name,
            "arguments": tool_call.arguments,
        }
    }


def stage_run_episode(
    model,
    tok,
    env,
    agent,
    max_steps: int,
    max_new_tokens: int,
    enable_thinking: bool,
) -> bool:
    _stage(
        f"Stage 5: run episode (max_steps={max_steps}, "
        f"max_new_tokens={max_new_tokens}, enable_thinking={enable_thinking})"
    )
    try:
        # Reset env, prime agent with first observation
        obs, info = env.reset()
        agent.update_from_env(obs, 0.0, False, info)
        _info(f"initial obs: {str(obs)[:200]!r}...")

        for step in range(max_steps):
            print(f"\n  --- step {step+1} ---")
            t0 = time.time()
            response = _generate(
                model, tok, agent.messages,
                max_new=max_new_tokens,
                enable_thinking=enable_thinking,
            )
            dt = time.time() - t0
            print(f"  model ({dt:.1f}s): {response[:300]!r}{' …' if len(response) > 300 else ''}")

            parse = agent.update_from_model(response)
            if not parse.tool_calls:
                _info("no tool call parsed (parser format errors? agent will be re-prompted next turn)")
                if parse.format_errors:
                    print(f"  errors: {[e.error_type for e in parse.format_errors]}")
                # Still feed something back so agent can recover
                obs = {"observation": "[error: please emit a tool call wrapped in <tool_call>...</tool_call>]"}
                agent.update_from_env(obs, 0.0, False, {})
                continue

            tc = parse.tool_calls[0]
            print(f"  tool: {tc.name}  args: {str(tc.arguments)[:200]}")

            if agent.is_memory_tool(tc.name):
                result = agent.execute_memory_tool(tc.name, tc.arguments)
                _ok(f"memory tool {tc.name} -> success={result.success}")
                if result.message:
                    print(f"  memory result: {result.message[:200]!r}")
                obs = {"observation": result.message or "(memory tool ran)"}
                agent.update_from_env(obs, 0.0, False, {})
            else:
                action = _wrap_env_action(tc)
                obs, reward, done, info = env.step(action)
                print(f"  env reward={reward}  done={done}")
                print(f"  next obs: {str(obs.get('observation','') if isinstance(obs, dict) else obs)[:200]!r}")
                agent.update_from_env(obs, reward, done, info)
                if done:
                    _ok(f"episode finished at step {step+1}, won={env.won}")
                    return True

        _info(f"hit max_steps={max_steps} without env-side done")
        return True
    except Exception as e:  # noqa: BLE001
        _fail(f"episode crashed: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Stage 6: dump memory state
# ---------------------------------------------------------------------------

def stage_dump_state(agent) -> bool:
    _stage("Stage 6: agent memory state")
    try:
        stats = agent.get_memory_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")
        if agent.compression_mode == "graph_db" and agent.context_db is not None:
            if hasattr(agent.context_db, "get_stats"):
                print("  graph stats:", agent.context_db.get_stats())
            if hasattr(agent.context_db, "list_entities"):
                ents = agent.context_db.list_entities()
                print(f"  entities ({len(ents)}): {ents[:20]}")
        return True
    except Exception as e:  # noqa: BLE001
        _fail(f"dump crashed: {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=os.environ.get(
            "SMOKE_MODEL",
            "/work/09281/chc_1996/vista/MemexRL/models/Qwen3-4B-Thinking-2507",
        ),
    )
    p.add_argument(
        "--alfworld-data",
        default=os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.alfworld")),
    )
    p.add_argument(
        "--mode",
        choices=("lossless_db", "lossy", "rag", "graph_db"),
        default="lossless_db",
        help="Memory compression mode for the agent.",
    )
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Per-turn generation budget. Thinking models (Qwen3-*-Thinking-*) "
             "need at least 2048 to finish thinking AND emit a tool call; "
             "smaller values get cut off mid-thinking and the agent sees "
             "format errors every turn.",
    )
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="Pass enable_thinking=False to the chat template, skipping the "
             "<think>...</think> block. Qwen3 supports this. Recommended for "
             "smoke tests where you want to see the tool call directly.",
    )
    args = p.parse_args()

    n_failed = 0
    if not stage_imports():
        sys.exit(1)
    model, tok = stage_load_model(args.model)
    if model is None:
        sys.exit(2)
    game = stage_find_game(args.alfworld_data)
    if game is None:
        sys.exit(3)
    env, agent = stage_make_env_agent(game, args.mode, args.model)
    if env is None:
        sys.exit(4)

    if not stage_run_episode(
        model, tok, env, agent,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=not args.no_thinking,
    ):
        n_failed += 1
    if not stage_dump_state(agent):
        n_failed += 1

    print(f"\n\033[1;36m==== summary ====\033[0m")
    if n_failed == 0:
        print("  \033[1;32mSMOKE PASSED — agent + env + 4B model all wired up\033[0m")
    else:
        print(f"  \033[1;31m{n_failed} stages had failures (see logs above)\033[0m")
    sys.exit(n_failed)


if __name__ == "__main__":
    main()
