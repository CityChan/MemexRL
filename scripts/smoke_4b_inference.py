"""
End-to-end inference smoke test on a single GPU.

Designed to run on a TACC Vista GH200 idev node right after the bare-metal
Slime install (scripts/install_slime_vista.sh). Verifies that the full
software stack we just compiled actually drives the GPU and produces
sensible model output. This is the cheapest "does the install really
work?" check before committing to multi-node training.

Stages, each pass/fail independently:
    1. CUDA + GPU sanity (torch.cuda info)
    2. Import sanity (flash_attn, transformer_engine, sglang, megatron, ...)
    3. HF model load (Qwen3-4B-Thinking-2507 by default, ~8 GB)
    4. Tokenize + forward pass
    5. Generate 50 tokens from a fixed prompt
    6. (optional) flash-attn 2 / 3 import only — already covered in stage 2

Usage:
    # Default: assumes ~/.cache or local-dir at $PROJ/models/Qwen3-4B-Thinking-2507
    python scripts/smoke_4b_inference.py

    # Custom model path
    python scripts/smoke_4b_inference.py --model /path/to/Qwen3-4B-Thinking-2507

    # Quick exit after stage 2 (skip the heavy model load)
    python scripts/smoke_4b_inference.py --imports-only

The script never raises on stage failure; it logs FAIL and continues so
you see the full picture. Final exit code = number of failed stages.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback


def _stage(name: str):
    print(f"\n\033[1;36m==== {name} ====\033[0m")


def _ok(msg: str):
    print(f"  \033[1;32m[OK]\033[0m {msg}")


def _fail(msg: str):
    print(f"  \033[1;31m[FAIL]\033[0m {msg}")


def stage_cuda() -> bool:
    _stage("Stage 1: CUDA + GPU sanity")
    try:
        import torch
        _ok(f"torch: {torch.__version__}  cuda: {torch.version.cuda}  cudnn: {torch.backends.cudnn.version()}")
        if not torch.cuda.is_available():
            _fail("torch.cuda.is_available() is False — no GPU visible to PyTorch")
            return False
        _ok(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            _ok(f"  GPU{i}: {props.name}  sm{props.major}{props.minor}  "
                f"{props.total_memory / 1024**3:.1f} GB")
        # Tiny on-GPU compute to prove drivers actually work, not just that
        # the C bindings imported.
        x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16)
        y = x @ x
        torch.cuda.synchronize()
        _ok(f"matmul 4096x4096 bf16 OK, result mean={y.mean().item():.4f}")
        return True
    except Exception as e:  # noqa: BLE001
        _fail(f"CUDA stage crashed: {e}")
        traceback.print_exc()
        return False


def stage_imports() -> bool:
    _stage("Stage 2: Slime stack import sanity")
    mods = [
        ("torch", None),
        ("flash_attn", "2.7.4"),
        ("flash_attn_3.flash_attn_interface", None),
        ("transformer_engine", "2.10.0"),
        ("apex", None),
        ("megatron", None),
        ("sglang", None),
        ("ring_flash_attn", None),
        ("transformers", None),
    ]
    all_ok = True
    for name, expect_prefix in mods:
        try:
            m = __import__(name, fromlist=["__version__"])
            v = getattr(m, "__version__", "?")
            if expect_prefix is not None and not v.startswith(expect_prefix):
                _ok(f"{name:38s} {v} (expected prefix {expect_prefix}; treating as warn)")
            else:
                _ok(f"{name:38s} {v}")
        except Exception as e:  # noqa: BLE001
            _fail(f"{name:38s} {type(e).__name__}: {e}")
            all_ok = False
    return all_ok


def stage_load(model_path: str) -> tuple[bool, object, object]:
    _stage(f"Stage 3: Load model from {model_path}")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not os.path.exists(model_path):
            _fail(f"model dir not found: {model_path}")
            _fail("  download with:")
            _fail(f"    huggingface-cli download Qwen/Qwen3-4B-Thinking-2507 --local-dir {model_path}")
            return False, None, None
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _ok(f"tokenizer loaded ({time.time() - t0:.1f}s)")
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        _ok(f"model loaded ({time.time() - t0:.1f}s)")
        # Brief memory report
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            _ok(f"GPU mem: {alloc:.1f} / {total:.1f} GB allocated")
        return True, model, tokenizer
    except Exception as e:  # noqa: BLE001
        _fail(f"model load crashed: {e}")
        traceback.print_exc()
        return False, None, None


def stage_forward(model, tokenizer) -> bool:
    _stage("Stage 4: Forward pass")
    try:
        import torch
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            t0 = time.time()
            out = model(**inputs)
            torch.cuda.synchronize()
            dt = time.time() - t0
        _ok(f"forward pass {dt*1000:.1f}ms  logits shape {tuple(out.logits.shape)}")
        return True
    except Exception as e:  # noqa: BLE001
        _fail(f"forward crashed: {e}")
        traceback.print_exc()
        return False


def stage_generate(model, tokenizer) -> bool:
    _stage("Stage 5: Generate 50 tokens")
    try:
        import torch
        # A simple test prompt that exercises the chat template if present.
        prompt = (
            "You are a helpful assistant. Answer in one short sentence.\n"
            "Q: What is 2 + 2?\nA:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        dt = time.time() - t0
        new_tokens = out[0, inputs.input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        toks = new_tokens.shape[0]
        _ok(f"generated {toks} tokens in {dt:.2f}s ({toks/dt:.1f} tok/s)")
        print(f"  prompt:     {prompt!r}")
        print(f"  completion: {text!r}")
        if not text.strip():
            _fail("model produced empty completion")
            return False
        return True
    except Exception as e:  # noqa: BLE001
        _fail(f"generate crashed: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "SMOKE_MODEL",
            os.path.expanduser(
                "/work/09281/chc_1996/vista/MemexRL/models/Qwen3-4B-Thinking-2507"
            ),
        ),
        help="Path to the local HF model directory (default points at the "
             "convention used in the install_slime_vista.sh layout).",
    )
    parser.add_argument(
        "--imports-only",
        action="store_true",
        help="Stop after stage 2 (no model load, no inference).",
    )
    args = parser.parse_args()

    n_failed = 0

    if not stage_cuda():
        n_failed += 1
    if not stage_imports():
        n_failed += 1

    if args.imports_only:
        print(f"\n\033[1;36mimports-only mode: stopping early\033[0m")
        sys.exit(n_failed)

    ok, model, tokenizer = stage_load(args.model)
    if not ok:
        n_failed += 1
        sys.exit(n_failed)

    if not stage_forward(model, tokenizer):
        n_failed += 1
    if not stage_generate(model, tokenizer):
        n_failed += 1

    print(f"\n\033[1;36m==== summary ====\033[0m")
    if n_failed == 0:
        print("  \033[1;32mALL SMOKE STAGES PASSED — install is functional\033[0m")
    else:
        print(f"  \033[1;31m{n_failed} STAGES FAILED\033[0m")
    sys.exit(n_failed)


if __name__ == "__main__":
    main()
