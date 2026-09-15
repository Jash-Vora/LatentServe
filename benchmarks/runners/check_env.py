"""
Phase 0 environment check.

Run this before writing any model code:

    python -m benchmarks.runners.check_env

It verifies: PyTorch sees the GPU(s), reports T4-relevant capabilities
(fp16 vs bf16 tensor core support, compute capability), runs a trivial
matmul to confirm the driver/toolkit actually work end-to-end, that the
Qwen2.5-1.5B-Instruct weights/tokenizer can actually be fetched from
Hugging Face and loaded onto the GPU (the fixed model substrate for the
whole project — see docs/methodology.md "Model Strategy"), and prints
library versions for the reproducibility record.

This does NOT require Triton, Nsight, or vLLM — those come later
(Phases 11, 12, 6 respectively).
"""

from __future__ import annotations

import sys
import time


def check_torch_cuda() -> bool:
    try:
        import torch
    except ImportError:
        print("[FAIL] torch is not installed.")
        return False

    print(f"[OK] torch {torch.__version__}")

    if not torch.cuda.is_available():
        print("[FAIL] torch.cuda.is_available() is False. Check driver/CUDA install.")
        return False

    n = torch.cuda.device_count()
    print(f"[OK] {n} CUDA device(s) visible")

    ok = True
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        cc = f"{props.major}.{props.minor}"
        vram_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name} | compute capability {cc} | {vram_gb:.1f} GB")

        # T4 = compute capability 7.5 (Turing): fp16 tensor cores yes, bf16 tensor cores NO.
        if props.major < 7:
            print(
                f"  [WARN] GPU {i} has compute capability < 7.0 — no tensor cores, "
                f"MLA/sparse kernel work will be unrepresentative of T4 behavior."
            )
        if props.major == 7 and props.minor == 5:
            print(
                f"  [INFO] GPU {i} looks like a Turing-class card (e.g. T4). "
                f"Use dtype=fp16, not bf16 — Turing lacks bf16 tensor core support."
            )
    return ok


def check_matmul() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            c = a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
        flops = 2 * 4096**3
        tflops = flops / dt / 1e12
        print(f"[OK] fp16 4096x4096 matmul: {dt * 1000:.2f} ms/iter (~{tflops:.1f} TFLOPS)")
        del a, b, c
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"[FAIL] matmul smoke test raised: {e}")
        return False


def check_qwen_loadable() -> bool:
    """Confirm the fixed model substrate is actually reachable/loadable.

    This is a Phase 0 gate specifically because it's the most likely
    first-run failure on a fresh Kaggle/cloud box (no HF auth cached, no
    disk space, no internet egress, etc.) and we'd rather find that out
    now than mid-way through Phase 1.
    """
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError:
        print("[FAIL] transformers is not installed (needed from Phase 1 onward).")
        return False

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    try:
        cfg = AutoConfig.from_pretrained(model_name)
        tok = AutoTokenizer.from_pretrained(model_name)
        print(f"[OK] {model_name} config + tokenizer reachable")
        print(
            f"  layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
            f"kv_heads={getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)} "
            f"hidden_dim={cfg.hidden_size}"
        )
        _ = tok("hello world")
        return True
    except Exception as e:
        print(f"[FAIL] could not fetch/load {model_name}: {e}")
        print("  Check internet egress, HF auth/cache, and disk space.")
        return False


def check_optional_libs() -> None:
    for lib, needed_by in [
        ("triton", "Phase 11 (custom kernels)"),
        ("vllm", "Phase 6 (vLLM baseline)"),
    ]:
        try:
            mod = __import__(lib)
            print(f"[OK] {lib} {getattr(mod, '__version__', 'unknown')} (needed by {needed_by})")
        except ImportError:
            print(f"[INFO] {lib} not installed yet — fine for now, needed by {needed_by}")


def check_nsight() -> None:
    import shutil

    for tool, phase in [("nsys", "Phase 12"), ("ncu", "Phase 12")]:
        path = shutil.which(tool)
        if path:
            print(f"[OK] {tool} found at {path}")
        else:
            print(f"[INFO] {tool} not on PATH — needed by {phase}, not required yet")


def main() -> int:
    print("=" * 60)
    print("LatentServe — Phase 0 environment check")
    print("=" * 60)

    ok = check_torch_cuda()
    if ok:
        ok = check_matmul() and ok

    print("-" * 60)
    ok = check_qwen_loadable() and ok

    print("-" * 60)
    check_optional_libs()
    print("-" * 60)
    check_nsight()
    print("=" * 60)

    if ok:
        print("Environment looks ready for Phase 1 (correctness baseline).")
        return 0
    else:
        print("Fix the [FAIL] items above before starting Phase 1.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
