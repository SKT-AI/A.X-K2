#!/usr/bin/env python
"""A.X K2 on vLLM: load the FP8 checkpoint with tensor parallelism and generate.

Stages are independent, so a failure still shows what did work.

Everything runs under a __main__ guard: vLLM starts its engine-core processes
with spawn, which re-imports this module in the child.
"""

import argparse
import time
import traceback

PROMPTS = [
    "대한민국의 수도는 어디인가?",
    "3의 제곱은 얼마인가?",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="skt/A.X-K2", help="local path or Hugging Face repo id")
    ap.add_argument("--tp", type=int, default=4, help="tensor-parallel size")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int, default=128)
    return ap.parse_args()


def main():
    args = parse_args()
    results = {}

    def stage(name, fn):
        print(f"\n{'=' * 72}\n### {name}\n{'=' * 72}", flush=True)
        t0 = time.time()
        try:
            out = fn()
            results[name] = ("PASS", f"{time.time() - t0:.1f}s")
            return out
        except Exception as e:
            results[name] = ("FAIL", f"{type(e).__name__}: {e}")
            traceback.print_exc()
            return None

    import torch

    print(f"torch {torch.__version__} | GPUs {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}: {p.name} sm{p.major}{p.minor} {p.total_memory / 2**30:.0f}GiB")

    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"vllm {vllm.__version__}")

    t_load = time.time()
    llm = stage(
        f"1. load, TP={args.tp}",
        lambda: LLM(
            model=args.ckpt,
            tensor_parallel_size=args.tp,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            enforce_eager=True,
            trust_remote_code=False,
        ),
    )
    print(f"\n>>> wall-clock to loaded engine: {time.time() - t_load:.1f}s")

    tok = AutoTokenizer.from_pretrained(args.ckpt)

    def _gen(enable_thinking):
        assert llm is not None, "model failed to load"
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                tokenize=False,
            )
            for p in PROMPTS
        ]
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        dt = time.time() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"{ntok} tokens in {dt:.1f}s ({ntok / dt:.1f} tok/s)")
        for p, o in zip(PROMPTS, outs):
            print(f"\n--- {p}\n{o.outputs[0].text[:600]}")
        assert ntok > 0, "no tokens generated"
        return None

    stage("2. generate - thinking mode", lambda: _gen(True))
    stage("3. generate - non-thinking mode", lambda: _gen(False))

    print(f"\n{'=' * 72}\n### SUMMARY\n{'=' * 72}")
    for k, (v, note) in results.items():
        print(f"  {v}  {k:44s} {note}")
    failed = [k for k, (v, _) in results.items() if v == "FAIL"]
    print(f"\n{len(failed)} stage(s) failed: {failed}" if failed else "\nALL STAGES PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
