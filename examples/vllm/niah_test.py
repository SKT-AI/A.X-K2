#!/usr/bin/env python
"""A.X K2 on vLLM: long-context retrieval (needle-in-a-haystack) up to 256K.

Places a needle at several depths for each target length and checks it comes
back. The published config.json already carries the YaRN configuration, so 256K
needs no extra flags.

Each (length, depth) uses a distinct code so a prefix-cache hit cannot pass the
check by accident. Prompt assembly converges on the target from below, and the
actual token count is reported rather than assumed.
"""

import argparse
import time
import traceback

NEEDLE_TMPL = "A.X K2 프로젝트의 내부 인증 코드는 {code}이다. 이 코드는 반드시 기억해야 한다."
QUESTION = (
    "\n\n위 문서에서 'A.X K2 프로젝트의 내부 인증 코드'를 찾아라. "
    "코드 값만 정확히 답하고 다른 말은 하지 마라."
)
FILLER_UNIT = (
    "그는 조용히 창밖을 바라보았다. 거리에는 사람들이 오가고 있었고, 낮은 구름이 "
    "천천히 흘러갔다. 오래된 건물의 벽에는 담쟁이가 자라 있었다. "
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="skt/A.X-K2", help="local path or Hugging Face repo id")
    ap.add_argument("--tp", type=int, default=4, help="tensor-parallel size")
    ap.add_argument("--max-model-len", type=int, default=262144)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--lengths", type=int, nargs="+", default=[32768, 131072, 262144])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.15, 0.5, 0.85])
    return ap.parse_args()


def build_prompt(tok, target_len, depth, code):
    """Assemble a chat prompt of at most target_len tokens, needle at `depth`."""
    needle = NEEDLE_TMPL.format(code=code)
    # measure the template + needle + question overhead instead of guessing
    overhead = len(
        tok(
            tok.apply_chat_template(
                [{"role": "user", "content": needle + QUESTION}],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=False,
            )
        )["input_ids"]
    )
    unit_len = len(tok(FILLER_UNIT, add_special_tokens=False)["input_ids"])
    n_units = max(1, (target_len - overhead) // unit_len)

    prompt, n = None, 0
    for _ in range(12):
        head_units = int(n_units * depth)
        body = FILLER_UNIT * head_units + needle + FILLER_UNIT * (n_units - head_units)
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": body + QUESTION}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
        n = len(tok(prompt)["input_ids"])
        if n <= target_len:
            break
        n_units -= max(1, int((n - target_len) / unit_len) + 1)
    return prompt, n


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
        f"1. load, TP={args.tp}, max_model_len={args.max_model_len}",
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
    sp = SamplingParams(temperature=0.0, max_tokens=32)

    def _niah(target_len):
        assert llm is not None, "model failed to load"
        prompts, meta = [], []
        for j, d in enumerate(args.depths):
            code = f"KX-{target_len // 1024}K-{j}{j}{j}7"
            p, n = build_prompt(tok, target_len, d, code)
            prompts.append(p)
            meta.append((d, code, n))

        t0 = time.time()
        outs = llm.generate(prompts, sp)
        dt = time.time() - t0
        total_in = sum(m[2] for m in meta)
        print(
            f"{len(prompts)} prompts, {total_in} input tokens in {dt:.1f}s "
            f"({total_in / dt:.0f} prefill tok/s aggregate)"
        )

        hits = 0
        for (d, code, n), o in zip(meta, outs):
            text = o.outputs[0].text.strip()
            ok = code in text
            hits += ok
            print(
                f"  depth={d:<5} n={n:<7} {'HIT ' if ok else 'MISS'} "
                f"want={code!r} got={text[:80]!r}"
            )
        assert hits == len(prompts), f"only {hits}/{len(prompts)} retrieved"
        return f"{hits}/{len(prompts)} at {total_in} total input tokens"

    for L in args.lengths:
        stage(f"2. NIAH @ {L} tokens ({L // 1024}K)", lambda L=L: _niah(L))

    print(f"\n{'=' * 72}\n### SUMMARY\n{'=' * 72}")
    for k, (v, note) in results.items():
        print(f"  {v}  {k:56s} {note}")
    failed = [k for k, (v, _) in results.items() if v == "FAIL"]
    print(f"\n{len(failed)} stage(s) failed: {failed}" if failed else "\nALL STAGES PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
