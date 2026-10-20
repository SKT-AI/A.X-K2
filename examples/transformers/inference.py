#!/usr/bin/env python
"""Single-process, multi-GPU inference for the A.X K2 FP8 checkpoint.

The checkpoint carries its own `quantization_config`, so Transformers keeps the
weights in block-scaled FP8 and shards them across the visible GPUs with
`device_map="auto"`. Do not pass an explicit `dtype` -- that dequantizes the
model to bfloat16 (~1.4 TB).
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", default="skt/A.X-K2", help="local path or Hugging Face repo id")
    p.add_argument("--prompt", default="A.X K2에 대해 간단히 소개해줘.", help="input prompt")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--no_thinking", action="store_true", help="run in non-thinking mode")
    p.add_argument(
        "--check_kernel_only",
        action="store_true",
        help="probe the FP8 matmul kernel and exit, without loading the model",
    )
    return p.parse_args()


def check_fp8_kernel():
    """Probe the FP8 matmul before loading.

    The kernel is fetched from the Hub at the *first forward pass*, i.e. after a
    load that takes tens of minutes. Failing here instead saves that hour.
    """
    from transformers.integrations.finegrained_fp8 import fp8_linear, load_finegrained_fp8_kernel

    kernel = load_finegrained_fp8_kernel()
    w = (torch.randn(256, 512, device="cuda") / 8).to(torch.float8_e4m3fn)
    scale = torch.ones(2, 4, device="cuda", dtype=torch.float32)
    x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    y = fp8_linear(x, w, scale, block_size=[128, 128])
    assert torch.isfinite(y).all(), "FP8 matmul produced non-finite values"
    print(f"FP8 kernel OK: {getattr(kernel, '__name__', type(kernel).__name__)}")


@torch.inference_mode()
def main():
    args = parse_args()

    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}: {prop.name} sm{prop.major}{prop.minor} {prop.total_memory / 2**30:.0f}GiB")

    check_fp8_kernel()
    if args.check_kernel_only:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto").eval()
    print(f"model loaded in {time.time() - t0:.0f}s (attn={model.config._attn_implementation})")

    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=not args.no_thinking,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    # sampling defaults (temperature 0.9, top_p 0.95) come from generation_config.json
    t0 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    dt = time.time() - t0
    print(f"{len(generated)} tokens in {dt:.1f}s ({len(generated) / dt:.1f} tok/s)\n")

    text = tokenizer.decode(generated, skip_special_tokens=True)

    # In thinking mode the template opens `<think>` in the prompt, so the model's
    # output starts inside the reasoning trace and closes it with `</think>`.
    # Both markers are ordinary tokens, so they survive decoding.
    if "</think>" in text:
        reasoning, _, answer = text.partition("</think>")
        print(f"[reasoning]\n{reasoning.strip()}\n\n[answer]\n{answer.strip()}")
    else:
        print(text.strip())


if __name__ == "__main__":
    main()
