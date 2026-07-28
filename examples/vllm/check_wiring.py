#!/usr/bin/env python
"""Check that vLLM resolves the A.X K2 checkpoint correctly. No GPU required.

Exercises the four registration points A.X K2 support touches, without loading
weights, so a misconfigured environment is caught in seconds instead of after a
multi-minute load.

    python check_wiring.py [--ckpt skt/A.X-K2]
"""

import argparse
import sys
import traceback


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="skt/A.X-K2", help="local path or Hugging Face repo id")
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt = args.ckpt
    results = []

    def check(name, fn):
        try:
            detail = fn()
            results.append((True, name, detail))
            print(f"PASS  {name}" + (f"  -> {detail}" if detail else ""))
        except Exception as e:
            results.append((False, name, f"{type(e).__name__}: {e}"))
            print(f"FAIL  {name}  -> {type(e).__name__}: {e}")
            traceback.print_exc()

    # 1. model_type "axk2" must reach vLLM's own AXK2Config. If it falls through
    #    to the Transformers config, keys the model needs get popped.
    def _config():
        from vllm.transformers_utils.config import get_config

        cfg = get_config(ckpt, trust_remote_code=False)
        assert type(cfg).__name__ == "AXK2Config", f"got {type(cfg).__name__}"
        assert cfg.model_type == "axk2", cfg.model_type
        for k in ("first_k_dense_replace", "moe_layer_freq", "index_topk", "kv_lora_rank"):
            assert hasattr(cfg, k), f"missing {k}"
        return f"layers={cfg.num_hidden_layers}, routed_experts={cfg.n_routed_experts}"

    check("config resolves to vLLM AXK2Config", _config)

    # 2. architectures -> module/class, and the model module actually imports.
    def _registry():
        from vllm.config import ModelConfig
        from vllm.model_executor.models.registry import ModelRegistry

        arch = "AXK2ForCausalLM"
        assert arch in ModelRegistry.get_supported_archs(), "not in supported archs"
        mc = ModelConfig(model=ckpt, tokenizer=ckpt, trust_remote_code=False, enforce_eager=True)
        model_cls, resolved = ModelRegistry.resolve_model_cls(arch, mc)
        assert resolved == arch, resolved
        return f"{model_cls.__module__}.{model_cls.__name__}"

    check("registry resolves AXK2ForCausalLM", _registry)

    # 3. MLA must be selected and head_size must be one the FlashMLA kernels take.
    def _mla():
        from vllm.config import ModelConfig

        mc = ModelConfig(model=ckpt, tokenizer=ckpt, trust_remote_code=False, enforce_eager=True)
        hs = mc.get_head_size()
        assert mc.is_deepseek_mla, "MLA not selected"
        assert hs == 576, f"head_size {hs} != 576 (kernels accept 320/576)"
        return f"head_size={hs}, is_deepseek_mla={mc.is_deepseek_mla}"

    check("MLA selected, head_size=576", _mla)

    # 4. the checkpoint is block-scaled FP8; vLLM must pick the fp8 path.
    def _quant():
        from vllm.config import ModelConfig

        mc = ModelConfig(model=ckpt, tokenizer=ckpt, trust_remote_code=False, enforce_eager=True)
        assert mc.quantization == "fp8", f"quantization={mc.quantization!r}"
        blk = mc.hf_config.quantization_config.get("weight_block_size")
        return f"quantization=fp8, weight_block_size={blk}"

    check("fp8 block quantization detected", _quant)

    print("\n" + "=" * 60)
    failed = [n for ok, n, _ in results if not ok]
    for ok, n, d in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {n}")
        if not ok:
            print(f"        {d}")
    print(f"\n{len(failed)} failed" if failed else "\nALL WIRING CHECKS PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
