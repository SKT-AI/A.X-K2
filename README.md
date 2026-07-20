<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="./assets/A.X_K2_BI_Negative.png">
    <img src="./assets/A.X_K2_BI_Primary.png" alt="A.X Logo" width="300">
  </picture>
</div>
<br/>
<p align="center">
  <a href="https://huggingface.co/collections/skt/ax-k">🤗 Models</a> |
  <a href="https://github.com/SKT-AI/A.X-K2">🖥️ Github</a> |
  <a href="https://github.com/SKT-AI/A.X-K2/blob/main/A_X_K2_Tech_Report.pdf">📄 Technical Report</a>
</p>

## Model Summary

**A.X K2** is a large-scale Mixture-of-Experts (MoE) language model trained from scratch as a high-performance, **agentic** foundation model, and the successor to A.X K1.
The model contains **688 billion total parameters**, with **33 billion active parameters**, delivering strong reasoning and instruction-following performance while maintaining practical inference efficiency.

Through a **Think-Fusion** training recipe, a single unified model supports both a **thinking** mode for complex problem solving and a **non-thinking** mode for concise, low-latency responses, allowing the user to trade quality for cost on a per-request basis.

A.X K2 is developed as part of the Korean government's **Sovereign AI foundation model** project, aiming to build a frontier-scale model with deep understanding of the Korean language and culture.

## Key Features

- **Large-Scale Sparse MoE (688B / 33B Active)**
  Employs a high-capacity Mixture-of-Experts architecture (256 routed experts + 1 shared expert) that activates only 8 experts per token, enabling strong reasoning performance with practical inference efficiency. Relative to A.X K1, the number of routed experts increased from 192 to 256 while keeping the activated parameter count at 33B.

- **Hybrid Reasoning Control (Think / Non-Think)**
  Supports user-controllable reasoning depth via the Think-Fusion recipe, allowing explicit multi-step reasoning or concise low-latency responses within a single unified model.

- **Sparse Gated Attention (SGA) for Efficient Long-Context Inference**
  Combines a lightweight sparse-attention indexer (top-*k* token selection) with head-specific gated attention on top of Multi-head Latent Attention (MLA). This prunes attention computation at long context while suppressing attention sinks, delivering higher throughput and lower latency at long sequence lengths.

- **Gated Norm (GN) for Stable Training and Low-Precision Serving**
  Applies an input-dependent gate immediately after RMSNorm to suppress massive activations and hidden-state outliers. This stabilizes large-scale training and produces an outlier-friendly activation distribution favorable for low-precision (FP8 / NVFP4) deployment.

- **Native FP8 Training and FP8 Checkpoint**
  A.X K2 was trained from scratch **natively in FP8** (MXFP8, E4M3) for both the forward and backward passes, and the released checkpoint is distributed in **block-scaled FP8 (E4M3)**. This halves the memory footprint relative to BF16 and enables efficient FP8 serving out of the box, without a separate post-hoc quantization step.

- **Tokenizer Optimized for Multilingual and Code Data**
  Inherits the large-vocabulary BBPE-based tokenizer from A.X K1, optimized for token efficiency across five languages (English, Korean, Chinese, Japanese, and Spanish), with a strong emphasis on source code, structured text, and programming-related patterns.

## Model Details

- **Architecture:** Decoder-only Transformer with Mixture-of-Experts (MoE)
- **Total parameters:** 688B (256 routed experts + 1 shared expert)
- **Active parameters:** 33B per token (8 experts + 1 shared expert)
- **Number of layers:** 61 (1 dense + 60 MoE)
- **Number of attention heads:** 64
- **Hidden size:** 7168
- **Intermediate size:** 18432 (dense layer)
- **Expert intermediate size:** 2048
- **Attention:** Multi-head Latent Attention (MLA) with head-specific output gate and sparse attention (SGA)
- **Normalization:** Gated Norm (gating applied after RMSNorm) with QK-normalization
- **Vocab size:** 163,840
- **Context length:** 262,144 tokens (256K) — 128K trained natively, extended to 256K via YaRN scaling
- **Training precision:** Native FP8 (MXFP8, E4M3) forward and backward passes; FP32 master weights and gradients
- **Checkpoint precision:** Block-scaled FP8 (E4M3, 128×128 weight blocks with dynamic activation scaling)

## Architecture Highlights

<div align="center">
  <img src="./assets/axk2_sga.png" alt="A.X K2 Transformer Block with the Sparse Gated Attention (SGA) sublayer" width="900">
</div>

<p align="center"><em>The A.X K2 transformer block. A lightweight indexer scores and ranks key–value candidates, and the selector keeps the top-<em>k</em> indices; MLA then performs latent attention over the selected KV cache. A head-specific output gate modulates the attention output before the output projection, while Gated Norm precedes the sublayer to smooth outliers for stable, low-precision (FP8/FP4) computation.</em></p>

### Sparse Gated Attention (SGA)

A.X K2 combines two mechanisms into **SGA** to make long-context inference efficient while keeping training stable:

- **Gated attention:** A head-specific output gate is applied throughout pre-training on top of MLA. It introduces non-linearity into the attention output, mitigates attention sinks, and improves loss convergence.
- **Sparse attention:** A lightweight indexer scores and ranks key–value candidates and selects a small set of top-*k* tokens (*k* = 2048) per query, substantially reducing attention compute at long sequence lengths.

The two mechanisms are mutually reinforcing: by suppressing the attention-sink mass on uninformative tokens, the output gate yields a better-calibrated signal so the indexer spends its limited budget on genuinely relevant positions. In long-context serving (inputs beyond 64K tokens), A.X K2 with SGA sustains substantially higher token throughput, lower time per output token (TPOT), and lower time-to-first-token (TTFT) than both A.X K1 and A.X K2 with standard dense attention.

### Gated Norm (GN)

A.X K2 adopts **Gated Norm**, in which a learned, input-dependent gate modulates the normalized activation before it enters the residual stream. By preventing the normalized output from fitting outlier scalars, GN suppresses massive activations—hidden units orders of magnitude larger than the rest that persist across layers and are a documented source of both training instability and low-precision quantization error.

With this design, training remained sufficiently stable **without** the dual-normalization scheme used in A.X K1: the single post-normalization gating of Gated Norm subsumes the stabilizing effect of the earlier design while reducing architectural complexity. As an additional benefit, the outlier-friendly activation distribution improves block-scale utilization under narrow low-precision formats such as FP8 and NVFP4.

### Think-Fusion (Hybrid Reasoning)

A.X K2 uses a single model to generate responses where reasoning before the answer can be enabled or disabled depending on usage requirements. This is achieved through the **Think-Fusion** recipe, which unifies thinking and non-thinking training into a single track via a paired supervised fine-tuning (SFT) dataset, followed by multi-stage on-policy reinforcement learning.

- **Think mode:**
  Generates reasoning steps (`<think>...</think>`) before producing the answer for complex problem solving and multi-step inference.
- **Non-Think mode:**
  Generates concise, direct responses optimized for low-latency usage.

Rather than a dual-track strategy, A.X K2 unifies training into a single track using a paired SFT dataset in which each prompt is associated with both a thinking and a non-thinking response, so the model learns mode control from explicit control tokens rather than from category-specific artifacts.

## Evaluation Results

### Thinking Mode

All models are evaluated in **thinking mode**. Benchmarks marked with an asterisk (*) are evaluated by [Artificial Analysis](https://artificialanalysis.ai). GDPval is reported as an Elo rating; all other benchmarks are reported as percentages. The best score in each row is shown in **bold**. See the [technical report](./A_X_K2_Tech_Report.pdf) for evaluation details.

| Domain | Benchmark | A.X K2 (688B-A33B) | A.X K1 (519B-A33B) | Qwen3.5 | Nemotron 3 Ultra | DeepSeek-V4 Flash | GLM-5.1 | Kimi-K2.6 | MiniMax M2.7 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Math | AIME26 | **97.1** | 91.3 | 92.5 | 85.4 | 96.7 | 95.4 | 95.4 | 88.8 |
| | Apex | **45.8** | 1.0 | 13.5 | 3.1 | 28.1 | 10.4 | 20.8 | 1.0 |
| | Apex-shortlist | **88.6** | 50.8 | 61.2 | 45.5 | 88.0 | 72.1 | 72.6 | 30.3 |
| | KMO26 (1st round) | 92.5 | 85.0 | 78.8 | 85.0 | **94.4** | 78.1 | 88.1 | 80.0 |
| Korean | KMMLU-Pro | **80.5** | 68.9 | 78.4 | 69.2 | 78.8 | 76.5 | 73.4 | 66.6 |
| | KoBALT | 73.0 | 51.0 | 69.9 | 59.1 | **75.3** | 72.0 | 66.0 | 51.4 |
| | CLIcK | **91.6** | 85.3 | 88.4 | 84.1 | 89.6 | 88.8 | 80.9 | 76.1 |
| Code | LiveCodeBench v6 (Feb–May) | 84.0 | 74.9 | 82.6 | 76.4 | **89.4** | 86.4 | 86.5 | 69.8 |
| | SciCode | 41.0 | 24.1 | 42.0 | 40.0 | 45.0 | 44.0 | **53.0** | 47.0 |
| | Terminal Bench v2.1* | 36.0 | – | 51.3 | 53.9 | 61.8 | 61.8 | **65.9** | 55.4 |
| Science & Knowledge | Humanity’s Last Exam | 27.8 | 8.3 | 27.0 | 27.0 | 32.0 | 28.0 | **36.0** | 28.0 |
| | GPQA Diamond | 85.6 | 76.4 | 89.0 | 87.0 | 89.0 | 87.0 | **91.0** | 87.0 |
| | AA-Omniscience* | 39.6 | 15.5 | 24.3 | 38.3 | 26.0 | 39.7 | **42.3** | 39.3 |
| General | IFBench | 75.9 | 63.2 | 79.0 | 78.9 | **81.2** | 78.6 | 74.0 | 73.2 |
| | AA-LCR | 66.0 | 26.0 | 66.0 | 67.0 | 63.0 | 62.0 | **70.0** | 69.0 |
| Agentic | GDPval* (Elo) | 1031 | 500 | 962 | 1164 | 1189 | **1257** | 1190 | 1158 |
| | τ²-Bench* (Telecom) | **98.0** | 86.0 | 95.6 | 83.3 | 95.0 | 97.7 | 95.9 | 84.8 |
| | τ³-Bench* (Banking) | 13.0 | – | 13.4 | 13.8 | **22.9** | 11.5 | 20.6 | 8.9 |
| | BrowseComp (≤10 searches) | 9.3 | – | 26.9 | 13.4 | 16.8 | **29.1** | 21.5 | 14.2 |

A.X K2 is strongest in the **Math** and **Korean** domains, achieving the best scores among all compared models on AIME26, Apex, Apex-shortlist, KMMLU-Pro, and CLIcK. Beyond final-answer benchmarks, A.X K2 scored **35/42 on IMO 2025**—reaching the gold-medal threshold with a perfect 7/7 on each of the first five problems—and produced **correct proofs for all 8 problems of the KMO26 2nd round**, using an iterative proof refinement method. It also achieves the best score on τ²-Bench Telecom among all compared models.

### Long Context

A.X K2 is trained natively to a 128K context via Adjusted Base Frequency (ABF) and extends to longer sequences through zero-shot YaRN scaling at inference. We assess long-context capability along two complementary axes—comprehensive understanding on **RULER** and long-range fact retrieval via **needle-in-a-haystack (NIAH)**. See the [technical report](./A_X_K2_Tech_Report.pdf) for details.

#### RULER

RULER measures long-context understanding across retrieval, multi-hop tracing, aggregation, and question-answering tasks at increasing sequence lengths. A.X K2 sustains strong performance out to 256K, achieving an overall average of 94.6.

| Model | 4K | 8K | 16K | 32K | 64K | 128K | 256K | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A.X K2 | 97.5 | 97.2 | 97.5 | 96.5 | 94.3 | 92.7 | 86.6 | 94.6 |

#### Needle-in-a-Haystack (NIAH)

NIAH probes exact fact retrieval by inserting a target fact (the "needle") at varying depths within a long context and asking the model to recover it. Under zero-shot YaRN scaling (scaling factors of 1, 2, and 4 for 128K, 256K, and 512K, respectively), A.X K2 attains a perfect retrieval score at every context length and needle depth—and does so even after NVFP4 (experts-only W4A4) quantization. The heatmaps below show the 256K and 512K results.

<div align="center">
  <img src="./assets/axk2_niah_256k_nvfp4.png" alt="A.X K2 NIAH fact retrieval at 256K context under NVFP4 (experts-only W4A4)" width="900">
  <img src="./assets/axk2_niah_512k_nvfp4.png" alt="A.X K2 NIAH fact retrieval at 512K context under NVFP4 (experts-only W4A4)" width="900">
</div>

<p align="center"><em>NIAH fact retrieval across context length (x-axis) and needle depth (y-axis): 256K (YaRN factor 2, top) and 512K (YaRN factor 4, bottom), under NVFP4 (experts-only W4A4) quantization. A.X K2 scores a perfect 100 at every position.</em></p>

## Usage

### Transformers

For users who wish to run direct inference with Hugging Face Transformers, please refer to the example scripts and instructions in [this repository](./examples/transformers/README.md).

### vLLM

An initial **vLLM** integration for A.X K2 will be made publicly available soon—we will update this section with a link to the repository once the PR is submitted.

## Lightweight Models

If you are interested in using lightweight versions of A.X K2 for business applications or scalable deployment, please contact us at **a.x@sk.com**.

## Limitations

- A.X K2 may generate incorrect or misleading information due to its stochastic nature.
- Performance may vary across domains and languages depending on data coverage.
- A.X K2 is currently text-only and does not support native multimodal understanding.

## Citation

If you use A.X K2 in your research, please cite the technical report:

```bibtex
@misc{TBD,
      title={A.X K2 Technical Report},
      author={},
      year={2026},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.XXXXX},
}
```
