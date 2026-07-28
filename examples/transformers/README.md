# A.X K2 Transformers 추론 예제

[A.X K2 모델](https://huggingface.co/skt/A.X-K2)을 Hugging Face Transformers로 추론하는 예제입니다.

`axk2` 아키텍처는 Transformers 라이브러리에 기본 포함되어 있으므로 `trust_remote_code=True`나 별도의 모델링 파일이 필요하지 않습니다. 공개된 체크포인트는 **block-scaled FP8 (E4M3)** 이며, `config.json`의 `quantization_config`가 FP8 실행을 그대로 설정해 주기 때문에 별도의 양자화 단계도 필요하지 않습니다.

## 요구사항

- **GPU**: compute capability 8.9 이상 (예: RTX 4090, H100, B200/B300)
- **GPU 메모리**: FP8 체크포인트는 약 **647 GiB**입니다. KV 캐시와 활성값을 포함해 **단일 노드에서 총 800 GiB 이상**을 확보하세요.
  - 검증 구성: **B300 4장** (장당 275 GB) — 장당 약 170–180 GiB, 합계 약 675 GiB 사용
  - H200(141 GB) 8장, B200/B300 4장 이상이면 한 노드에 들어갑니다
  - H100(80 GB) 8장은 합계 640 GiB로 체크포인트보다 작아 한 노드에 들어가지 않습니다. 이 예제는 단일 프로세스 · 단일 노드 구성이므로, 80 GB 카드로 서빙하려면 다중 노드를 지원하는 [vLLM 경로](../vllm/README.md)를 사용하세요
  - A100은 compute capability 8.0으로 FP8 실행 대상이 아닙니다 (아래 참고 항목)
- **Python**: 3.10+

> **참고**
> - compute capability 8.9 미만이거나 GPU가 보이지 않으면 Transformers가 가중치를 bfloat16으로 자동 역양자화하며, 이 경우 약 **1.4 TB**의 메모리가 필요합니다.
> - SM90 이상에서는 DeepGEMM 커널을 우선 사용하고, 그 외에는 `kernels` Triton 경로가 폴백으로 동작합니다. 모델이 2장 이상의 GPU에 분산되면 DeepGEMM은 설계상 비활성화되고 Triton 경로가 사용됩니다.

## 설치

`axk2` 통합은 Transformers **v5.14.1** 이후에 머지되었습니다. 다음 릴리스가 나오기 전까지는 소스에서 설치해야 합니다. FP8 실행에는 block-scaled FP8 matmul을 제공하는 `kernels` 패키지가 추가로 필요합니다.

```bash
# CUDA 버전에 맞는 torch를 먼저 설치하세요
pip install git+https://github.com/huggingface/transformers.git accelerate "kernels>=0.15.2,<0.16.0"
```

`main` 브랜치는 자주 변경되므로, 재현 가능한 환경이 필요하면 커밋을 고정하세요 (`...transformers.git@<commit>`). `axk2`가 포함된 릴리스가 공개된 뒤에는 `pip install -U transformers accelerate "kernels>=0.15.2,<0.16.0"`만으로 충분합니다.

검증에 사용한 조합은 다음과 같습니다.

| 패키지 | 버전 |
|------|------|
| `torch` | 2.13.0+cu130 |
| `transformers` | 5.15.0.dev0 (source) |
| `accelerate` | 1.14.0 |
| `kernels` | 0.15.2 |

## 파일 구성

| 파일 | 설명 |
|------|------|
| `inference.py` | 추론 스크립트 (`device_map="auto"`로 다중 GPU 분산) |
| `inference.batch` | SLURM 배치 스크립트 |

## 실행 방법

### 직접 실행

`device_map="auto"`가 모델을 보이는 GPU들에 나누어 배치하므로 `torchrun` 없이 **단일 프로세스**로 실행합니다.

```bash
python inference.py \
    --model_dir skt/A.X-K2 \
    --prompt "MoE에 대해서 설명해 주세요" \
    --max_new_tokens 1024
```

### SLURM 환경

1. `inference.batch`에서 가상환경 경로를 설정합니다:
   ```bash
   export VENV=<YOUR_VENV_PATH>
   ```
2. 배치 작업을 제출합니다:
   ```bash
   sbatch inference.batch
   ```

## 인자

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--model_dir` | 모델 경로 또는 Hugging Face repo id | `skt/A.X-K2` |
| `--prompt` | 입력 프롬프트 | `A.X K2에 대해 간단히 소개해줘.` |
| `--max_new_tokens` | 최대 생성 토큰 수 | 1024 |
| `--no_thinking` | non-thinking 모드로 실행 | False (기본은 thinking) |
| `--check_kernel_only` | FP8 커널만 점검하고 종료 | False |

## 참고사항

### FP8 커널을 먼저 점검하세요

FP8 matmul 커널은 **첫 forward pass 시점**에 Hub에서 내려받습니다. 즉 수십 분이 걸리는 모델 로딩이 모두 끝난 뒤에야 커널 문제가 드러납니다. `inference.py`는 로딩 전에 커널을 미리 점검하며, 다음과 같이 커널만 따로 확인할 수도 있습니다.

```bash
python inference.py --check_kernel_only
```

### 로딩 시간과 처리량

검증 환경(B300 4장, 공유 Lustre)에서 로딩에 약 **55분**이 소요되었으며, **매 로드마다 반복됩니다** — 두 번 측정해 각각 54.9분, 56.2분이었고, 디스크에 남아 재사용되는 산출물이 없어 두 번째 로드도 빨라지지 않습니다.

원인은 스토리지도 체크포인트 크기도 아닙니다. 346개 샤드 647 GiB를 실효 약 200 MB/s로 읽은 셈인데, 로딩이 진행 중인 같은 노드에서 4개 스트림 기준 약 4 GB/s를 읽을 수 있었습니다(20배 여유). 같은 체크포인트를 같은 스토리지·같은 GPU 4장에서 vLLM은 3분에 적재합니다(아래 표). 시간은 Transformers 로더 안에서 소비됩니다(원인은 프로파일링하지 않았습니다). 이 경로를 쓴다면 로딩을 한 번만 하고 프로세스를 유지하는 방식으로 사용하세요.

**로딩 시간이 중요하면 [vLLM 경로](../vllm/README.md)를 쓰세요.** 같은 하드웨어(B300 4장, vLLM은 TP=4)에서 측정한 값입니다.

| 단계 | Transformers | vLLM 첫 기동 | vLLM 재기동 (커널 캐시 있음) |
|------|---:|---:|---:|
| 가중치 적재 | 약 55분 | 약 3분 | 약 3분 |
| 커널 warmup · JIT 컴파일 | 없음 | 약 21분 | 약 1분 |
| **추론 가능해지기까지** | **약 55분** | **약 25분** | **약 6분** |

마지막 행은 프로세스·엔진 초기화까지 포함한 실측 wall-clock이라 위 두 행의 단순 합과 정확히 일치하지 않습니다.

- vLLM의 21분은 DeepGEMM warmup과 flashinfer JIT 컴파일이며, 캐시되므로 **한 번만** 지불합니다.
- Transformers의 55분은 캐시되지 않아 **기동할 때마다** 지불합니다.
- Transformers 경로에는 이에 상응하는 warmup 단계가 없습니다 — FP8 커널을 첫 forward에 내려받지만 55분에 비하면 무시할 수준입니다.

생성 속도는 같은 환경(B300 4장)에서 배치 1 기준 약 **1.5 tok/s**였습니다. Transformers 경로는 정확성 검증과 소규모 실험을 위한 것이며, 서빙에는 [vLLM 경로](../vllm/README.md)를 사용하세요.

### 컨텍스트 길이

Transformers 경로에서 컨텍스트 길이는 GPU 메모리에 제약됩니다. attention이 `[batch, heads, S, S]` 텐서를 실체화하므로 필요 메모리가 길이의 제곱으로 늘어나며, 이 예제의 `device_map="auto"` 구성에서는 그 텐서가 한 장에 통째로 잡힙니다. B300 4장 기준으로는 수만 토큰 규모가 한계이니, 더 긴 컨텍스트가 필요하면 GPU를 늘리거나(장당 여유 증가) 텐서 병렬로 실행하세요 — 텐서 병렬에서는 헤드가 랭크별로 나뉘면서 이 텐서도 함께 쪼개집니다. 장문이 목적이라면 [vLLM 경로](../vllm/README.md)가 맞습니다. 그쪽은 indexer의 top-k 인덱스를 sparse MLA 커널에 직접 넘겨 이 텐서를 만들지 않으며, 256K 서빙까지 검증되어 있습니다.

### 로더 경고

로딩 시 `weight_scale_inv` 관련 **242개의 missing key 경고**가 나타나는 것은 정상입니다. Gated Norm의 게이트 MLP(`fc1`/`fc2`)는 FP8로 양자화되지 않고 bfloat16으로 유지되므로 해당 스케일 텐서를 사용하지 않습니다. unexpected key는 발생하지 않아야 합니다.

### 에어갭 환경

`HF_HUB_OFFLINE=1`만으로는 부족합니다. `kernels`가 커널 *버전*을 Hub를 통해 커밋으로 해석하기 때문에, 스냅샷이 이미 캐시에 있어도 이 조회가 실패합니다. 네트워크가 되는 장비에서 먼저 내려받아 고정하세요.

```bash
# 1. 네트워크가 되는 장비에서 공유 스토리지로 내려받기
export HF_HOME=/shared/hf
python -c "from transformers.integrations.hub_kernels import lazy_load_kernel; lazy_load_kernel('finegrained-fp8')"
ls $HF_HOME/hub/kernels--kernels-community--finegrained-fp8/snapshots   # -> <commit>

# 2. 에어갭 노드에서 해당 스냅샷을 LOCAL_KERNELS로 지정
export HF_HOME=/shared/hf HF_HUB_OFFLINE=1
export LOCAL_KERNELS="kernels-community/finegrained-fp8=$HF_HOME/hub/kernels--kernels-community--finegrained-fp8/snapshots/<commit>"
```

### Attention 구현

A.X K2는 sparse-attention 마스크를 명시적인 additive mask로 적용하므로 `sdpa`(기본값, 권장)와 `eager`에서 동작합니다. `flash_attention_2`와 `flex_attention`은 지원되지 않습니다.

### Thinking / Non-Thinking 모드

`apply_chat_template`의 `enable_thinking` 인자로 모드를 제어합니다.

- `enable_thinking=True` (기본): 프롬프트가 `<think>`로 끝나므로 모델 출력은 추론 과정에서 시작하여 `</think>`로 닫힙니다.
- `enable_thinking=False`: 프롬프트가 `</think>`로 끝나며 모델은 곧바로 답변을 생성합니다.

`<think>`와 `</think>`는 special token이 아닌 일반 토큰이므로 `skip_special_tokens=True`로 디코딩해도 남아 있습니다. `inference.py`는 이를 이용해 추론 과정과 최종 답변을 분리해 출력합니다.

### 샘플링 파라미터

`generation_config.json`에 권장값(`temperature=0.6`, `top_p=0.95`, `do_sample=True`)이 포함되어 있으므로 `generate()`에서 따로 지정하지 않아도 적용됩니다.

### `dtype`을 지정하지 마세요

`from_pretrained`에 `dtype=torch.bfloat16`을 넘기면 FP8 체크포인트가 bfloat16으로 역양자화되어 메모리 사용량이 두 배가 됩니다. 체크포인트의 `quantization_config`가 실행 정밀도를 결정하도록 두세요.
