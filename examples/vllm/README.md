# A.X K2 vLLM 검증 예제

[A.X K2 모델](https://huggingface.co/skt/A.X-K2)을 vLLM으로 서빙할 때 환경이 올바르게 구성됐는지 확인하는 스크립트입니다.

A.X K2 지원은 업스트림 vLLM PR로 준비 중이며, 머지 전까지는 [SKT-AI/vllm](https://github.com/SKT-AI/vllm)의 `axk2-v0.23.0` 브랜치를 사용합니다. 이 브랜치는 업스트림 vLLM **v0.23.0**에 A.X K2 지원만 더한 것입니다.

```bash
git clone -b axk2-v0.23.0 https://github.com/SKT-AI/vllm.git
cd vllm
pip install -e .   # 빌드 사전 요구사항은 vLLM 문서를 참고하세요
```

`axk2` config와 모델이 in-tree로 등록되어 있으므로 `--trust-remote-code`는 필요하지 않습니다.

## 요구사항

- **GPU 메모리**: 적재 시 FP8 가중치가 **합계 약 656 GiB**를 차지하며 TP rank에 균등 분산됩니다. KV cache 여유를 포함해 TP 크기를 정하세요.
  - 검증 구성: **NVIDIA B300 (장당 275 GB) 4장, TP=4** — GPU당 가중치 164 GiB, KV cache 약 968K 토큰
  - 아래의 모든 메모리·시간 수치는 이 구성(B300 4장, TP=4)에서 측정한 값입니다
- **컴파일러 툴체인**: vLLM이 DeepGEMM·flashinfer 커널을 최초 사용 시 JIT 컴파일하므로 `ninja`와 `nvcc`가 PATH에 있어야 합니다. 없으면 **가중치를 모두 적재한 뒤에** 실패합니다.

## 파일 구성

| 파일 | 설명 | GPU |
|------|------|-----|
| `check_wiring.py` | config·registry·MLA·FP8 등록 지점 확인 | 불필요 |
| `smoke_test.py` | 적재 후 thinking / non-thinking 생성 | 필요 |
| `niah_test.py` | 32K / 128K / 256K 장문 검색 (NIAH) | 필요 |
| `verify.batch` | 위 3개를 순서대로 실행하는 SLURM 스크립트 | 필요 |

## 실행 방법

### 1. 배선 확인 (GPU 불필요, 수 초)

가중치를 적재하지 않고 등록 지점만 확인하므로, 잘못된 환경을 수 분짜리 적재 전에 걸러냅니다.

```bash
python check_wiring.py --ckpt skt/A.X-K2
```

확인 항목은 네 가지입니다.

- `model_type: "axk2"`가 vLLM의 `AXK2Config`로 해석되는지 (Transformers config로 흘러가면 모델이 필요한 키가 유실됩니다)
- `AXK2ForCausalLM`이 registry에서 resolve되고 모델 모듈이 실제로 import되는지
- MLA가 선택되고 `head_size == 576`인지 (FlashMLA 커널은 320/576만 받습니다)
- block-scaled FP8 양자화가 감지되는지

### 2. 생성 확인

```bash
python smoke_test.py --ckpt skt/A.X-K2 --tp 4
```

### 3. 장문 검색 확인

```bash
python niah_test.py --ckpt skt/A.X-K2 --tp 4 \
    --max-model-len 262144 \
    --lengths 32768 131072 262144
```

길이별로 depth 0.15 / 0.5 / 0.85에 needle을 심고 회수 여부를 확인합니다. 길이·depth마다 코드값이 달라서 prefix 캐시 히트로 우연히 통과하는 경우가 배제됩니다. 프롬프트는 목표 길이에서 아래로 수렴시키며 **실제 토큰 수를 함께 출력**합니다.

### SLURM 환경

`verify.batch`에서 가상환경 경로와 TP를 설정한 뒤 제출합니다.

```bash
sbatch verify.batch
```

## 참고사항

### 기동 시간

첫 기동을 지배하는 것은 가중치 읽기가 아니라 **DeepGEMM warmup과 flashinfer JIT 컴파일**입니다. B300 4장(TP=4), 공유 Lustre 스토리지 기준으로 가중치 적재는 약 3.5분이었지만 엔진 전체가 올라오기까지 약 25분이 걸렸습니다.

이 커널들은 `VLLM_CACHE_ROOT`(기본 `~/.cache/vllm`)와 flashinfer 캐시에 저장되며, 캐시가 채워진 뒤 같은 구성에서 재기동은 **6분 미만**이었습니다. 캐시를 영구 스토리지에 두세요. 다중 노드에서는 공유 스토리지에 두면 노드 간 재사용됩니다.

### 첫 로딩이 더 느리지는 않습니다

가중치 적재는 측정한 모든 조합에서 **196–221초** 사이였고, 조건에 따른 방향성이 없었습니다.

| 파일시스템 캐시 | `--load-format` | 적재 |
|---|---|---:|
| cold (체크포인트 잔존율 0% 확인) | `auto` | 206초 |
| cold (동일) | `fastsafetensors` | 208초 |
| warm | `auto` | 196초 |
| warm | `fastsafetensors` | 221초 |

즉 **`--load-format fastsafetensors`를 쓸 이유가 없습니다.** 기본 로더가 cold에서도 약 3.1 GiB/s를 내는데 이 스토리지의 4스트림 가용 대역폭이 약 4 GB/s이므로 짜낼 여지가 거의 없고, 게다가 vLLM은 TP가 1보다 크면 GPU Direct Storage를 스스로 비활성화합니다(`nogds = pg.size() > 1`) — fastsafetensors의 이점이 나오는 지점이 바로 거기입니다.

로컬 NVMe에 두고 TP=1로 돌리는 등 GDS가 실제로 동작하는 구성에서는 결과가 다를 수 있습니다. 캐시 상태는 `mincore(2)`로 확인했습니다(추측이 아니라 측정값입니다).

### 장문 컨텍스트

공개된 `config.json`이 YaRN 설정을 이미 담고 있어 256K는 추가 플래그 없이 동작합니다.

256K 요청 3개를 동시에 처리하려면 KV cache가 약 786K 토큰 필요합니다. B300 4장(TP=4)에서는 약 968K 토큰이 확보되어 들어갔습니다. 부족하면 vLLM이 순차적으로 스케줄링하므로 실패하지는 않고 시간만 늘어납니다.

### 툴 호출과 thinking 모드

OpenAI 호환 엔드포인트로 서빙할 때는 파서 두 개를 지정하세요.

```bash
vllm serve skt/A.X-K2 --tensor-parallel-size 4 \
    --tool-call-parser hermes \
    --reasoning-parser deepseek_v3
```

- **`--tool-call-parser hermes`** — 모델은 툴 호출을 `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` 형태로 출력하며 hermes 파서가 이 형식을 그대로 처리합니다. 한 응답에 여러 개를 출력하는 parallel 호출도 각각 추출됩니다.
- **`--reasoning-parser deepseek_v3`** — 요청의 `chat_template_kwargs`에 담긴 `enable_thinking` 값을 보고 thinking이면 `</think>` 기준으로 추론 문장을 분리하고, 아니면 분리하지 않습니다.

**`deepseek_r1`을 쓰지 마세요.** non-thinking 프롬프트는 `</think>`로 끝나기 때문에 출력에는 `</think>`가 없는데, `deepseek_r1`은 종료 토큰이 없으면 출력 전체를 reasoning으로 분류합니다. 답변과 툴 호출이 모두 `reasoning` 필드에 묻히고 `content`는 비게 됩니다. `deepseek_v3`는 이 경우 분리를 건너뛰도록 위임합니다.

모드는 요청마다 전환합니다.

```python
client.chat.completions.create(
    model="skt/A.X-K2",
    messages=[...],
    tools=[...],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)
```

reasoning 추출이 툴 파싱보다 먼저 실행되므로 thinking과 툴을 함께 써도 추론 문장이 `content`나 툴 인자를 오염시키지 않습니다. hermes는 `arguments`를 JSON 문자열로 돌려주는데, 채팅 템플릿이 그 형식을 그대로 받아 다시 렌더링하므로 멀티턴 툴 사용에서 왕복이 맞물립니다.

### throughput 수치에 대해

세 스크립트 모두 `enforce_eager=True`로 실행합니다. 이는 정확성 검증을 위한 설정이며, CUDA graph가 비활성화되므로 출력되는 tok/s는 **성능 지표가 아닙니다**. 성능을 측정하려면 `enforce_eager`를 제거하고 vLLM의 벤치마크 도구를 사용하세요.
