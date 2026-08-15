---
layout: post
title: LLM 추론 가속의 정점 추측 디코딩(Speculative Decoding)부터 MTP와 DeepSeek DSpark까지
tags: [LLM, 추론최적화, SpeculativeDecoding, MTP, DeepSeek, DSpark]
---

## LLM 추론 가속의 정점: 추측 디코딩(Speculative Decoding)부터 MTP와 DeepSeek DSpark까지

대규모 언어 모델(LLM)을 서빙할 때 가장 큰 고민은 단연 '토큰 생성 속도(Tokens/sec)'와 '비용'입니다. 프롬프트가 길어지고 모델 파라미터가 100B, 671B로 커질수록 한 글자 한 글자 출력되는 속도는 답답할 정도로 느려집니다.

흔히 LLM 연산이 느린 이유를 "GPU 연산량(FLOPs)이 부족해서"라고 생각하기 쉽지만, 사실 진짜 병목은 메모리 대역폭(Memory Bandwidth)에 갇힌 Memory-Bound 현상에 있습니다.

본 포스팅에서는 LLM 추론 지연의 근본 원인인 루프라인 모델(Roofline Model)을 살펴보고, 이를 극복하기 위해 진화해 온 추측 디코딩(Speculative Decoding), DeepSeek-V3의 MTP(Multi-Token Prediction), 그리고 2026년 최신 프레임워크인 DSpark의 아키텍처와 작동 원리를 심층 분석합니다.

---

### 1. 왜 LLM 추론은 느릴까? (Memory-Bound와 Memory Wall)

트랜스포머 기반 LLM의 텍스트 생성은 이전 토큰들을 보고 다음 토큰 1개를 생성하는 자기회귀(Autoregressive) 디코딩 방식입니다.

```
루프라인 모델(Roofline Model) 관점:
- 70B 모델 기준 1개 토큰을 생성할 때:
  1. GPU 연산 코어는 수십~수백 GB의 모델 가중치를 VRAM에서 통째로 한 번 읽어옴 (약 10ms 소요)
  2. 가중치를 읽어오는 동안 GPU 연산 코어의 95% 이상은 아무 일도 하지 않고 유휴(Idle) 상태로 대기
  3. 연산 집약도(Arithmetic Intensity) ≈ 1 FLOP / Byte (심각한 메모리 대역폭 낭비)
```

이 비효율을 타파하기 위해 떠오른 핵심 아이디어가 바로 이것입니다:

> "어차피 가중치를 VRAM에서 한 번 읽어오는 데 10ms가 걸린다면, 한 번 읽어온 김에 놀고 있는 연산 코어로 토큰 여러 개를 한꺼번에 병렬 검증(GEMM 연산)하자!"

이것이 추측 디코딩(Speculative Decoding)의 출발점입니다.

---

### 2. 추측 디코딩의 기본 메커니즘 (Draft & Verify)

트랜스포머는 Causal Masking 구조 덕분에 "입력으로 들어간 시퀀스 길이 $N$만큼, 각 위치별 다음 토큰의 로짓(Logit) $N$개를 단 1회의 순전파(Forward Pass)로 동시 계산"할 수 있습니다.

```
[1턴 생성 및 검증 사이클 예시]

1. 초안 제안 (Drafting):
   - 입력: "오늘"
   - 경량 초안 모델이 4개 단어 후보를 고속 생성: ["날씨가", "정말", "추워요", "!"]

2. 메인 모델 1회 병렬 검증 (Verification):
   - 메인 모델에 통째로 입력: ["오늘", "날씨가", "정말", "추워요"]
   - 단 1번의 순전파(Forward Pass)로 각 위치별 정답 로짓 동시 산출:
     • 위치 1 ("오늘" 다음): 초안 "날씨가" == 메인 모델 1등 로짓 (일치 ✅)
     • 위치 2 ("오늘 날씨가" 다음): 초안 "정말" == 메인 모델 1등 로짓 (일치 ✅)
     • 위치 3 ("오늘 날씨가 정말" 다음): 초안 "추워요" != 메인 모델 1등 로짓 "좋아요" (불일치 ❌)

3. 수정 및 즉시 확정 (Replacement & Lossless Quality):
   - 틀린 3번째 단어를 메인 모델의 정답 로짓인 "좋아요"로 즉시 교체
   - 틀린 시점 이후의 4번째 단어("!")는 폐기
   - ➔ 결과: 단 1회 메인 모델 실행으로 ["오늘 날씨가 정말 좋아요"] (3개 토큰) 획득! (수학적 무손실)
```

추측 디코딩의 가장 큰 장점은 메인 모델의 출력 확률 분포를 100% 보존(Lossless)한다는 점입니다. 초안 모델이 틀리더라도 메인 모델의 검증 단계에서 즉시 정답 로짓으로 교체되므로 모델 성능 저하가 전혀 없습니다.

---

### 3. MTP (Multi-Token Prediction): DeepSeek-V3의 내장형 진화

전통적인 추측 디코딩은 별도의 소형 드래프트 모델(예: LLaMA-70B를 위해 LLaMA-7B를 함께 띄움)을 사용했습니다. 하지만 이 방식은 VRAM을 이중으로 점유하고, 두 모델 간 학습 데이터/지식 분포 차이로 인해 초안 채택률(Acceptance Rate)이 떨어진다는 한계가 있었습니다.

DeepSeek-V3는 이를 해결하기 위해 MTP(Multi-Token Prediction)를 도입했습니다.

```
                 ┌────────────────────────────────┐
                 │ MTP Module (단 1개 경량 층)    │ ➔ 1.7% 파라미터 오버헤드
                 ├────────────────────────────────┤   (임베딩 / LM Head 공유)
                 │                                │
                 │      메인 트랜스포머 스택      │
                 │         (DeepSeek-V3)          │
                 │                                │
                 └────────────────────────────────┘
```

 극소 파라미터 오버헤드: 메인 모델 상단에 단 1개의 경량 트랜스포머 층을 덧붙인 구조로, 전체 가중치(671B) 대비 약 1.7%(11.5B)에 불과합니다.
 표현력 향상: 사전 학습 단계부터 미래 $K$개 토큰을 내다보며 예측하도록 훈련되어 모델 자체의 추론 및 계획 능력이 향상됩니다.
 추론 시 내장 초안기: 별도 모델 로딩 없이 메인 모델 내부에서 2~3배의 추론 가속을 무손실로 실현합니다.

---

### 4. DSpark: MTP의 한계를 넘은 차세대 추측 디코딩

DeepSeek-AI가 2026년 발표한 DSpark (DeepSpec 프로젝트)는 MTP와 기존 프레임워크(Eagle3, DFlash)의 병목을 해결한 최첨단 프레임워크입니다.

```
       [입력 프롬프트]
              │
              ▼
   ┌───────────────────────────────────────────────────────────┐
   │ [DSpark 초안 생성]                                        │
   │  1. Parallel Backbone (대략적인 틀을 병렬로 초고속 추출)      │
   │  2. Lightweight Sequential Head (단어 간 문맥 선 따기 보정) │
   │  3. Confidence Head (단어별 신뢰도 점수 자체 측정)           │
   └──────────────────────────────┬────────────────────────────┘
                                  │
                                  ▼ (신뢰도 낮은 뒷단어는 사전 절단 ✂️)
   ┌───────────────────────────────────────────────────────────┐
   │ [메인 모델 1회 검증] (Confidence-Scheduled Verifier)        │
   │  - 필요한 앞부분만 깔끔하게 로짓 계산                        │
   │  - GPU 불필요 연산 낭비 차단 ➔ Throughput 60~85% 추가 향상  │
   └───────────────────────────────────────────────────────────┘
```

#### 1) 준자기회귀 초안 생성 (Semi-Autoregressive Draft Head)
초안을 순차적으로 1개씩 생성하면 드래프트 단계 자체가 느려지고, 반대로 4개 단어를 완전 병렬로 뽑으면 뒤쪽 단어로 갈수록 문맥이 깨지는 접미사 예측 정확도 저하(Suffix Decay)가 발생합니다.  
DSpark는 병렬 백본(Parallel Backbone)으로 대략적인 뼈대를 잡고 경량 순차 헤드로 문맥을 연결하여, 초고속 생성 속도와 높은 채택률을 동시에 달성했습니다.

#### 2) 신뢰도 기반 동적 절단 검증 (Confidence-Scheduled Verifier)
실제 고부하 서빙(High Concurrency) 환경에서는 4개 단어 중 3, 4번째 단어가 틀릴 것이 뻔한데도 메인 모델에 4개 단어 전체를 검증 요청하면 GPU 연산이 낭비됩니다.  
DSpark는 초안 생성 시 자체 신뢰도(Confidence) 점수를 측정하여, 기준치 이하의 불확실한 뒷단어는 메인 모델에 보내지 않고 사전에 잘라냅니다(Early Truncation). 이를 통해 서빙 처리량(Throughput)을 60~85% 추가 향상시켰습니다.

---

### 5. 핵심 기술 비교 총정리

| 구분 | 전통적 추측 디코딩 | MTP (DeepSeek-V3) | DSpark (최신 진화) |
| :--- | :--- | :--- | :--- |
| 초안 생성 방식 | 독립된 별도 소형 LLM | 메인 모델 상단 1개 층 | 준자기회귀 (병렬 뼈대 + 순차 보정) |
| 검증 메커니즘 | 단순 순전파 1회 검증 | 메인 모델 1회 병렬 검증 | 신뢰도 기반 동적 조기 절단 검증 |
| VRAM 오버헤드 | 큼 (소형 모델 별도 로드) | 매우 작음 (~1.7%) | 매우 작음 (모듈형 경량 헤드) |
| 동시접속 처리량 | 낮음 (배치 증가 시 병목) | 중간 | 최상 (불필요 GEMM 연산 차단) |
| 품질 손실 여부 | 100% 무손실 (Lossless) | 100% 무손실 (Lossless) | 100% 무손실 (Lossless) |

---

### 6. 실전 오픈소스 툴킷: DeepSpec

DeepSeek-AI는 DSpark, DFlash, Eagle3 알고리즘을 누구나 직접 학습하고 평가할 수 있도록 [DeepSpec](https://github.com/deepseek-ai/DeepSpec) 저장소를 오픈소스로 공개했습니다.

```bash
# DeepSpec 설치 및 평가 예시
git clone https://github.com/deepseek-ai/DeepSpec.git
cd DeepSpec
pip install -r requirements.txt

# Qwen3-4B 타깃 모델에 대한 DSpark 초안 모델 평가
bash scripts/eval/eval.sh
```

---

### 마치며

LLM 가속의 역사는 "메모리 벽(Memory Wall)을 어떻게 우회할 것인가"의 역사였습니다.

단순한 프롬프트 최적화나 양자화를 넘어, 하드웨어 특성을 영리하게 이용하는 추측 디코딩과 MTP, 그리고 DSpark의 신뢰도 기반 동적 스케줄링은 고성능 LLM 서비스를 구축하려는 엔지니어에게 필수적인 핵심 아키텍처로 자리 잡았습니다.

 관련 논문: [DSpark: Confidence-Scheduled Speculative Decoding (arXiv:2607.05147)](https://arxiv.org/abs/2607.05147)
 GitHub Repository: [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec)
