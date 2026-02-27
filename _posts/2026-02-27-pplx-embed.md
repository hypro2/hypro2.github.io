---
layout: post
title: pplx-embed 확산 기반 사전학습을 통한 고성능 다국어 임베딩 시스템 구축
---

pplx-embed: 확산 기반 사전학습을 통한 고성능 다국어 임베딩 시스템 구축

pplx-embed는 Perplexity에서 공개한 실제 웹 규모 검색 작업에 최적화된 다국어 임베딩(Embedding) 모델 컬렉션입니다. Qwen3를 백본(Backbone)으로 하며, 확산 기반 사전학습과 양방향 주의(Bidirectional Attention) 메커니즘을 결합하여 기존 단방향 LLM 기반 임베딩 모델의 한계를 극복했습니다.





## 개념 및 배경

기존의 인스트럭션 튜닝(Instruction Tuning) 기반 임베딩 모델은 사용자가 입력하는 프롬프트 접두사(Prefix)에 따라 임베딩 공간이 가변적이라는 취약점이 있습니다. 또한, 대부분의 고성능 LLM이 채택하고 있는 인과적(Causal) 디코더 구조는 텍스트 전체의 양방향 문맥을 완벽하게 포착하는 데 한계가 있습니다. pplx-embed는 이러한 문제를 해결하기 위해 확산(Diffusion) 모델의 원리를 사전학습에 도입했습니다.


## 핵심 기술 메커니즘

### 1. 확산(Diffusion) 기반 사전학습

Qwen3(0.6B, 4B) 모델의 인과적 주의 마스크(Causal Attention Mask)를 제거하고, 확산 제거 목적 함수(Diffusion Denoising Objective)를 통해 지속적 사전학습(Continued Pretraining)을 수행합니다. 랜덤 마스킹된 토큰을 양방향 문맥으로 복원하는 과정을 통해 모델은 강력한 양방향 표현 능력을 획득하게 됩니다.

### 2. 4단계 분기형 대조 학습(Contrastive Learning)

사전학습 이후 다음의 네 단계를 거쳐 모델의 정밀도를 높입니다.

* **1단계 (Pair Training):** InfoNCE 손실 함수를 사용하여 쿼리와 문서 간의 시맨틱 정렬을 수행합니다. 영어에서 시작하여 다국어로 학습 범위를 점진적으로 확장합니다.
* **2단계 (Contextual Training):** `pplx-embed-context-v1` 모델을 위한 단계입니다. 청크와 문서 레벨에서 동시에 대조 학습을 적용하는 이중 손실(Dual Loss)을 사용하여 전체 문맥 반영 능력을 강화합니다.
* **3단계 (Triplet Training):** 하드 네거티브(Hard Negative) 마이닝을 통해 임베딩 경계의 정밀도를 극대화합니다.
* **4단계 (Checkpoint Merging):** 구면 선형 보간(SLERP, Spherical Linear Interpolation) 방식을 사용하여 컨텍스트 체크포인트와 트리플렛 체크포인트를 병합함으로써 최종 성능을 완성합니다.



## 주요 특징 및 최적화

### Native INT8 Quantization-Aware Training

사후 양자화가 아닌 훈련 과정 전체에서 임베딩을 INT8 정밀도로 유지합니다. 미분 불가능한 양자화 단계는 STE(Straight-Through Gradient Estimation)를 통해 역전파를 수행하며, 이를 통해 낮은 정밀도에서도 정보 손실을 최소화한 표현을 학습합니다.

### Late Chunking 전략

긴 문서를 처리할 때 각 청크를 독립적으로 임베딩하지 않고, 전체 문서를 모델에 통과시킨 후 나중에 청크 단위로 평균 풀링(Mean Pooling)을 수행합니다. 이 방식은 양방향 주의 메커니즘을 통해 각 청크가 문서 전체의 정보를 보존하게 만듭니다.

---

## 구현 방법

Python의 `sentence-transformers` 라이브러리를 사용하여 `pplx-embed-v1-0.6B` 모델을 구동하는 방법입니다.

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 모델 로드 (Hugging Face Hub)
model_name = "perplexity-ai/pplx-embed-v1-0.6b"

def run_embedding_example():
    try:
        # 모델 로딩 시 trust_remote_code 옵션이 필요할 수 있습니다.
        model = SentenceTransformer(model_name, trust_remote_code=True)
        
        # 테스트 데이터 정의
        queries = [
            "오늘 날씨가 너무 맑고 좋네요.",
            "화창한 하늘에 구름이 조금 있습니다.",
            "최신 AI 언어 모델의 추론 능력이 빠르게 발전하고 있습니다."
        ]
        
        # 인스트럭션 접두사 없이 임베딩 생성
        embeddings = model.encode(queries)
        
        # 유사도 분석 (코사인 유사도 권장)
        sim_positive = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        sim_negative = cosine_similarity([embeddings[0]], [embeddings[2]])[0][0]
        
        print(f"임베딩 차원: {embeddings.shape}")
        print(f"유사 문장 간 유사도: {sim_positive:.4f}")
        print(f"무관 문장 간 유사도: {sim_negative:.4f}")

    except Exception as error:
        print(f"모델 실행 중 오류 발생: {error}")

if __name__ == "__main__":
    run_embedding_example()

```



## 실행 결과 및 평가

Perplexity 내부 벤치마크 데이터셋인 `PPLXQuery2Query` 및 `PPLXQuery2Doc` 테스트 결과, pplx-embed 시리즈는 BGE-M3 및 기존 Qwen3-Embedding 모델을 상회하는 성능을 기록했습니다. 특히 `pplx-embed-context-v1`은 컨텍스트 텍스트 임베딩 벤치마크(ConTEB)에서 우수한 성과를 보이며 RAG(Retrieval-Augmented Generation) 시스템 구축에 최적화된 성능을 입증했습니다



## 참고 링크

* Arxiv 논문: [https://arxiv.org/pdf/2602.11151](https://arxiv.org/pdf/2602.11151)
* Hugging Face 저장소: [https://huggingface.co/perplexity-ai](https://huggingface.co/perplexity-ai)

추가적으로 `pplx-embed-context-v1`을 활용한 구체적인 RAG 파이프라인 최적화 사례가 필요하시면 말씀해 주시기 바랍니다.