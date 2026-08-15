---
layout: post
title: LettuceDetect 어디가 거짓인지 딱 짚어내는 초경량 스팬 단위 할루시네이션 탐지기
tags: [LLM, RAG, 할루시네이션, 오픈소스, AI에이전트]
---

## LettuceDetect: 어디가 거짓인지 딱 짚어내는 초경량 스팬 단위 할루시네이션 탐지기 🥬🔍

RAG(검색 증강 생성)나 AI 코딩 에이전트를 실무 프로덕션에 배포할 때 가장 까다로운 장벽 중 하나는 바로 **출력물의 사실 검증(Grounding Verification)**입니다.

기존의 `LLM-as-a-judge` 방식은 답변 전체를 두고 "이 문장은 맞다/틀리다(True/False)" 혹은 0~5점 척도로 평가하는 데 그쳤습니다. 이로 인해 다음과 같은 실무적 한계가 발생했습니다:

1. **불투명한 이진 판정 (Binary Output)**: 전체 문단 중 정확히 몇 번째 글자(어느 단어, 수치, API)가 허위인지 알 수 없습니다.
2. **비싼 비용과 높은 지연 시간 (Latency)**: 검증을 위해 거대 LLM을 다시 호출하므로 API 비용과 응답 대기 시간이 2배로 증가합니다.
3. **과잉 거절 (Over-flagging)**: 95%가 정확하고 단 하나의 수치가 틀렸을 뿐인데 전체 답변을 통째로 폐기해 버립니다.

이러한 문제를 해결하기 위해 등장한 오픈소스 프레임워크가 바로 **[LettuceDetect (KRLabsOrg/LettuceDetect)](https://github.com/KRLabsOrg/LettuceDetect)**입니다.

---

### 1. LettuceDetect란?

**LettuceDetect**는 RAG 검색 문서, 소스 코드, 도구(CLI/API) 실행 결과 등 **출처 컨텍스트(Source Evidence)**를 바탕으로, AI 생성물에서 근거가 없거나 모순된 부분을 **문자 스팬(Span-level, 시작/끝 글자 오프셋) 및 토큰 단위**로 정밀 국소화(Localization)해 내는 특화 검증 프레임워크입니다.

```
[입력 예시]
- Context : "프랑스의 수도는 파리이며, 인구는 약 6,700만 명이다."
- Question: "프랑스 인구는 몇 명인가요?"
- Answer  : "프랑스의 인구는 6,900만 명입니다."

[LettuceDetect 검출 결과]
-> Span: start=13, end=23 ("6,900만 명")
-> Category: Contradiction (모순)
-> Confidence: 0.994
```

---

### 2. 핵심 아키텍처와 검증 메커니즘

LettuceDetect는 무거운 생성형 모델 대신 **토큰 분류 헤드(Token Classification Head)**와 **스팬 분류 헤드(Taxonomy Head)**가 결합된 경량 인코더/소형 모델 구조(ModernBERT, EuroBERT, Qwen-2B)를 채택했습니다.

```
[Context / Code / Tool Logs] + [Question] + [Generated Answer]
                            │
                            ▼
              ┌───────────────────────────┐
              │ ModernBERT / Qwen-2B      │
              │ Cross-Attention 인코더    │
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Token Classification      │
              │ & Taxonomy Head           │
              └─────────────┬─────────────┘
                            ▼
  [Span Output: start, end, category, confidence]
```

#### 지원 모델 라인업
* **ModernBERT 기반 (`lettucedect-base-modernbert-en-v1`)**: 최신 Flash Attention 및 4K 컨텍스트를 지원하여 Sub-100ms 수준의 초고속 추론이 가능합니다.
* **EuroBERT 기반**: 8K 컨텍스트 및 영어/독어/불어/스페인어 등 7개 국어 다국어 검증을 지원합니다.
* **Qwen-2B 기반 (`lettucedect-v2-qwen-2b`)**: 2026년 6월 릴리즈된 v2 모델로, 코드 및 툴 호출 결과까지 정밀 검증합니다.
* **TinyLettuce (17M ~ 68M)**: 모바일/에지 디바이스 및 초저지연 온프레미스 서버용 초경량 임베디드 모델입니다.

---

### 3. 2026 v2 최신 특징: 코딩 에이전트 & 도구 출력 검증

최근 릴리즈된 LettuceDetect v2는 일반 줄글 문서뿐만 아니라 **코딩 에이전트(Claude Code, Antigravity, OpenManus 등)의 코드 생성물**까지 검증 영역을 대폭 확장했습니다.

* **존재하지 않는 가짜 API/식별자 포착**: 저장소 소스코드에 정의되지 않은 가짜 메서드나 잘못된 인자(Argument)를 정확히 추출합니다.
* **개발 도구 출력 왜곡 탐지**: 터미널 실행 결과나 Linter 피드백을 왜곡하여 사용자에게 보고하는 행위를 검출합니다.
* **벤치마크 성능**: 통합 테스트셋 기준 **스팬 F1 0.689**를 달성하여, 대형 LLM 판정 모델의 과잉 플래깅(False Positive) 한계를 효과적으로 극복했습니다.

---

### 4. 실전 코드: Python으로 5줄 만에 검증하기

#### 설치
```bash
pip install lettucedetect -U
```

#### Python 추론 예제
```python
from lettucedetect.models.inference import HallucinationDetector

# 1. 인코더 기반 스팬 검출기 초기화
detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
)

# 2. 검증할 컨텍스트, 질문, 답변 준비
context = ["France is a country in Europe. The population of France is 67 million."]
question = "What is the population of France?"
answer = "The population of France is 69 million."

# 3. 스팬 단위 환각 탐지 수행
predictions = detector.predict(
    context=context,
    question=question,
    answer=answer,
    output_format="spans",
    min_confidence=0.8
)

print("검출된 할루시네이션 스팬:", predictions)
# 출력: [{'start': 26, 'end': 38, 'confidence': 0.994, 'text': ' 69 million.'}]
```

#### FastAPI Web API 마이크로서비스 서빙
LettuceDetect는 자체 Web API 서버를 내장하고 있어 사내 RAG 파이프라인의 독립적인 가드레일 마이크로서비스로 즉시 띄울 수 있습니다:

```bash
pip install lettucedetect[api]
python -m lettucedetect_api.server --port 8000 --model KRLabsOrg/lettucedect-base-modernbert-en-v1
```

```python
from lettucedetect_api.client import LettuceClient

client = LettuceClient(base_url="http://localhost:8000")
result = client.detect(
    context=context,
    question=question,
    answer=answer
)
```

---

### 5. 프로덕션 아키텍처 적용 전략

LettuceDetect를 프로덕션 RAG 및 에이전트 파이프라인에 적용하면 다음과 같은 아키텍처를 구현할 수 있습니다:

1. **초저지연 실시간 가드레일 (Real-time Guardrail)**  
   답변을 사용자에게 스트리밍하기 직전 30~50ms 내외로 인코더 검증을 통과시켜, 허위 정보가 포함된 경우 즉시 경고 배지를 부착하거나 출력을 차단합니다.
2. **선택적 부분 재생성 (Selective Targeted Re-generation)**  
   전체 답변을 처음부터 다시 생성하지 않고, 오류가 발생한 스팬(`start`~`end`) 위치만 타깃팅하여 해당 문장만 재작성(Re-prompt)함으로써 토큰 낭비와 지연 시간을 획기적으로 줄입니다.
3. **UI 시각화 및 투명성 증대**  
   사용자 화면에서 근거가 확실한 텍스트는 정상 표시하고, 근거가 의심되는 텍스트는 노란색/빨간색 하이라이트로 시각화하여 AI 생성물의 신뢰도를 극대화할 수 있습니다.

---

### 마치며

AI 답변 검증은 이제 "맞다/틀리다"의 단순 이진 분류를 넘어 **"정확히 어느 단어가 왜 틀렸는지"를 명시하는 세밀한 국소화(Localization) 단계**로 발전하고 있습니다.

비싼 LLM API 비용을 아끼면서 밀리초 단위로 동작하는 정밀한 팩트체크 가드레일을 구축하고자 한다면 **LettuceDetect**를 적극 검토해보시길 권장합니다.

* **GitHub Repository**: [https://github.com/KRLabsOrg/LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect)
* **HuggingFace Dataset**: [KRLabsOrg/lettucedetect-code-hallucination](https://huggingface.co/datasets/KRLabsOrg/lettucedetect-code-hallucination)
