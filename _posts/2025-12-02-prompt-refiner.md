---
layout: post
title: LLM 입력 최적화 및 비용 절감을 위한 경량화 라이브러리 Prompt Refiner
---


상용 LLM 애플리케이션, 특히 RAG(검색 증강 생성) 시스템을 운영함에 있어 **API 비용 효율화**와 **컨텍스트 윈도우(Context Window) 관리**는 핵심적인 엔지니어링 과제입니다.

본 포스팅에서는 불필요한 토큰을 제거하여 API 비용을 절감하고, 제한된 토큰 예산 내에서 프롬프트를 동적으로 구성할 수 있도록 지원하는 파이썬 라이브러리, `prompt-refiner`를 분석합니다.



## 1\. 개요 및 핵심 가치

https://github.com/JacobHuang91/prompt-refiner

`prompt-refiner`는 LLM 입력 데이터의 전처리를 자동화하는 경량 라이브러리입니다. 외부 의존성(Dependency)을 최소화하여 설계되었으며, 프로덕션 환경에서의 안정성과 성능 최적화에 초점을 맞추고 있습니다.

* **토큰 비용 절감**: HTML 태그, 중복 공백, 보이지 않는 특수 문자를 제거하여 평균 10\~20%의 API 비용을 절감합니다.
* **스마트 컨텍스트 관리**: 시스템 프롬프트, RAG 문서, 대화 기록 간의 우선순위를 기반으로 토큰 한도 내에서 입력을 자동 구성합니다.
* **보안 강화**: 입력 데이터 내의 PII(개인 식별 정보)를 자동으로 감지하고 마스킹 처리합니다.
* **고성능**: 1,000 토큰 처리 당 0.5ms 미만의 오버헤드로, 전체 응답 지연 시간(Latency)에 미치는 영향이 미미합니다.

## 2\. 아키텍처 및 주요 모듈

이 라이브러리는 파이프라인(Pipeline) 패턴을 지원하며, 다음 4가지 핵심 모듈로 구성됩니다.

### 2.1 Cleaner (데이터 정제)

비정형 데이터에 포함된 노이즈를 제거하여 토큰 효율을 높입니다.

* `StripHTML`: HTML 태그 제거 및 Markdown 변환.
* `NormalizeWhitespace`: 불필요한 공백 및 개행 문자 정규화.
* `FixUnicode`: 제로 위스(Zero-width) 문자 등 유니코드 오류 수정.
* `JsonCleaner`: JSON 데이터의 Null 값 제거 및 압축.

### 2.2 Compressor (데이터 압축)

정보의 밀도를 높여 컨텍스트 윈도우를 확보합니다.

* `Deduplicate`: RAG 문서 간의 중복 내용을 제거.
* `TruncateTokens`: 문장 경계를 고려한 스마트한 토큰 절삭(Truncation).

## 3\. 구현 코드 예시

`prompt-refiner`는 파이프(`|`) 연산자를 활용하여 전처리 로직을 직관적으로 구성할 수 있습니다.

### 3.1 기본 전처리 파이프라인

HTML이 포함된 텍스트를 정제하고 개인정보를 비식별화하는 과정입니다.

```python
from prompt_refiner import StripHTML, NormalizeWhitespace, RedactPII

# 원본 데이터: HTML 태그와 불필요한 공백 포함
raw_input = "<div> User  Input \n\n with <email>test@example.com</email> </div>"

# 파이프라인 정의: HTML 제거 -> 공백 정규화 -> PII 제거
pipeline = (
    StripHTML() 
    | NormalizeWhitespace() 
    | RedactPII(redact_types={"email"})
)

# 실행 결과
clean_input = pipeline.run(raw_input)
# 결과: "User Input with [EMAIL]"
```

### 3.2 RAG 시스템을 위한 스마트 패킹 (Smart Packing)

토큰 제한(Max Tokens)이 있는 상황에서 RAG 문서를 동적으로 삽입하는 예시입니다.

```python
from prompt_refiner import MessagesPacker, ROLE_SYSTEM, ROLE_CONTEXT, ROLE_QUERY, StripHTML

# 1000 토큰 제한 설정
packer = MessagesPacker(max_tokens=1000)

# 1. 시스템 프롬프트 (최우선 순위)
packer.add("You are a helpful AI assistant.", role=ROLE_SYSTEM)

# 2. RAG 문서 삽입 (JIT 정제 적용)
# refine_with 파라미터를 통해 삽입 시점에 즉시 HTML 제거 수행
packer.add(
    "<div>...Very Long HTML Document...</div>", 
    role=ROLE_CONTEXT, 
    refine_with=StripHTML()
)

# 3. 사용자 질문 (높은 우선순위)
packer.add("Explain the document summary.", role=ROLE_QUERY)

# 패킹 실행: 토큰 예산을 계산하여 List[Dict] 형태로 반환
messages = packer.pack()

# OpenAI API 호출에 즉시 사용 가능
# openai.chat.completions.create(messages=messages, ...)
```

## 4\. 성능 벤치마크 및 결론

자체 벤치마크(SQuAD + RAG 시나리오) 결과, 해당 라이브러리 적용 시 다음과 같은 효과가 입증되었습니다.

* **비용 효율성**: 'Aggressive' 전략 적용 시 API 비용 최대 15% 절감.
* **품질 유지**: 토큰을 축소하더라도 응답 품질(Cosine Similarity)은 96.4% 이상 유지.
* **저지연(Low Latency)**: 10k 토큰 처리 시 약 2.5ms 소요 (네트워크 레이턴시 대비 무시할 수준).

**결론적으로**, `prompt-refiner`는 대규모 LLM 서비스를 운영하는 엔지니어링 조직에게 **비용 절감**과 **데이터 파이프라인의 간소화**라는 두 가지 이점을 동시에 제공하는 실용적인 도구로 평가됩니다.
