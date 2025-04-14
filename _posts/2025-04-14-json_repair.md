---
layout: post
title: LLM이 만든 망가진 JSON, 어떻게 고치지? - json_repair로 해결하기
---

대형 언어 모델(LLM)을 활용하다 보면 이런 경험 한 번쯤 해보셨을 겁니다.  
**“JSON 포맷으로 결과를 반환해줘”** 라고 요청했는데, 결과는...

```json
{ "name": "ChatGPT", "skills": ["NLP", "reasoning" "code writing"] } // 쉼표 어디 갔니?
```

이처럼 괄호가 빠졌거나, 따옴표가 없거나, 아예 사람 말이 들어가 있는 JSON을 마주할 때마다, 하나하나 수작업으로 고치고 계셨다면 이 글이 큰 도움이 될 거예요.




## 🎯 `json_repair`란?

[`json_repair`](https://github.com/mangiucugna/json_repair)는 LLM이 만들어낸 **유효하지 않은 JSON 문자열을 자동으로 복구**해주는 파이썬 라이브러리입니다.  
LLM 특유의 실수 — 괄호 빠짐, 따옴표 누락, 잘못된 값 등 — 을 감지하고 자동으로 수정하여 **완전한 JSON으로 되살려줍니다.**

개발자는 Stefano Baccianella이며, “딱 이걸 고쳐주는 가볍고 믿을만한 라이브러리가 없어서 직접 만들었다”고 합니다.  
아직까지도 GPT-4o의 Structured Output 기능만으로는 완전한 JSON을 담보할 수 없기에, 여전히 실무에서 쓰이고 있다고 해요.

---

## 🛠 주요 기능 정리

### 1. JSON 구문 오류 자동 복구
- 누락된 따옴표, 쉼표, 괄호 등 수정
- `true`, `false`, `null` 등의 잘못된 사용 고침
- 깨진 key-value 구조 복구

### 2. 배열과 객체의 구조 수정
- 닫히지 않은 배열/객체 자동 보정
- 필요시 `null`이나 `""` 등 기본값으로 보완

### 3. 누락된 값 자동 완성
- 필드에 값이 없을 경우 기본값으로 채워 유효성 유지

### 4. LLM 특유의 "말 섞인 JSON" 정리
- JSON 외 불필요한 주석, 설명 문구 제거

---

## ⚙️ 사용법

### 설치
```bash
pip install json-repair
```

### 간단 사용 예
```python
from json_repair import repair_json

bad_json = "{ name: 'Alice', age: 30, hobbies: [reading, 'coding'] }"
fixed = repair_json(bad_json)
print(fixed)
# 👉 {"name": "Alice", "age": 30, "hobbies": ["reading", "coding"]}
```

### 완전한 객체로 바로 파싱하고 싶다면:
```python
import json_repair

obj = json_repair.repair_json(bad_json, return_objects=True)
print(obj["name"])  # Alice
```

### 파일로부터 읽기
```python
from json_repair import from_file

data = from_file("broken.json")
```

---

## 🌏 다국어 문자 처리

한글, 일본어, 중국어 등의 비 Latin 문자가 있는 경우엔 `ensure_ascii=False` 옵션을 꼭 사용하세요.

```python
repair_json("{'korean': '안녕하세요'}", ensure_ascii=False)
# 👉 {"korean": "안녕하세요"}
```

---

## ⚡ 성능 팁

- `return_objects=True`: 문자열로 다시 직렬화하지 않아서 더 빠릅니다.
- `skip_json_loads=True`: 처음부터 유효하지 않은 JSON인 걸 알고 있다면 이 옵션으로 더 빠르게 처리할 수 있어요.
- 외부 종속성 없이 구현되어 있어 어느 환경에서도 가볍게 작동합니다.

---


## 💬 마치며

`json_repair`는 단순하지만 강력합니다.  
특히 LLM과 자동화 파이프라인을 다루는 환경이라면 필수 도구라고 해도 과언이 아니죠.  
