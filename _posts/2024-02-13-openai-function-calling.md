---
layout: post
title: OpenAI의 Function Calling의 원리를 알아보자
---


한 3달전에 캐글에 랭체인을 이용한 라마와 펑션콜링에 대한 주제로 글을 쓴게 있는데 1110뷰를 돌파 했습니다. ㅎㅎ

OpenAI의 function calling은 출시 됬을 때 chatgpt를 이용한 개발자들에게 혁신적인 인기가 있었습니다. 

이번 포스트는 Jsonformer를 통해서 OpenAI의 function calling이 무엇인가? 어떻게 구현 됬는지 알아보려고 합니다.  

![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/64d1135f-55d9-4848-bab3-f5bf7abe0d46)


모델이 구조화된 JSON을 출력하는 것은 문법적으로 올바르고 JSON 구조를 명시하는 스키마를 준수해야 합니다. 이 문제를 해결하는 것은 매우 어려운 작업입니다.

현재 이 문제에 대한 접근 방식은 부서지기 쉽고 오류가 발생하기 쉽습니다. 이는 프롬프트 엔지니어링, 세세한 조정 및 후처리에 의존하지만 여전히 많은 경우에 문법적으로 올바른 JSON을 생성하지 못합니다.

OpenAI의 Function Calling을 유사하게 구현한 Jsonformers를 통해 동작 원리을 파악할려고 합니다. Jsonformers는 function calling의 원리를 허깅페이스 transformers에 적용한 확장성이 높은 라이브러리입니다.

구조화된 데이터에서 많은 토큰은 고정을 하여, 예측이 가능합니다. Jsonformer는 Hugging Face 모델을 감싸고, 생성 과정에서 고정된 토큰만을 채우며, 언어 모델에게는 내용 토큰의 생성을 위임합니다.

딕셔너리 { }를 선언하고 properties(required)의 title(name)과 value의 타입을 먼저 확인 합니다. input과 함께 들어가서 생성하고 마지막에 딕셔너리를 출력합니다.

방법으로는 transformers에서는 LogitsWarper와 StoppingCriteria를 상속 받아, 생성하면 안되는 토큰의 로짓 값을 -inf로 만드는 방법과, 스탑 조건을 만들어서 poperties의 value 포맷에 맞게 끔 조정합니다.

![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/635296ba-ce1a-4fe1-824c-61615ac2aa9e)


-   number # 숫자외 모든 토큰의 확률을 제거함
-   boolean # True, False 외 모든 토큰의 확률을 제거함
-   string # 2번째 따옴표가 등장할때 까지 생성함
-   array # 반점(콤마) , 랑 닫기 대괄호 \] 이 등장할때 까지 생성함
-   object # array와 비슷

**예시)**

```
functions = [
    {
        "name": "get_current_location",
        "description": "주어진 지역을 알려줍니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",                               # 2번, 타입 확인 # ex) "string"이면 따옴표 안에 열기 따옴표부터 시작해서 문자열 토큰 생성하고 닫기 따옴표가 등장하면 stop하는 규칙을 이용
                    "description": "지역, e.g. 서울, 부산, 제주도", # 3번 description을 추가하고 LLM 생성 요청
                },
            },
            "required": ["location"],                               # 1번, 필요한 인자 확인 # 여러개면 여러번 확인
        },
    }
]

# 실제 출력물
"function_call": 
{
 "name": "get_current_location",
  "arguments": "{"location": "서울"}"
}
```

\# 출력물 빨간색은 미리 만들어지는 부분, # 파란색이 LLM이 생성하는 부분 

\# required에 따라서 "input" + "name, description"이 같이 LLM으로 전달되서,  빨간색 부분 까지  미리 만든 다음, 조건에 따라서 파란색 부분을 생성하는 것이 Function Calling의 동작원리.

작동의 단순화 시켜서 보여주기

**\### 입력**

**"여기는 서울역 근처입니다"**

**{name : "location",  description : "지역, e.g. 서울, 부산, 제주도"** }

**\### 출력**

**{** **"location" : "** (출력에 필요한 구조 미리 생성) **서울 "**  두번째 따옴표 생성 stop

 **}**  조건이 완료되면 닫기

**이러한 원리를 모른다면?**

![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/051073db-eb3f-4b4c-9a44-bb74c5d59fd7)


로컬 모델별로 Function calling 데이터를 잔뜩 모아서 훈련 시킨 모델을 돈주고 구매하는 일이 생겨납니다.

물론 이게 나쁘다는 것은 아닙니다. 보다 function calling의 개념을 훨씬 더 많이 이해한 모델일 수 있습니다. 

모델의 크기에 따라 30달러에서 100달러가 넘는 경우가 존재합니다. 하지만 전혀 그럴 필요 없습니다.
