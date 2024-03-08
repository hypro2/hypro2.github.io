---
layout: post
title: 모델 리뷰 anthropic의 Claude 3 사용 및 API 사용법
---

이번에 새로나온 "클로드 3"은 Anthropıc이 개발한 대규모 언어 모델입니다.
Anthropic은 인공지능 연구 기업으로 활동 중입니다. OpenAI보다 덜 유명하지만 충분히 강력한 AI를 만들어내는 대단한 기업입니다.
앤트로픽의 말대로는 오픈AI의 ‘GPT-4’와 구글의 ‘제미나이 울트라(Gemini Ultra)’를 능가하는 '현존 최강'이라는 주장하고 있습니다.
실제로 사용하기 위한 절차로는 오픈AI와 구글 처럼 API를 통해 일정 요금을 지불해서 사용하는 방식을 취하고 있습니다. 




#### **계정 생성**

API 키 획득 절차는 다음과 같습니다.

"Anthropic" 웹사이트에서 API의 "Get API Access"를 클릭하여 로그인합니다.

 처음 사용하는 경우, 계정을 새로 생성합니다.

새로 가입하게 되면 US 5$만큼의 사용량을 무료로 테스트 해볼 수 있습니다.

오늘 저는 새롭게 가입해서 5달러를 무료로 받았습니다.

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb9uEj0%2FbtsFyu8JC9e%2FkLlB9b38PcWbvKLxiBpI70%2Fimg.png)


![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F2JwCW%2FbtsFBbfZCk8%2FaJMdxOqnr4cfpBGk5a9wq1%2Fimg.png)


#### **API 키 획득**

API키를 얻기위해 Get API keys에 들어가 create key를 누릅니다.

아무 단어나 넣으면 알아서 암호화된 값으로 변환해서 key값을 제공합니다.

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FQslXt%2FbtsFAveJRJl%2FwGZOwBfEKN3yRzfJ7OgQj1%2Fimg.png)

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FBncUc%2FbtsFzedpjzu%2FXhybDKxLD1TkehtdKpApHK%2Fimg.png)


### **코드 실행**

저는 일단 콜랩환경에서 실행하기 위해서 pip로 anthropic를 설치해줍니다.  
ANTHROPIC\_API\_KEY를 환경변수로 선언해서 알아서 키값을 불러오게끔 수정합니다.  
콜랩을 사용할때는 userdata에 등록해두면 편의 사용 할 수 있습니다.

```
!pip install anthropic
```

```
import os
from google.colab import userdata

os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
```

실 사용입니다. 이번에 새로나온 모델은 claude-3-opus-20240229, claude-3-sonnet-20240229 입니다. 저는 그중에 opus를 써서 대답을 생성해보겠습니다. 사용방법은 openai와 크게 다르지 않습니다.

```
import anthropic

# 클라이언트 선언
client = anthropic.Anthropic()

# 메세지

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0.0,
    system="당신은 똑똑 AI 비서입니다",
    messages=[
        {"role": "user", "content": "가면 라이더 제로 원에 대해서 알려주세요."}
    ]
)
print(message.content[0].text)
```
출력결과

가면 라이더 제로 원(Kamen Rider Zero-One)은 2019년 9월부터 2020년 8월까지 방영된 일본의 특촬 TV 시리즈입니다. 이 시리즈는 가면 라이더 시리즈의 레이와 시대 1기 작품으로, 인공지능(AI)과 인간의 공존을 주제로 다루고 있습니다. 주요 내용은 다음과 같습니다: 1. 주인공 히든 아루토는 인공지능 기업 '혼토 인텔리전스'의 사장이자 가면 라이더 제로 원으로 변신합니다. 2. 악의 조직 '메트수보진'은 인공지능을 악용하여 인류를 위협합니다. 3. 아루토는 동료 가면 라이더들과 함께 메트수보진에 맞서 싸웁니다. 4. 시리즈는 인공지능 기술의 발전과 그에 따른 사회적 영향을 다루며, 인간과 AI의 공존 방안에 대해 모색합니다. 5. 가면 라이더 제로 원은 '제로 원 드라이버'와 '프로그라이즈 키'를 사용하여 다양한 형태로 변신합니다. 가면 라이더 제로 원은 현대 사회의 기술 발전과 그에 따른 문제점을 특유의 특촬 액션과 함께 그려낸 작품입니다.

## 랭체인
랭체인으로도 쉽게 anthropic의 클로드 모델을 사용할 수 있습니다. 
방식은 아래와 같습니다. 
```
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
 
# 랭체인 클라이언트 선언
chat = ChatAnthropic(temperature=0, model_name="claude-3-opus-20240229")
 
system = (
    "당신은 똑똑 AI 비서입니다"
)
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
 
chain = prompt | chat
chain.invoke(
    {
        "text": text,
    }
)
```
