---
layout: post
title: Gemini Pro API 사용해보기 (Python, Langchain)
---

제미나이 프로의 API 발급과 사용방법을 알아보고 Python에서 작동하는 것을 보고 랭체인에 적용해보자

# Gemini Pro API 사용료

현재(23년 12월) 무료 버전만 사용가능하고, 1분당 60번 호출 가능

유료버전은 내년 초 준비 중

[https://blog.google/technology/ai/gemini-api-developers-cloud/](https://blog.google/technology/ai/gemini-api-developers-cloud/)

<img width="769" alt="image" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FUyNya%2FbtsB7jbp4KF%2F4b0SFSmbk2mjJjdKkxlKB1%2Fimg.png">


Gemini Pro의 생산 비용은 1000 자당 $ 0.0005이며 GPT-4-Turbo （ $ 0.03 / 1K ）보다 훨씬 저렴

아직 사용자가 별로 없어서그런가 TPU를 사용해서 그런가 비교할때, GPT4보다 조금 빠름

# API 발급 받기

[https://ai.google.dev/?hl=en](https://ai.google.dev/?hl=en)
[https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

한번만 보여줌. 잘 복사해서 사용해야됨
<img width="769" alt="image" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F5h3lp%2FbtsCf36aEn7%2F9JOeJFTkTyyQXvEtGf4uhK%2Fimg.png">


# 실전 텍스트 모델


```
import pathlib
import textwrap
import google.generativeai as genai
from google.colab import userdata
from IPython.display import display
from IPython.display import Markdown
from google.colab import userdata
 
# TEXT 활용
genai.configure(api_key=userdata.get('GOOGLE_API_KEY')) #API를 입력해주세요!!!!!
model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("가면라이더의 역사를 알려줘.")
```
답변: 

가면라이더 시리즈는 이시노모리 쇼타로가 창작한 일본의 토쿠사츠 액션 히어로 시리즈입니다. 1971년 4월 3일 마이니치 방송(MBS)에서 첫 번째 시리즈인 가면라이더가 방영되었고, 그 이후로 현재까지 여러 시리즈가 방영되고 있습니다.

가면라이더 시리즈는 주인공이 가면을 쓰고 슈트를 입고 악의 조직과 싸우는 이야기입니다. 주인공은 대부분 오토바이를 타고 다니며, 특수 능력을 사용하여 적과 싸웁니다.

가면라이더 시리즈는 일본에서 매우 인기가 많으며, 많은 상품이 출시되고 있습니다. 또한, 가면라이더 시리즈를 원작으로 한 영화, 게임, 애니메이션도 제작되었습니다.

가면라이더 시리즈의 역사는 크게 3개의 시대로 나눌 수 있습니다.

-   **쇼와 시대 (1971-1989)**

쇼와 시대는 가면라이더 시리즈의 초기 시대입니다. 이 시대에는 가면라이더 1호, 가면라이더 2호, 가면라이더 V3 등의 인기 작품이 제작되었습니다.

-   **헤이세이 시대 (1989-2019)**

헤이세이 시대는 가면라이더 시리즈의 중기 시대입니다. 이 시대에는 가면라이더 쿠우가, 가면라이더 아기토, 가면라이더 덴오 등의 인기 작품이 제작되었습니다.

-   **레이와 시대 (2019-현재)**

레이와 시대는 가면라이더 시리즈의 후기 시대입니다. 이 시대에는 가면라이더 제로ワン, 가면라이더 세이버, 가면라이더 리바이스 등의 인기 작품이 제작되었습니다.

가면라이더 시리즈는 일본의 대표적인 토쿠사츠 액션 히어로 시리즈입니다. 가면라이더 시리즈는 50년이 넘는 역사를 가지고 있으며, 현재까지도 많은 사람들에게 사랑받고 있습니다.

※ 놀라운 점 : 없음 GPT랑 비슷함

※ 아쉬운 점 : 토쿠사츠(특촬)을 일본어를 그대로 차용했고 한국말로 제대로 못함, 제로ワン(제로원)을 대답 못함 이것저것 GPT랑 비슷한 정도 한글을 잘하는 것 같지는 않음

# 실전 비전 모델

```

import PIL.Image
 
img = PIL.Image.open("/content/다운로드.jpg")
 
odel = genai.GenerativeModel('gemini-pro-vision')
response = model.generate_content([
    "이미지 속 사람이 되서 저한테 조언해주세요.", 
    img
], stream=True)
response.resolve()
response.text
```

<img width="769" alt="image" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbsSjmb%2FbtsCc5wFmlV%2FvVC8mL0vcE9Www5nnKSxlK%2Fimg.png">


### 예시1)

※ 놀라운 점 : 해당 만화에 대해서 잘모르는데, 등장인물이  '가면라이더 위저드' 를 정확히 맞춤

※ 아쉬운 점: 일본 방영일 2012년 9월 2일 ~ 2013년 9월 29일, GPT만큼 적당히 **할루시네이션**이 존재함


### 예시2) 

<img width="500" alt="image" src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbLSZXJ%2FbtsB6mmnJnn%2FnrpAeBLoVW6Y2EHkIW8H0K%2Fimg.png">



# LangChain 사용법

링크:

[https://python.langchain.com/docs/integrations/chat/google\_generative\_ai](https://python.langchain.com/docs/integrations/chat/google_generative_ai)

### 텍스트

```
# pip install -U --quiet langchain-google-genai pillow
 
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-pro")
 
# 텍스트 생성
result = llm.invoke("Write a ballad about LangChain")
print(result.content)
 
# 스트리밍
for chunk in llm.stream("Write a limerick about LLMs."):
    print(chunk.content)
```

### 비전 모델

이미지를 제공하려면 목록\[딕셔너리\] 유형의 콘텐츠가 포함된 휴먼 메시지를 전달하고,

각 딕셔너리에는 이미지 값(이미지\_url 유형) 또는 텍스트 값(텍스트 유형)이 포함됩니다.

image\_url의 값은 다음 중 하나가 될 수 있습니다:

-   A public image URL
-   A local file path
-   A base64 encoded image
-   A PIL image

  
해당 이미지 포맷을 지원함

```
import requests
from IPython.display import Image
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
 
 
image_url = "https://picsum.photos/seed/picsum/300/300"
content = requests.get(image_url).content
#Image(content)
 
llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")
 
# example
message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What's in this image?",
        },  # 선택적으로 텍스트 부분을 제공할 수 있습니다.
 
        {"type": "image_url", "image_url": image_url},
    ]
)
llm.invoke([message])
```
