---
layout: post
title: Pydantic Class 란?
---

  
[https://docs.pydantic.dev/latest/](https://docs.pydantic.dev/latest/)

pydantic은 이제 안사용하는 곳이 없을 정도로 필수 라이브러리가 되었습니다.데이터 클래스와 자료구조 관리에 필수템이 되어 모든 개발자들이 사용하고 있습니다.랭체인에서는 pydantic한 구조 사용은 적극적으로 권장(거의 필수)하며 거의 모든 부분에서 사용되고 있습니다.


#### **pydantic을 사용하는 주요 기업**

**OpenAI** ([https://github.com/pydantic/pydantic/discussions/6372](https://github.com/pydantic/pydantic/discussions/6372))

"OpenAI는 ChatCompletions API의 새로운 펑션 콜링 기능에 JSON Schema를 사용하므로, Pydantic 2.0과 함께 작동하도록 JSON Schema 문서를 업데이트 할 예정이 있습니다."

**마이크로 소프트**([https://github.com/search?q=repo%3Amicrosoft%2FDeepSpeed%20%20pydantic&type=code](https://github.com/search?q=repo%3Amicrosoft%2FDeepSpeed%20%20pydantic&type=code)),

**페이스북**([https://github.com/search?q=org%3Afacebookresearch+pydantic&type=code](https://github.com/search?q=org%3Afacebookresearch+pydantic&type=code)),

**구글**([https://github.com/search?q=org%3Agoogle+pydantic&type=code](https://github.com/search?q=org%3Agoogle+pydantic&type=code)),

**인텔, 엔비디아, 애플, 어도비 등등** 안쓰는 곳이 없습니다.

#### **pydantic 클래스의 확장성**

  
딕셔너리에 { } 중괄호 속에서 관리하는 코드보다 class화 시킨 코드가 검색, 관리, 유지 보수 측면에서 활용성이 높다.

단순한 구조에서 당연히 적용이 가능하며 복합적인 구조에서도 관리를 편하게 해준다.

**단순한구조 예시**

이런경우는 수정할때 조금만 불편하면된다. 하지만 복잡한 경우는 번거로운 작업이 될 수 있다. 

```
class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")



{'name': 'Tagging',
 'description': 'Tag the piece of text with particular info.',
 'parameters': {'type': 'object',
  'properties': {'sentiment': {'description': 'sentiment of text, should be `pos`, `neg`, or `neutral`',
    'type': 'string'},
   'language': {'description': 'language of text (should be ISO 639-1 code)',
    'type': 'string'}},
  'required': ['sentiment', 'language']}}
```

**complex한 구조에서 예시**

information 안에 Person이 들어가게되고 옵션으로 age가 들어가는 complex한 경우

만약, person에 키, 몸무게를 추가하게 된다면 어떻게 수정해야될까?

사람 1, 사람 2, 사람 3이 될때는 어떻게 해야될까?

다른 파일에 동일한 function이 있을 때 수정을 어떻게 해야될까?

이러한 경우의 문제를 파이덴틱 클래스는 한번에 해결해주게된다.

```
class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="person's name")
    age: Optional[int] = Field(description="person's age")

class Information(BaseModel):
    """Information to extract."""
    people: List[Person] = Field(description="List of info about people")



{'name': 'Information',
 'description': 'Information to extract.',
 'parameters': {'type': 'object',
  'properties': {'people': {'description': 'List of info about people',
    'type': 'array',
    'items': {'description': 'Information about a person.',
     'type': 'object',
     'properties': {'name': {'description': "person's name", 'type': 'string'},
      'age': {'description': "person's age", 'type': 'integer'}},
     'required': ['name']}}},
  'required': ['people']}}
```

#### **pydantic 클래스의 유효성**

validator를 통해 생성된 필드에서 유형 검사를 진행해서 불필요한 경우를 미리 필터 해주는 케이스를 만들 수 있다.

```
class Joke(BaseModel):
    """
    setup과 punchline을 정의한다.
    """
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # 유형 검사, 생성된  setup 필드에 ?표로 끝나는지 간단히 검사 가능하다.
    @validator("setup")
    def question_ends_with_question_mark(cls, field):
        if field[-1] != "?":
            raise ValueError("Badly formed question!")
        return field
```
