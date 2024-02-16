---
layout: post
title: 랭체인의 LCEL 문법
---
langChain Expression Language (LCEL)는 체인을 쉽게 구성 할 수있는 선언적인 방법입니다


[https://python.langchain.com/docs/expression\_language/get\_started](https://python.langchain.com/docs/expression_language/get_started)  


**기본 : 일자식 구성**  
프롬프트 + 모델 + 출력 파서

chain = prompt | model | output\_parser

![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/cd0c3440-1df3-4a7c-9bab-859725bf9397)


**분기 : RunnableParallel과 RunnablePassThrough를 이용**

![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/85f9699c-4ab7-4572-a6b3-66bdbcfdb10d)


**심화 : 사용자 함수와 함께, Runable하게 구성하기**

```
def format\_docs(docs):  
    backgrounds = \[\]  
    max\_token = 3000  
    count\_token = 0  
    for doc in docs:  
        count\_token += len(tokenizer.encode(doc.page\_content))  
        if count\_token < max\_token:  
            backgrounds.append(doc.page\_content)  
        else:  
            break  
    return "\\n\\n".join(backgrounds)  


    rag\_chain = (  
            {"backgrounds": retriever | format\_docs,  
             "language": RunnablePassthrough(),  
             "question": RunnablePassthrough()}  
            | chat\_prompt  
            | llm\_model  
            | StrOutputParser()  
    )  

rag\_chain.invoke({'language': language, 'question': question})  
```

※ retriever를 사용할 때, 여러개의 input이 들어가게 되면 질문은 꼭 "question"으로 해야된다. langchain 내부 retriever에서 그렇게 지정되어 있다.

![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/f403eb21-001d-42ca-a8a3-30d2389ff37a)

**좋은 예시)**

LCEL을 이용해서 복잡한 체인을 구성한 다음, invoke, stream, batch 마음껏 사용가능하고, 필요한 경우 앞단에 선언한 model, prompt등등만 수정하면 동일하게 돌아가게된다.

홈페이지 참조 : [https://python.langchain.com/docs/expression\_language/why](https://python.langchain.com/docs/expression_language/why)
