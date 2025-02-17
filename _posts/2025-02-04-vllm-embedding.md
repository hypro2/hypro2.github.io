---
layout: post
title: vLLM으로 임베딩 모델 서빙
---

LLM에서는 최근 RAG에서 많이 쓰는 임베딩 모델을 일부 지원하기 시작함.  지원하는 아키텍쳐가 적지만, LLM기반과 가장 많이 사용되는 BAAI의 SOTA 모델과 LLM 기반의 임베딩 모델 위주로 지원함. 

과거 모델은 쓰기 어려우나, 앞으로 나올 새로운 BAAI의 버트 기반 SOTA 모델, LLM 기반의 SOTA 임베딩 모델로 임베딩 모델을 전환 한다면 유용 할 것이라고 생각함. 로컬에서 사용하려면 WSL을 사용해서 vllm 명령어를 통해서 실행 할 수 있다. 


 

**장점**

GPU: 페이지 어텐션을 이용해서 필요한 GPU 메모리는 2배 정도 절약

속도:  추론 속도는 7배 정도 절약

확장성 : OpenAI API 형식의 호출 방식을 지원,  LoRA 지원 등등 커스텀 모델의 확장성.

기대 효과: 임베딩 모델로 추론만 하는 과정(ex 클러스터링, 단순 유사도 비교, RAG 등등)에서 1개의 vLLM 서버로 API호출로 인해  매 작업마다 새로운 GPU서버가 필요하지 않음

 

**단점**

예전 모델들을 지원하지 않음, 기존 st 모델 (MPnet 모델 등등)은 지원 안 함.

기존 st와 완전히 동일한 임베딩 값을 만들지 않음. 바꾸게 된다면 두 개를 동시에 같이 쓰는 등의 통합은 어려움 (vllm으로 완전한 전환이 필요)

새로운 모델에 대한 확장성이 쉽지 않다. 경우에 따라 vllm 커스텀 코드를 만들어야되는 기술이 필요할 수 있음.

**비교**
지원 모델
이전에 많이 사용하던 MPnet모델들을 사용 할 수 가 없지만, 일부 버트는 BertModel이 지원해줘서 config.architectures = ["BertModel"] 바꿔준다면 계속 사용할 수 있다. 
 

**속도 비교**
vllm과 sentence transformer의 속도는 1.1만건 추론하는 데, 1분과 7분으로 약 7배 정도 차이가 남.

vllm과 달리, sentence transformer는 batch_size를 키워서 메모리를 더 사용했지만 더 느렸음.

코사인 유사도를 통해서 동일한 문서의 임베딩 끼리 비교 했을 때, 0.999999가 나오는 것으로 보아 거의 비슷한 임베딩을 생성한다고 확인함. ! 하지만 값은 확실히 다름 !
```
# vLLM
from vllm import LLM
 
# Create an LLM. # You should pass task="embed" for embedding models
model = LLM(
    model="BAAI/bge-large-en-v1.5",
    task="embed",
    enforce_eager=True,
)
 
outputs = model.embed(new)
 
embeds_list = []
for prompt, output in zip(prompts, outputs):
    embeds = output.outputs.embedding
    embeds_list.append(embeds)
 
# Processed prompts: 100%|██████████| 11314/11314 [01:02]

 
# SentenceTransformer
from sentence_transformers import SentenceTransformer, models
 
model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cuda')
 
embedding = model.encode(new, show_progress_bar=True)
 
# Batches: 100% |██████████| 354/354 [07:14<00:00,  1.95it/s]
 
 
# 동일성 비교
cosine_similarity(embeds_list, embedding) # 완전 대각선이 0.9999 동일함
 
"""
array([[0.9999997 , 0.71350806, 0.72416524, ..., 0.66439212, 0.68226502,
        0.75055861],
       [0.71345857, 0.99999968, 0.74491331, ..., 0.72500172, 0.70744717,
        0.70277408],
       [0.72411196, 0.74493437, 0.9999997 , ..., 0.74316506, 0.71033881,
        0.66227622]
"""
"""
vllm : array([ 0.04547119, -0.01013947,  0.0087738 , ..., -0.00678253,
       -0.01145172, -0.03146362])
st : array([ 0.04544434, -0.01014779,  0.00876168, ..., -0.00678259,
       -0.01146165, -0.03142513], dtype=float32)
 ```

특이사항 : 실험을 통해서 최근 모델에서는 비슷한 임베딩을 생성함. 대신 옛날 bert를 구동할 경우, pooling방식에 따라 값이 많이 차이가 난다.
"""
메모리 비교
메모리도 2배 정도 차이가 남.



**OPENAI API 호환**
vLLM은 OpenAI의 임베딩 호출을 지원하기  때문에 1개의 서버에서 HTTP 통신으로 임베딩 값을 호출 수 있음

```
!nohup vllm serve BAAI/bge-base-en-v1.5 --dtype auto --api-key token-abc123 > nohup.out &
 
from openai import OpenAI
 
openai_api_key = "token-abc123"
openai_api_base = "http://localhost:8000/v1"
 
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
 
models = client.models.list()
model = models.data[0].id
 
responses = client.embeddings.create(
    input=prompts,
    model=model,
)
 
for data in responses.data:
    print(data.embedding)
```

**랭체인**

랭체인에서도 바로 사용 할 수 있다. OpenAIEmbeddings을 이용하면 바로 사용 할 수 있다. 이때 중요한 점은 openai_api_base를 제대로 지정해야된다. 

```
from langchain_openai import OpenAIEmbeddings

emb_model = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    openai_api_base="http://localhost:8000/v1",
    openai_api_key="token-abc123")

emb_model.embed_query("A sentence to encode.")
```
왠지 cli화면에서는 모르겠지만 들어가는 텍스트가 깨지는 것을 볼 수 있지만, 실제로 사용할 때는 제대로 동작하는 것 같다. 

![image](https://github.com/user-attachments/assets/02303f49-d871-40c6-b12b-7177425efa8a)
