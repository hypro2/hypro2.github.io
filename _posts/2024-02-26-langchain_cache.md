---
layout: post
title: 랭체인 CacheBackedEmbeddings으로 캐시 임베딩 만들기
---

임베딩은 재계산을 피하기 위해 저장되거나 임시로 캐시될 수 있습니다. 임베딩 캐싱은 CacheBackedEmbeddings를 사용하여 수행할 수 있습니다. 

캐시 백드 임베더는 임베딩을 키-값 저장소에 캐시하는 래퍼입니다. 텍스트는 해싱되고 해시가 캐시에서 키로 사용됩니다.

CacheBackedEmbeddings를 초기화하는 주요한 방법은 from_bytes_store입니다. 



## 매개변수
**underlying_embedder** : 임베딩에 사용할 임베더입니다. OpenAIEmbeddings나 HuggingFaceEmbeddings를 사용합니다.

**document_embedding_cache**: 문서 임베딩을 캐싱하기 위한 ByteStore입니다.

**namespace**: (옵션, 기본값은 "") 문서 캐시에 사용할 네임스페이스입니다. 이 네임스페이스는 다른 캐시와의 충돌을 피하기 위해 사용됩니다. 예를 들어, 임베딩 모델의 이름으로 설정하는 것을 추천합니다.

## Memory
CacheBackedEmbeddings는 cache를 저장하는 store로는 LocalFileStore와 InMemoryByteStore를 사용 할 수 있습니다. 

InMemoryByteStore의 경우, 메모리에 사용하는 InMemoryCache와 비슷한 작업을 하는데, InMemoryCache를 대신 넣어 사용 할 수 있습니다.

메모리에서 사용하는 만큼 휘발될 가능성이 있어서 권장되지는 않습니다. 아래는 간단히 함수로 구현해본 코드입니다. 

## 코드

```
from langchain.storage import InMemoryByteStore
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

def cache_embed_wrapper(embedding_model, local_store_path=None):
    if local_store_path is not None:
        store = LocalFileStore(local_store_path)
    else:
        store = InMemoryByteStore()

    cache_embed_wrapper = CacheBackedEmbeddings.from_bytes_store(embedding_model,
                                                                 document_embedding_cache=store,
                                                                 namespace=embedding_model.model)
    return cache_embed_wrapper
```

embedding_model.underlying_embeddings 밑에 embedding_model이 들어 있는 모습을 볼 수 있습니다. 

그 외 동작은 embedding_model에서 사용한 명령어를 그대로 사용하면 됩니다. 
