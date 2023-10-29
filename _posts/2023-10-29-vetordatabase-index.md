---
layout: post
title: 벡터 데이터베이스와 벡터 인덱스 Faiss
---
주로 LLM 관련되서 중장기적인 기억을 담당하는 부분을 수행으로 복합 비정형 데이터를 위해 효율적으로 저장 및 검색을 하기 위해 개발된 데이터베이스 구조

복합 비정형 데이터를 저장하기 위해서는 데이터를 일련의 숫자로 구성된 임베딩으로 변환하는 것이 필요하다. 임베딩을 하는 이유는 한 공간안에 유사한 것은 비슷한 공간에 몰려있다는 것을 전제로 진행한다.

예시 그림 Mnist 3차원같이 한 공간에 표현할 수 있어야 한다. (에 표현한 그림이라서 2차원같지만 3차원)

![](https://blog.kakaocdn.net/dn/c7ICK5/btszk506KEn/CDZH5hSINW3WEcLfj3c6cK/img.png)

기존의 키-밸류 기반의 DB에서 이러한 복합 비정형 데이터를 찾을 때 상당히 느리고 정확하지 않은 문제점이 있어서 고안 됬다고 한다.

쿼리가 주어줬을 때 vecotor간의 거리를 계산해서 가장 가깝게 있는 것을 호출한다는 과정으로

![이미지 출처 : https://youtu.be/7WCRhW1Z8NI](https://blog.kakaocdn.net/dn/bi5H88/btszk7khk0W/XfFEaiYKHbmjrHiRosaZhk/img.png)

이미지 출처 : https://youtu.be/7WCRhW1Z8NI
주로 사용되는 거리 계산 법은 두가지로 유클리디안 거리와 코사인 거리 두가지를 가장 맣이 사용합니다. 유클리드 거리는 두 벡터 간의 직선 거리를 계산하는 방법입니다. 반면에 코사인 유사도는 두 벡터 간의 각도를 계산하여 유사도를 측정하는 방법입니다.

그렇다면 왜 자연어 처리에서는 보통 코사인 유사도를 사용하는지에 대해서 설명하자면, 첫째, 벡터의 크기를 무시할 수 있습니다.

벡터의 크기는 유클리드 거리를 계산할 때 중요한 역할을 합니다. 하지만 자연어 처리에서는 대부분 벡터의 크기가 달라집니다. 예를 들어, "cat"과 "fried chicken"의 벡터 크기는 다릅니다. 따라서 벡터 크기를 무시할 수 있는 코사인 유사도를 사용하는 것이 더 나은 결과를 가져올 수 있습니다.

둘째, 각도를 기반으로 유사성을 측정할 수 있습니다. 자연어 처리에서는 대부분의 경우 두 단어나 문장 벡터 간의 각도가 중요한 역할을 합니다.

예를 들어, "cat"과 "dog"은 유사한 단어이므로, 그들의 벡터 간의 각도는 작아집니다. 반면에 "cat"과 "car"은 서로 관련이 없는 단어이므로, 그들의 벡터 간의 각도는 크게 됩니다. 따라서 각도를 기반으로 유사성을 측정할 수 있는 코사인 유사도를 사용하는 것이 더 나은 결과를 가져올 수 있습니다.

결론적으로, 자연어 처리에서 코사인 유사도를 사용하는 이유는 두 벡터 간의 각도를 중요하게 생각하기 때문입니다. 또한, 코사인 유사도는 벡터의 크기를 무시할 수 있기 때문에 더 일반적으로 사용됩니다.

![이미지 출처 : https://youtu.be/7WCRhW1Z8NI](https://blog.kakaocdn.net/dn/MzZKY/btsznCDNzGc/FnoA8GzICT3huiq1wL3rRK/img.png)

이미지 출처 : https://youtu.be/7WCRhW1Z8NI

**벡터 데이베이스 종류**

주로 현재 오픈소스 개발자들은 Pinecone(파인콘), Chroma  API를 이용해서 많이 클라우드 기반 벡터 데이터베이스 통신을 사용을 많이 합니다. 

이러한 API를 사용해서 상당히 빠른 속도를 보여주지만 보안적인 면에서 외부 서버에 저장해야된다는 애로사항이 존재합니다.

Pinecone:  
설명: 속도, 확장성, 신속한 프로덕션 배포를 위해 설계된 관리형 벡터 데이터베이스.  
타입: 관리  
하이브리드 검색: 가능  
확장성: 대규모  
배포 옵션: 관리  
주목할만한 기능: SPLADE 희소 벡터에 대한 기본 지원

Weaviate:  
설명: 수십억 개의 데이터 개체로 원활하게 확장할 수 있도록 구축된 오픈 소스 벡터 검색 엔진.  
타입: 오픈 소스  
하이브리드 검색: 가능  
확장성: 수억 규모  
배포 옵션: 자체 호스팅, 관리  
주목할만한 기능: 효율적인 키워드 검색

Zilliz:  
설명: 수십억 규모의 데이터를 위해 설계된 관리형 클라우드 네이티브 벡터 데이터베이스.  
타입: 관리  
하이브리드 검색: 아니요  
확장성: 수억 규모  
배포 옵션: 관리  
주목할만한 기능: 다양한 기능, 전체 RBAC

Milvus:  
설명: 수십억 개의 벡터로 확장 가능한 오픈 소스 클라우드 네이티브 벡터 데이터베이스.  
타입: 오픈 소스  
하이브리드 검색: 아니요  
확장성: 수억 규모  
배포 옵션: 자체 호스팅, 관리  
주목할만한 기능: Zilliz와 유사한 기능, 클라우드 확장 가능

Qdrant:  
설명: 문서 및 벡터 임베딩을 저장할 수 있는 벡터 데이터베이스.  
타입: 둘 다 (오픈 소스 및 관리형 클라우드)  
하이브리드 검색: 아니요  
확장성: 지정되지 않음  
배포 옵션: 자체 호스팅, 관리형 클라우드  
주목할만한 기능: 유연한 배포 옵션

### 벡터 인덱스

 FAISS(Facebook AI Similarity Search) 같은 벡터 인덱스도 벡터 임베딩 검색을 개선하지만, DB의 기능을 가지고 있지는 않음

Faiss를 이용해서 vector index를 만들어보면 IndexFlatL2 인덱스(Euclidean Distance), IndexFlatIP(Cosine Distance)로 호출해서 사용할 수 있다. 

 [https://github.com/facebookresearch/faiss/wiki/Faiss-indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

 Faiss를 이용한 코드를 작성해보자. 
 아래 경우는 직접 코드를 작성해서 class로 쉽게 구현한 벡터서치 클래스이다.

```
from typing import List
import faiss
import numpy as np
from vector_search.i_vector_search import VectorSearchInterface
 
 
class FaissSearch(VectorSearchInterface):
 
    def __init__(self, sentences):
        self.index = None
        self.sentences: List[str] = sentences
 
    def build(self, vectors: np.array):
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        return self.index
 
    def search(self, query: np.array, k: int) -> np.array:
        if k >= len(self.sentences):
            k = len(self.sentences)
 
        distance: np.array
        indices: np.array
 
        distance, indices = self.index.search(query, k)
        return np.array(self.sentences)[indices[0]]
 
    def save(self, path):
        faiss.write_index(self.index, path)
 
    def load(self, path):
        self.index = faiss.read_index(path)
 
if __name__ == '__main__':
 
    sentence = np.array(["hello","my","name","is"])
    vectors = np.array([[1.2, 2.5, 0.8], [0.1, 1.0, 2.0], [2.0, 1.5, 0.3], [1.5, 0.2, 1.8]], dtype=np.float32)
    query = np.array([[1.1,2.2,0.5]], dtype=np.float32)
 
    fd = FaissSearch(sentence)
    fd.build(vectors)
 
    print(fd.search(query,500))
```
