---
layout: post
title:
---

리트리버 검색기를 통해서 쿼리에 참조 하기위한 문서를 빠르면서 정확하게 찾기위한 전략을 여러가지 짤 수 있습니다.  
LangChain이 제공하는 Retrieval Augmented Generation RAG는 외부 데이터를 검색하고 LLM으로 전달하여 사용자 특정 데이터를 활용하는 방법을 설명합니다.


![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/8f32565f-3276-4b1f-a090-fdebb40df41c)

LangChain은 다양한 모듈로 이 과정을 지원하는데, 문서 로더로 다양한 소스에서 문서를 불러오고, 문서 변환기로 문서를 적절히 가공합니다.  
또한 텍스트 임베딩 모델을 사용해 문서에 임베딩을 생성하고, 벡터 저장소를 통해 효율적으로 저장하고 검색합니다.검색 알고리즘 또한 다양한 방식으로 구현되어 있어,  
쉬운 의미적 검색부터 부모 문서 검색, 셀프 쿼리 검색, 앙상블 검색 등 다양한 방법을 사용할 수 있습니다.  
더불어 LangChain의 인덱싱 API는 중복 콘텐츠 작성과 임베딩 재계산 등을 방지하여 데이터를 효율적으로 처리하고 결과를 개선합니다.  
주로, 중요하다고 봐야 될 건 BM25, MMR 방식과 멀티 쿼리, 앙상블 리트리버 입니다.

**MMR 리트리버**

```
# Faiss 벡터 인덱스 생성
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
docsearch = FAISS.from_texts(texts=texts,  # 청크 배열
                             embedding=embeddings,  # 임베딩
                             metadatas=metadatas  # 메타데이터
                             )

found_docs = docsearch.max_marginal_relevance_search(query,
                                                     k=2, # 최종 가져올 문서
                                                     fetch_k=10 # 후보 문서 (fetch_k > k)
                                                     lambda_mult=0.5 # 람다 (다양성)
                                                     )
```

**BM25 리트리버**

키워드 기반의 랭킹 알고리즘 - BM25  
BM25(a.k.a Okapi BM25)는 주어진 쿼리에 대해 문서와의 연관성을 평가하는 랭킹 함수로 사용되는 알고리즘으로,TF-IDF 계열의 검색 알고리즘 중 SOTA 인 것으로 알려져 있다.  
IR 서비스를 제공하는 대표적인 기업인 엘라스틱 서치에서도 ElasticSearch 5.0서부터 기본(default) 유사도 알고리즘으로 BM25 알고리즘을 채택하였다.

```
from langchain.retrievers import BM25Retriever # pip install rank_bm25

retriever = BM25Retriever.from_texts(texts, k=4)

found_docs = retriever.get_relevant_documents(query)


sklearn을 이용한 리트리버
from langchain.retrievers import SVMRetriever #pip install lark
from langchain.retrievers import KNNRetriever, TFIDFRetriever


"""
SVM : 분류, 회귀 및 이상치 탐지에 사용되는 일련의 감독 학습 방법입니다.
"""

retriever = SVMRetriever.from_texts(texts,
                                    OpenAIEmbeddings(openai_api_key=openai_api_key),
                                    k=3)

found_docs = retriever.get_relevant_documents(query)



"""
KNN
"""


retriever = KNNRetriever.from_texts(texts,
                                    OpenAIEmbeddings(openai_api_key=openai_api_key),
                                    k=3)

found_docs = retriever.get_relevant_documents(query)



"""
TF-IDF
"""


retriever = TFIDFRetriever.from_texts(texts, k=3)

found_docs = retriever.get_relevant_documents(query)


# 저장

# retriever.save\_local("testing.pkl")
```

**멀티 쿼리 리트리버**  
거리 기반 벡터 데이터베이스 검색은 쿼리를 고차원 공간에 내장시키고 "거리"를 기반으로 유사한 내장 문서를 찾습니다.

그러나 검색 결과는 쿼리 문장의 미묘한 변경이나 임베딩이 데이터의 의미를 잘 포착하지 못하는 경우에 다른 결과를 나타낼 수 있습니다.

때로는 이러한 문제를 수동으로 해결하기 위해 프롬프트 엔지니어링/조율 작업이 수행될 수 있지만 이는 지루할 수 있습니다.

MultiQueryRetriever는 사용자 입력 쿼리에 대해 다양한 관점에서 여러 쿼리를 생성하는 LLM을 사용하여 프롬프트 조정 프로세스를 자동화합니다.

각 쿼리에 대해 해당하는 일련의 관련 문서를 검색하고 모든 쿼리에서 고유한 합집합을 취하여 잠재적으로 관련성이 있는 더 큰 문서 세트를 얻습니다.

동일한 질문에 대해 여러 관점을 생성함으로써 MultiQueryRetriever는 거리 기반 검색의 일부 한계를 극복하고 더 다양한 결과를 얻을 수 있을 것입니다.

```
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectordb = FAISS.from_documents(documents=texts, embedding=embedding)


question = "What are the approaches to Task Decomposition?"
llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(),
                                                  llm=llm)


question = "What are the approaches to Task Decomposition?" #  "작업 분해를 위한 접근 방식은 무엇인가요?"
unique_docs = retriever_from_llm.get_relevant_documents(query=question)
print(unique_docs)

"""
INFO:langchain.retrievers.multi_query:Generated queries: 
['1. How can Task Decomposition be approached?', '2. What are the different methods for Task Decomposition?', '3. What are the various approaches to decomposing tasks?']
['1. 작업 분해는 어떻게 접근할 수 있나요?', '2. 작업 분해에는 어떤 방법이 있나요?', '3. 작업 분해를 위한 다양한 접근 방식에는 어떤 것이 있나요?']
하나의 쿼리에 의존해서 찾는게 아니라 비슷한 쿼리를 생선하고 참고해서 찾음
"""
```

**앙상블 리트리버**  
EnsembleRetriever 검색기 리스트를 입력으로 가져 와서 결과를 앙상블 get\_relevant\_documents() 방법을 기반으로 결과를 리랭크 앙상블 알고리즘입니다.

다른 알고리즘의 장점을 활용하여 EnsembleRetriever 단일 알고리즘보다 더 나은 성능을 달성 할 수 있습니다.

가장 일반적인 패턴은 키워드기반 리트리버 (예 : BM25)와 조밀 한 리트리버 (예 : 유사성 포함)를 결합하는 것입니다.

"하이브리드 검색"이라고도합니다". 키워드 기반 리트리버는 키워드를 기반으로 관련 문서를 찾는 데 능숙하고, 벡터기반 리트리버는 의미론적 유사성을 기반으로 관련 문서를 찾는 데 능숙합니다.앙상블할 가중치를 통해 얼마나 유사한 문서인지 찾아냅니다.

```
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
faiss_vectorstore  = FAISS.from_documents(documents=texts, embedding=embedding)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 2

ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                       weights=[0.5, 0.5])

question = "What are the approaches to Task Decomposition?"
docs = ensemble_retriever.get_relevant_documents(question)
print(docs)
```

**컨텍스트 압축 리트리버**

검색에서의 한 가지 어려움은 일반적으로 문서 저장 시스템이 질의될 때 어떤 구체적인 질의가 발생할지를 알 수 없다는 점입니다.  
이는 질의와 가장 관련성 있는 정보가 많은 관련 없는 텍스트로 가득 찬 문서에 들어있을 수 있다는 것을 의미합니다.  
이러한 전체 문서를 응용 프로그램을 통과시키면 보다 비용이 많이 들고 응답이 더 나빠질 수 있습니다.  
이를 해결하기 위해 Contextual Compression(맥락적 압축)이 고안되었습니다.

아이디어는 간단합니다:  
검색된 문서를 그대로 반환하는 대신, 주어진 쿼리의 맥락을 사용하여 그것들을 압축하여 관련 정보만 반환할 수 있습니다.  
여기서 "압축"이란 개별 문서의 내용을 압축하는 것과 문서 전체를 걸러내는 것을 모두 의미합니다.  
Contextual Compression Retriever를 사용하려면기본 검색기(base retriever),문서 압축기(Document Compressor)가 필요합니다.  
Contextual Compression Retriever는 쿼리를 기본 검색기로 보내고, 초기 문서를 가져와 Document Compressor를 통과시킵니다.  
Document Compressor는 문서 목록을 가져와 문서 내용을 줄이거나 문서를 완전히 제거하여 그 크기를 줄입니다.

![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/c6238307-84bd-4dfb-b4d1-0f9b3178ad6e)

```
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter



documents = TextLoader('../dataset/state_of_the_union.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)[:10]
retriever = FAISS.from_documents(texts, OpenAIEmbeddings(openai_api_key=openai_api_key)).as_retriever()

llm = OpenAI(temperature=0,openai_api_key=openai_api_key)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

# 컨텍스트 압축
compressed_docs = compression_retriever.get_relevant_documents("What did the president say about Ketanji Jackson Brown")
print(compressed_docs)

# 임베딩 필터 적용 (비슷한거 0.76 이상만)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)

compressed_docs = compression_retriever.get_relevant_documents("What did the president say about Ketanji Jackson Brown")
print(compressed_docs)
```
