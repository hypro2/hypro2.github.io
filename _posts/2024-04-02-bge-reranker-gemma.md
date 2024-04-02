---
layout: post
title: LLM기반 임베딩 모델, bge 리랭커 모델 'bge-reranker-v2-gemma'
---
리랭커 모델 소개

이 포스트에서는 'bge-m3'를 기반으로 한 '리랭커' 모델을 살펴보겠습니다. 기존의 '임베딩' 모델과는 달리 '리랭커' 모델은 질문과 문서를 입력으로 받아들이고 유사도를 출력합니다. 

다른 임베딩 모델과는 달리, 리랭커는 질문과 문서를 입력으로 사용하며, 임베딩 대신 유사도를 직접 출력합니다. 리랭커는 쿼리와 메시지를 입력으로 받으면 관련성 점수를 계산하며, 이 점수는 시그모이드 함수를 사용하여 [0,1] 범위의 부동 소수점 값으로 매핑될 수 있습니다.



또한, 다국어를 지원하기 위해 BAAI/bge-reranker-v2-m3와 BAAI/bge-reranker-v2-gemma 두 가지 버전이 존재합니다. gemma 버전은 LLM(Large Language Model) 기반의 리랭커 LLM-based reranker로 작동됩니다. 

예시로 "팬더란?" "팬더는 중국 남서부 산악지대에 서식하는 포유류의 일종이다."라는 질문에는 "팬더는 중국 남서부 산악지대에 서식하는 포유류의 일종이다." 라는 문서가 관련되어 있음을 알 수 있습니다.

**시그모이드 함수를 통한 유사도 계산**

리랭커 모델에서는 유사도를 [0,1] 범위의 부동 소수점 값으로 계산하기 위해 시그모이드 함수를 사용합니다.


**작동 코드 분석**

주어진 코드를 살펴보면, 쿼리 A와 패시지 B가 주어졌을 때, 패시지에 쿼리에 대한 답변이 포함되어 있는지 여부를 결정하여 '예' 또는 '아니오'의 예측을 제공하는 작동 코드입니다.

```
yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]

scores = model(**inputs, return_dict=True).logits[:, -1, yes_loc].view(-1, ).float()
```

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_inputs(pairs, tokenizer, prompt=None, max_length=1024):
    if prompt is None:
        prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    
	sep = "\n"
    prompt_inputs = tokenizer(prompt,
                              return_tensors=None,
                              add_special_tokens=False)['input_ids']
							  
    sep_inputs = tokenizer(sep,
                           return_tensors=None,
                           add_special_tokens=False)['input_ids']
						   
    inputs = []
    for query, passage in pairs:
        query_inputs = tokenizer(f'A: {query}',
                                 return_tensors=None,
                                 add_special_tokens=False,
                                 max_length=max_length * 3 // 4,
                                 truncation=True)
								 
        passage_inputs = tokenizer(f'B: {passage}',
                                   return_tensors=None,
                                   add_special_tokens=False,
                                   max_length=max_length,
                                   truncation=True)
								   
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id] + query_inputs['input_ids'], sep_inputs + passage_inputs['input_ids'],
            truncation='only_second',
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False
        )
		
        item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
        item['attention_mask'] = [1] * len(item['input_ids'])
        inputs.append(item)
		
    return tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors='pt',
    )

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-gemma')
model = AutoModelForCausalLM.from_pretrained('BAAI/bge-reranker-v2-gemma')
yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
model.eval()

pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = get_inputs(pairs, tokenizer)
    scores = model(**inputs, return_dict=True).logits[:, -1, yes_loc].view(-1, ).float()
    print(scores)
```
