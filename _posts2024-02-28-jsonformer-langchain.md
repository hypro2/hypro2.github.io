---
layout: post
title: langchain과 Jsonformer를 이용해서 function calling 구현하기
---

langchain과 Jsonformer를 이용해서 function calling 구현하기
https://github.com/1rgs/jsonformer

Jsonformers를 사용하여 function calling을 흉내내는 Language Model (LLM)을 만드는 과정을 자세히 설명하겠습니다.

Jsonformers는 구조화된 데이터에서 많은 토큰이 고정되어 있어 예측 가능한 경우에 적합한 방법입니다. 이 방법은 Hugging Face의 모델을 활용하고, 생성 과정에서 고정된 토큰만을 채우며, 언어 모델에게는 내용 토큰의 생성을 위임합니다.



먼저, parser와 pydantic을 이용하여 클래스를 만들고, `schema()`을 사용하여 쉽게 JSON 스키마를 생성할 수 있습니다. 이 JSON 스키마는 생성에 필요한 인자와 스키마를 제공하고, Jsonformers에 전달됩니다.

Jsonformers를 통해 제공된 JSON 스키마를 기반으로 랭체인과 llama2에서 생성 과정이 이루어집니다. 이 과정에서 딕셔너리 { }를 선언하고, properties의 title과 value의 타입을 확인합니다. 그 후, 입력으로부터 하나씩의 properties를 가져와 생성을 반복하고, 마지막에는 딕셔너리를 출력합니다. 이 과정은 동일한 입력과 그 다음 properties를 반복하여 생성하고, 생성되지 않아야 할 토큰의 로짓 값을 -inf로 만들어주는 LogitsWarper와 StoppingCriteria를 사용하여 조절됩니다.

Jsonformers는 현재 JSON 스키마의 일부만을 지원하며, 지원되는 스키마 유형은 다음과 같습니다: number, boolean, string, array, object.

이를 통해 랭체인으로 허깅페이스 모델을 감싸는 것으로 JsonformersLLM을 정의할 수 있습니다.


```
import json
from functools import partial
from typing import List, Mapping, Optional, Any, Dict
 
import jsonformer
import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer
 
 
class JSONformersLLM(LLM):
    model_folder_path: str = Field(None, alias='model_folder_path')
    model_name: str = Field(None, alias='model_name')
    backend: Optional[str] = 'llama'
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.1
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 200
    repetition_penalty: Optional[float] = 1.15
 
    ## 추가 ##
    model: Any = None
    tokenizer: Any = None
    #########
 
    def __init__(self, model_folder_path, callbacks=None, **kwargs):
        super(JSONformersLLM, self).__init__()
        self.model_folder_path: str = model_folder_path
        self.callbacks = callbacks
 
        ## 추가 ##
        self.model = AutoModelForCausalLM.from_pretrained(self.model_folder_path,
                                                          torch_dtype=torch.float16,
                                                          trust_remote_code=True,
                                                          do_sample=True,
                                                          device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_folder_path, use_fast=False)
        #########
 
    @property
    def _get_model_default_parameters(self):
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
        }
 
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            'model_name': self.model_name,
            'model_path': self.model_folder_path,
            'model_parameters': self._get_model_default_parameters
        }
 
    @property
    def _llm_type(self) -> str:
        return 'llama'
 
    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              json_schema: Dict = None,
              **kwargs) -> str:
 
        params = {
            **self._get_model_default_parameters,
            **kwargs
        }
 
        ## 추가 ##
        model = jsonformer.Jsonformer(model=self.model,
                              tokenizer=self.tokenizer,
                              json_schema=json_schema,
                              prompt=prompt,
                              max_array_length=params['max_tokens'],
                              max_number_tokens=params['max_tokens'],
                              max_string_token_length=params['max_tokens'],
                              temperature=params['temperature']
                              )
 
        text = model()
 
        return json.dumps(text)
```

```
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")
     
json_schema = Joke.schema()
print(json_schema)

"""
{'title': 'Joke', 'type': 'object', 'properties': {'setup': {'title': 'Setup', 'description': 'question to set up a joke', 'type': 'string'}, 'punchline': {'title': 'Punchline', 'description': 'answer to resolve the joke', 'type': 'string'}}, 'required': ['setup', 'punchline']}
"""
 
text = "Tell me a joke."
prompt = llama_prompt.format(text=text)
output = llm(prompt, json_schema=json_schema)
print(output)

"""
{"setup": "Why don't scientists trust atoms?", "punchline": "Because they make up everything!"}
"""
 
print(parser.parse(output))
"""
(setup="Why don't scientists trust atoms?" punchline='Because they make up everything!')
"""
```
