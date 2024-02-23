---
layout: post
title: gemma colab에서 사용하기
---


새로운 LLM(언어 모델)인 Gemma는 Google이 Gemini를 기반으로 개발한 제품군입니다. Gemma에는 두 가지 크기의 모델이 있습니다: 2B와 7B. 이 두 가지 크기의 모델은 기본 모델과 명령 모델을 갖추고 있습니다. Gemma의 모든 변형은 양자화를 거치지 않고 다양한 종류의 소비자 하드웨어에서 실행될 수 있으며, 컨텍스트 길이는 8K 토큰입니다.
기본 모델은 프롬프트 형식이 없습니다. 이 모델은 다른 기본 모델과 마찬가지로 합리적인 연속으로 입력 순서를 계속하거나 Zero-Shot/Few-Shot 추론에 활용할 수 있습니다. 또한 사용자의 특정 사용 사례에 맞게 미세 조정할 수 있어 유연성을 제공합니다.



## 프롬프트 양식
```
<start_of_turn>user
knock knock<end_of_turn>
<start_of_turn>model
who is there<end_of_turn>
<start_of_turn>user
LaMDA<end_of_turn>
<start_of_turn>model
LaMDA who?<end_of_turn>
```

## 실제사용기

콜랩에서 bnb 4bit로 돌려봤습니다. 그래서 정확도가 좀 떨어집니다. 

```
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

model_name_or_id =  "google/gemma-7b-it"

model = AutoModelForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.float16, device_map="auto", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_id)


chat = [
    { "role": "user", "content": " 유희왕에서 가장 강력한 카드가 무엇인지 알고 있습니까?" },
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **input_ids,
    max_new_tokens=128,
    do_sample=True,
    top_p=0.95,
    temperature=0.7,
    repetition_penalty=1.1,
)

print(tokenizer.decode(outputs[0]))
```

