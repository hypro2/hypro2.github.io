---
layout: post
title: Mixtral 
---

Mixtral 7bx8는 Mistral 7B와 유사한 구조를 가지고 있지만, Mixture of Experts (MoE)라는 기술을 사용하여 8개의 "전문가" 모델을 하나로 통합한 모델입니다. 트랜스포머 모델에서 MoE가 작동하는 방식은 몇 개의 Feed-Forward 레이어를 희소한 MoE 레이어로 교체하는 것입니다. MoE 레이어에는 어떤 전문가가 어떤 토큰을 가장 효율적으로 처리할지 선택하는 라우터 네트워크가 포함되어 있습니다. Mixtral의 경우, 각 시간 단계마다 두 개의 전문가가 선택되어 모델이 4배 많은 유효 매개변수를 포함하면서도 12B 매개변수 밀도 모델의 속도로 디코딩할 수 있게 합니다.

Mixtral의 주요 특징은 다음과 같습니다:
- 베이스 및 Instruct 버전 출시
- 32,000 토큰의 컨텍스트 길이 지원
- Llama 2 70B를 능가하며 대부분의 벤치마크에서 GPT3.5를 따라잡거나 뛰어넘음
- 영어, 프랑스어, 독일어, 스페인어, 이탈리아어 지원
- 코딩 능력이 뛰어나며 HumanEval에서 40.2%의 성능을 보임
- Apache 2.0 라이선스로 상업적 이용이 허용됨

Mixtral 모델의 성능은 얼마나 우수한가요? 다른 오픈 모델과 비교하여 LLM 리더보드에서 베이스 모델과 성능을 개요로 살펴볼 수 있습니다.

주로 많이 사용되는 리더보드로는 2가지를 많이 참고 합니다.
챗봇 아레나를 통한 정성평가와 벤치마크를 이용한 평가 2가지를 많이 검토 합니다. 
이경우 전부 영어로 평가가 되기 때문에 다른 언어에 대해서는 평가를 하기 어렵습니다. 
그럼에도 불구하고 상당히 지피티 3.5보다 나은 성적을 얻은 것을 볼 수 있습니다. 

https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

사용되는 프롬프트 구조로는 mistral과 비슷하게 
```
<s> [INST] User Instruction 1 [/INST] Model answer 1</s> [INST] User instruction 2[/INST]
```
구조로 사용됩니다. 

구동 동작 또한 최신 트랜스포머를 사용하면 사용가능합니다. 
```
from transformers import AutoTokenizer
import transformers
import torch

model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
)

messages = [{"role": "user", "content": "Explain what a Mixture of Experts is in less than 100 words."}]
prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])
```

A100으로도 float 타입의 모델을 바로 불러 올 수 없는데요. 
하지만 Vram에 대한 메모리 제한 있다면 exllama의 양자화 방식이나 gptq 방식의 모델을 검토할 수 있습니다. 

float16	> 90 GB
8-bit	> 45 GB
4-bit	> 23 GB

optimum과 auto-gptq 를 통한 GPTQ방식과 Exllamav2를 이용한 방식이 추천됩니다. 
https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GPTQ
https://huggingface.co/turboderp/Mixtral-8x7B-instruct-exl2


