---
layout: post
title: Paper 리뷰 Chat Vector 학습된 가중치 매개변수를 더하고 빼는 것으로 사전 학습된 모델에 대화 능력을 부여함
---
LLM 관련 논문중에 재밌는 것을 발견 했습니다. Llama-3-Open-Ko-8B-Instruct-preview의 README를 보던 중 Chat Vector라는 것을 알게 되었습니다. Chat Vector 학습된 가중치 매개변수를 더하고 빼는 것으로 사전 학습된 모델에 대화 능력을 부여해준다는게 흥미로웠습니다.

"With applying the idea from [Chat Vector paper](https://arxiv.org/abs/2310.04799), I released Instruction model named [Llama-3-Open-Ko-8B-Instruct-preview](https://huggingface.co/beomi/Llama-3-Open-Ko-8B-Instruct-preview)." ([https://huggingface.co/beomi/Llama-3-Open-Ko-8B-Instruct-preview](https://huggingface.co/beomi/Llama-3-Open-Ko-8B-Instruct-preview)) 발췌



**Chat Vector: A Simple Approach to Equip LLMs with Instruction Following and Model Alignment in New Languages**

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FlC9p3%2FbtsHalVajVn%2FtvB8tLpM9H2XLEqloJu7sk%2Fimg.png)

그림 중 PLM은 Pretrained Language Model(사전 학습 모델)을, CP는 Continual Pre-training(계속된 사전 학습)을 의미합니다. LLM의 가중치에서도 word2vec 처럼 " 한국 - 서울 + 도쿄 = 일본"과 같은 것이 가능하다는 내용입니다.

딥러닝 모델에서 가중치에 대한 덧셈과 뺄셈 산술이 가능하다는게 신선하게 느껴지는데, 이렇게 한다면 대규모 계산 리소스가 필요하지 않아지고, Raw Data 학습으로만으로 Chat의 능력을 기대 할 수 있을까요? 새로운 모델을 학습하지 않고도 대화 능력을 전이시키는 것으로 대상 언어나 도메인에 대한 데이터 부족 상황에서도 효과적으로 사용될지 기대됩니다.

첫 번째 임베디드 계층과 마지막 lm\_head의 차원은 vocab size의 차이로 인해 다르지만, 그 외의 중간 레이어들은 모두 state\_dict의 키와 size가 동일 합니다. layernorm은 가중치가 없기때문에 굳이 계산은 하지 않습니다.

```
from transformers import AutoModelForCausalLM
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
inst_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

cp_model = AutoModelForCausalLM.from_pretrained(
    "maywell/Mistral-ko-7B-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

for k, v in base_model.state_dict().items():
    print(k, v.shape)
for k, v in cp_model.state_dict().items():
    print(k, v.shape)

skip_layers = ["model.embed_tokens.weight", "lm_head.weight"]

for k, v in cp_model.state_dict().items():
    if (k in skip_layers) or ("layernorm" in k):
        continue

    chat_vector = inst_model.state_dict()[k] - base_model.state_dict()[k]
    new_value = v + chat_vector.to(v.device)
    v.copy_(new_value)
```

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F73sAY%2FbtsG8ANjsay%2FKJebbS4LTnxQeYZ1I8YryK%2Fimg.png)

더 진행하고 싶은데!!! 디스크가 모잘라 ㅠㅠㅠ !!!!

보다 자세한 정보는 아래 참고 자료의 링크를 참조해주세요.

([https://arxiv.org/abs/2310.04799](https://arxiv.org/abs/2310.04799))

([https://qiita.com/jovyan/items/ee6affa5ee5bdaada6b4](https://qiita.com/jovyan/items/ee6affa5ee5bdaada6b4))
