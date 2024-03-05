---
layout: post
title: 모델 리뷰 야놀자 이브이 콜랩구동 후기
---

야놀자에서 이브이라는 한글 모델을 출시 했습니다. 해당 모델은 라마2를 기반으로 만들어 진 모델이며 한글이 잘되도록 튜닝을 한 모델로 보입니다. 2.8B 모델과 10.8B 모델 두가지고 존재합니다. Solar를 기반으로 튜닝 했는지 모르겠지만 파라미터가 10.8B를 사용하고 있습니다. 


**2.8B**
처음에는 2.8b를 먼저 사용해봤습니다. colab에서 실행해서 파라미터도 적고 바로 돌릴 수 있을 거 같았는데 생각보다 아슬 아슬하네요.
간단히 테스트 해봤으나 실제 사용하기는 좀 어렵다고 판단이 들었습니다. 바로 10.8B를 쓰는 것으로 넘어가겠습니다. 
```
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("yanolja/EEVE-Korean-Instruct-2.8B-v1.0", trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-Instruct-2.8B-v1.0", trust_remote_code=True)

prompt_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {prompt}
Assistant:
"""
```

![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/ffa982e8-9860-48c0-8c96-f614ac43fc5d)


**10.8B**

10.8B의 eeve는 생각보다 메모리를 많이 사용해서, bnb 4bit로 로드 했습니다. 원래 가중치로 돌리고 싶었으나 콜랩을 프로로 바꿔야될 거 같습니다.
일단 bnb로 로드 해도 상당히 잘 대답하는 모습을 볼 수 있습니다. 예제로 유희왕 카드의 텍스트를 보고 나름 복잡한 규정에 대해서 물어봤습니다.
결과는 맞게 나왔지만 약간 아쉬운 모습을 볼 수 있었습니다. 구할 수 있는 데이터 중에 유희왕 카드게임의 룰에 대한 자료가 다른 데이터보다 소규모기때문에 거의 없거나 적게 들어갔을거같은데 생각보다 대답 잘하네요.

```
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0", trust_remote_code=True, device_map="auto", quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0", trust_remote_code=True)

prompt_template = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: {prompt}
Assistant:
"""
```

![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/3b996fcc-2aa0-4f80-a22f-18285ec67fa8)


```
text = """
할루시네이션 테스트

유희왕 카드 퀴즈


하루 우라라  
이 카드명의 효과는 1턴에 1번밖에 사용할 수 없다.  
①: 이하의 어느 효과를 포함하는 마법 / 함정 / 몬스터의 효과가 발동했을 때, 이 카드를 패에서 버리고 발동할 수 있다. 그 효과를 무효로 한다.  
●덱에서 카드를 패에 넣는 효과  
●덱에서 몬스터를 특수 소환하는 효과  
●덱에서 카드를 묘지로 보내는 효과


이 카드는 자신 턴 / 상대 턴 상관없이 패에서 효과를 발동해서 상대를 견제할 수 있는 패트랩입니다.
덱에서 카드를 패에 넣거나, 묘지로 보내거나, 몬스터를 특수 소환하는 효과를 무효화할 수 있습니다.
이 카드의 텍스트에서 가장 주의해야 하는 부분은 이하의 어느 효과를 포함하는의 서술입니다.
텍스트 어딘가에 저 세 가지 효과 가운데 하나라도 포함이 되었다면 무효화가 가능하다는 뜻입니다.

Q1. 상대가 개기일식의 서를 발동했습니다. 개기일식의 서의 효과에 체인하여 하루 우라라를 발동하여 상대의 효과를 무효화 수 있을까요?


개기일식의 서  
속공 마법  
①: 필드의 앞면 표시 몬스터를 전부 뒷면 수비 표시로 한다. 이 턴의 엔드 페이즈에, 상대 필드의 뒷면 수비 표시 몬스터를 전부 앞면 수비 표시로 하고, 그 후, 이 효과로 앞면 수비 표시로 한 몬스터의 수만큼 상대는 덱에서 드로우한다.


Q1의 정답:"""

model\_inputs = tokenizer(prompt\_template.format(prompt=text), return\_tensors='pt').to("cuda")

outputs = model.generate(\*\*model\_inputs, streamer=streamer, max\_new\_tokens=256)  
output\_text = tokenizer.batch\_decode(outputs, skip\_special\_tokens=True)\[0\]

print(output\_text)
```

Assistant:  
Q1. 상대가 개기일식의 서를 발동했습니다. 개기일식의 서의 효과에 체인하여 하루 우라라를 발동하여 상대의 효과를 무효화할 수 있을까요?

답변: 네, 가능합니다. 개기일식의 서의 효과는 필드의 앞면 표시 몬스터를 전부 뒷면 수비 표시로 하는 것으로, 이는 덱에서 카드를 패에 넣는 효과에 해당합니다. 따라서 하루 우라라를 개기일식의 서의 효과에 체인하여 발동하면 상대의 효과를 무효화할 수 있습니다.

하루 우라라는 자신 턴이나 상대 턴에 상관없이 패에서 효과를 발동하여 상대를 견제할 수 있는 패트랩입니다. 덱에서 카드를 패에 넣거나, 묘지로 보내거나, 몬스터를 특수 소환하는 효과를 무효화할 수 있습니다. 이 카드의 텍스트에서 가장 주의해야 할 부분은 이하의 어느 효과를 포함하는 서술입니다. 텍스트 어딘가에 저 세 가지 효과 가운데 하나라도 포함이 되었다면 무효화가 가능합니다.

개기일식의 서의 효과는 필드의 앞면 표시 몬스터를 전부 뒷면...

4bit bnb로 구동시키는거라 너무 느려서 전부다 확인하지 못했지만 상당히 잘 이해하는 모습을 볼 수 있었습니다.
