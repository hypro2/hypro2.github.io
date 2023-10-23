---
layout: post
title: AutoGPTQ로 양자화
---

오늘은 AutoGPTQ로 한글 라마13b 모델을 양자화해볼려고 한다.  오늘도 koalpaca 데이터셋을 사용하려고 한다. 가볍게 데이터셋 프롬프트 형식만 맞춰서 만들어주고... 

```
ds = load_dataset("beomi/KoAlpaca-v1.1a", split="train")

ds_list = []
for i in range(len(ds)):
    ds_list.append(f"### User:\n{ds[i]['instruction']}\n\n### Assistant:\n{ds[i]['output']}</s>")

pickle.dump(ds_list, open("./ds.pkl", 'wb'), protocol=4)
```

AutoGPTQ를 통해서 양자화 하기 위한 코드르 준비한다. 뭐 별거 없다. 다 만들어주는 패키지ㄷㄷㄷ;;; 
당연히 4비트 128그룹으로 준비한다. 

```

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = 
quantized_model_dir =

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)

examples = []
for i in ds:
    examples.append(tokenizer(i))

quantize_config = BaseQuantizeConfig(
    bits=4, 
    group_size=128,  
    desc_act=False, 
)

model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
model.quantize(examples[:1000])
model.save_quantized(quantized_model_dir)
model.save_quantized(quantized_model_dir, use_safetensors=True)
```

![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/2ca6ad91-ee0f-4f52-ad15-c544e6d60f62)
돌아는 가고 있는데 로스가 심상치 않다 잘 안될거 같은 이 불길한 기운 ㅠㅠ 
나중에 돌와서 작성하겠음..
