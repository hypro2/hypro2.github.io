---
layout: post
title: exllamav2로 exl2형식으로 양자화하기
---

데이터셋은 준비했고, 모델도 형식에 맞게 변환했고 리눅스에서 명령어로 exl2를 변환해주는 코드를 작성해주면된다.

보면서 따라 했는데도 힘들구먼 결과가 잘나올려나

<img width="769" alt="image" src="https://github.com/hypro2/hypro2.github.io/assets/84513149/a2af334a-563a-4ee6-9f50-f4d388a1b769">

[https://github.com/turboderp/exllamav2/blob/master/doc/convert.md](https://github.com/turboderp/exllamav2/blob/master/doc/convert.md)

```
#데이터셋 만들기 parquet형식을 만들어야된다.
# 한글모델 양자화를 위해 코알파카셋을 사용한다.
from datasets import load_dataset
ds = load_dataset("beomi/KoAlpaca-v1.1a", split="train")

ds_list = []
for i in range(len(ds)):
    ds_list.append(f"### User:\n{ds[i]['instruction']}\n\n### Assistant:\n{ds[i]['output']}")
    
df = pd.DataFrame({'instruction':ds_list})
df.to_parquet("./ds.parquet")


# safetensors 형식으로 저장해줘야된다.
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             )

model.save_pretrained(f"{model_name_or_path}/quant_model/",max_shard_size="40GB", safe_serialization=True)
```

```
# 리눅스 명령어
python convert.py -i llama2-ko-en-platypus-13B -o llama2-ko-en-13B-temp -cf llama2-ko-en-4.0bpw-h8-exl2 -c cal_dataset.parquet -l 4096 -b 4 -hb 8 -ss 4096
```

