---
layout: post
title: PaliGemma 구글의 오픈소스 멀티모달 리뷰
---

구글의 오픈소스 멀티모달 Paligemma입니다. 

이 모델 또한 llava나 다른 모델 처럼 visual model과 llm을 선형 프로젝션해서 구현한 모델입니다. Joint Fusion의 멀티모달 이미지 텍스트	모델입니다. 이러한 joint fusion은 학습 과정에서 특정 레이어에서 다른 모달리티 데이터를 융합을 시도합니다.

주로 임베딩층에서 clip모델과 llm 모델을 사용합니다. 이 경우에도 SiglipVisionModel과 GemmaModel을 사용해서 중간에 multi_modal_projector을 통해서 이미지 정보와 함께 llm에 전달해서 사용되는 것으로 보입니다. 

이번에도 콜랩에서 작업을 진행했습니다. 궁금하신분은 https://github.com/hypro2/LLM-Multimodal-Colab/blob/main/multimodal/paligemma_3b_mix_224_colab.ipynb 여기서 확인 바랍니다. 




![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/fe70e6f6-7961-4933-bd53-9b85f8e6fc2c)


```
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

model_id = "google/paligemma-3b-mix-224"

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)
```
```
PaliGemmaForConditionalGeneration(
(vision_tower): SiglipVisionModel(
(vision_model): SiglipVisionTransformer(
(embeddings): SiglipVisionEmbeddings()
)
(multi_modal_projector): PaliGemmaMultiModalProjector(
(linear): Linear(in_features=1152, out_features=2048, bias=True)
)
(language_model): GemmaForCausalLM(
(model): GemmaModel()
(lm_head): Linear(in_features=2048, out_features=257216, bias=False)
)
)
```
```
prompt = "What is this?"
raw_image = Image.open("/content/download.jpg")
inputs = processor(prompt, raw_image, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=20)

print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])
```

load_in_4bit=True를 통해서 열면 아주 작은 GPU로만으로도 작동이 가능합니다. 
2.4GB만큼의 메모리만큼만 사용하는 것을 확인 할 수 있습니다. 


![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/9f3ca291-0339-4b71-86e1-6902a7e42406)
