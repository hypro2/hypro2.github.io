---
layout: post
title: LLaVA-1.5 이미지 텍스트 멀티모달
---

LLaVA-1.5는 이미지 분석이 가능한 멀티모달의 오픈소스로서 11개 벤치마크에서 소타를 달성했다.

[https://llava-vl.github.io/](https://llava-vl.github.io/)

<img width="458" alt="image" src="https://github.com/hypro2/hypro2.github.io/assets/84513149/e230ea96-31de-457a-acf5-b7e96e850c5d">

원문에서 중요한 아키텍쳐 부분 발췌

\`\`\`

주요 목표는 사전 훈련된 언어 모델(Language Model)과 시각 모델(Visual Model)의 능력을 효과적으로 활용하는 것입니다. 네트워크 아키텍처는 Figure 1에서 보여집니다. 우리는 LLaMA를 우리의 LLM fφ(·)로 선택하였으며, 이는 여러 오픈소스 언어 모델 튜닝 연구에서 그 효과가 입증되었습니다. \[43, 45, 34\].  
  
입력 이미지 Xv에 대해, 우리는 사전 훈련된 CLIP 시각 인코더 ViT-L/14 \[36\]를 고려합니다. 이는 시각 특성인 Zv = g(Xv)를 제공합니다. 우리는 마지막 트랜스포머 레이어 이전과 이후의 그리드 특성을 실험에서 고려합니다.  
이미지 특성을 단어 임베딩 공간으로 연결하기 위해 간단한 선형 레이어를 고려합니다. 구체적으로, 우리는 trainable projection matrix W를 적용하여 Zv를 언어 임베딩 토큰인 Hq로 변환합니다. 이는 언어 모델의 언어 임베딩 공간과 동일한 차원을 갖습니다.  
Hv = W · Zv, 여기서 Zv = g(Xv) (1)  
이렇게 하면 시각 토큰 Hv의 시퀀스가 생성됩니다. 우리의 간단한 프로젝션 방식은 가벼우며 비용 효율적이며, 데이터 중심 실험을 빠르게 반복할 수 있도록 합니다. 시각 및 언어 표현을 연결하기 위한 더 정교한(하지만 비용이 많이 드는) 방법도 고려할 수 있으며, Flamingo의 게이트드 크로스-어텐션 \[2\] 및 BLIP-2의 Q-former \[25\]와 같은 방법이나 객체 수준 특성을 제공하는 SAM \[21\]과 같은 다른 시각 인코더도 고려할 수 있습니다. 더 효과적이고 정교한 아키텍처 디자인을 탐구하는 것은 LLaVA의 미래 작업으로 남겨두었습니다.

\`\`\`

모델 아키텍처  
  
LLaVA: LLaVA는 언어와 비전을 통합하기 위해 간단한 선형 프로젝션 레이어를 사용하는 경량 모델로서, 이미지의 시각 특성을 언어 임베딩으로 변환하여 언어 모델과 통합합니다. 이것은 상대적으로 간단한 방법입니다. Flamingo 및 BLIP-2: Flamingo와 BLIP-2는 보다 복잡한 모델 아키텍처를 사용합니다. Flamingo는 게이트드 크로스-어텐션과 같은 특별한 메커니즘을 사용하여 언어와 비전 간의 상호 작용을 조절하며, BLIP-2는 "Q-former"와 같은 특수한 언어 처리 모듈을 도입하여 다양한 다모달 작업을 수행합니다. 하지만 라바는 간단한 선형프로젝션 레이어를 통해서 경량화하고 성능을 높혔다고 합니다.

<img width="463" alt="image" src="https://github.com/hypro2/hypro2.github.io/assets/84513149/4f6110e4-e8e7-40a3-9ed0-87595d53249c">


```
!git clone https://github.com/haotian-liu/LLaVA.git
%cd LLaVA
!pip install -e .
```

```
!python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-13b \
    --image-file "./image.jpg" \
    --load-8bit
```

[https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/model/language\_model/llava\_llama.py#L41](https://github.com/haotian-liu/LLaVA/blob/e854a2bf85118c504f6f16bf5c3c7c92f8fa8c6b/llava/model/language_model/llava_llama.py#L41)
