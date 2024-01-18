---
layout: post
title: 모델 리뷰 고퀄리티 애니 이미지 모델 animagine-xl-3.0
---
**ANIMAGINE XL 3.0 개요:**  

**모델 설명:**  

Animagine XL 3.0은 Animagine XL 2.0을 계승하는 고급 오픈 소스 애니메이션 텍스트-이미지 모델입니다.  
Cagliostro Research Lab에서 개발한 확산 기반의 텍스트-이미지 생성 모델입니다. Stable Diffusion XL을 사용하여 Cagliostro Research Lab에서 개발했습니다. 
  
  
**디퓨저 설치:**  

사용자는 애니매진 XL 3.0을 활용하기 위해 필수 라이브러리(디퓨저, 트랜스포머, 가속, 세이프텐서)를 설치해야 합니다.  
사용 지침:  
  
**훈련 및 하이퍼파라미터:**  

2x A100 GPU에서 21일 동안 3단계 훈련을 통해 훈련되었습니다.  
학습 중 다양한 하이퍼파라미터 및 구성이 자세히 설명되어 있습니다.  
  
**개선 사항 및 기능:**  

Stable Diffusion XL을 기반으로 제작되어 뛰어난 이미지 생성 성능을 발휘합니다.  
향상된 손 구조, 효율적인 태그 순서 지정, 애니메이션 개념에 대한 심층적인 이해 등이 개선되었습니다.  
미학보다는 모델 개념을 가르치는 쪽으로 초점이 옮겨졌습니다.  
  
**사용 지침:**  

태그 순서는 구조화된 프롬프트 템플릿에 따라 최적의 결과를 얻는 데 중요합니다.  
특수 태그, 품질 수정자, 등급 수정자 및 연도 수정자가 이미지 생성에 영향을 미칩니다.  
부정적인 프롬프트는 특정 결과에 대한 모델을 안내합니다.  
  
**제한사항 및 라이센스:**  

애니메이션 스타일 아트워크를 위해 디자인된 아트 스타일보다 컨셉을 우선시합니다.  
해부학적 구조 및 잠재적인 NSFW 콘텐츠 위험과 관련된 문제.

**사용후기:**

상당히 고퀄리티의 이미지를 생성해냄 ㄷㄷㄷ;;;
![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/9beeed0a-c12c-467d-b999-6fa9423bb469)


```
def dummy(images, **kwargs): return images, False 
pipe.safety_checker = dummy
```

[https://huggingface.co/cagliostrolab/animagine-xl-3.0](https://huggingface.co/cagliostrolab/animagine-xl-3.0)
**Google Colab**: [Open In Colab](https://colab.research.google.com/#fileId=https%3A//huggingface.co/Linaqruf/animagine-xl/blob/main/Animagine_XL_demo.ipynb)
