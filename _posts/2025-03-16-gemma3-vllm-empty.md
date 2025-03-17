---
layout: post
title: gemma3 vllm에서 dtype bfloat16과 float16 빈칸 문제
---

현재 젬마3가 나와서 구동을 돌려보는데 제대로 안되는 경우가 발생한다. vllm에서 dtype을 float16으로 돌릴때 문제가 나온다.

기본적으로 젬마3가 bfloat16으로 학습이 진행됬는데, 콜랩 무료환경에서는 bfloat16이 T4 GPU의 Capability가 7.5이기 때문에 지원을 하지 않는다.

vllm에서 최신 버전을 깃허브로 precompied된 버전을 다운받고 transformers를 @v4.49.0-gemma-3을 설치해도 제대로 작동하지 않을 것이다.

Bfloat16 is only supported on GPUs with compute capability of at least 8.0. Your Tesla T4 GPU has compute capability 7.5. You can use float16 instead by explicitly setting the `dtype` flag in CLI


```
!git clone https://github.com/vllm-project/vllm.git
!cd /content/vllm && VLLM_USE_PRECOMPILED=1 pip install --editable .

!pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
```

float16으로 돌리게되면 빈칸만 나오게되는 문제가 발생한다. 젬마는 여러므로 vllm하고 사이가 않좋은거 같다. 

vllm 버그 리포트에도 https://github.com/vllm-project/vllm/issues/6177,https://github.com/vllm-project/vllm/issues/14817 계속 올라오고 있다.

float16은 쓰면 안되고 bfloat16 혹은 float32를 사용해야되는다. float32를 쓰는 사람은 없겠지... 다행히 콜랩 L4 GPU에서는 무사히 돌아간다. 
