---
layout: post
tilte: 모델 리뷰 LLaVA1.5 Colab으 실행하기
---

오늘은 예전에 llava 1.5에 대해서 잠깐 언급했었던 적이 있는 모델입니다.LLaVA 1.5는 비전 기능을 갖춘 오픈 소스 모델로서 LLaVA는 대규모 언어 모델과 비전 어시스턴트를 결합하는 것을 목표로 하는 오픈 소스 프로젝트입니다.언어와 이미지를 모두 이해할 수 있는 엔드 투 엔드 멀티모달 모델을 만드는 것이 목표입니다.

**모델 아키텍처:**  
LLaVA는 사전 훈련된 CLIP 모델을 기반으로 하는 비전 인코더와 대규모 언어 모델(vicuna 13B)을 사용하여 GPT-4의 비전 기능을 모방합니다.

**성능 지표:**  
이 모델은 합성 다중 모드 명령 따르기 데이터세트에서 GPT-4에 비해 85% 점수를 달성하며, 130억 개의 매개변수를 고려할 때 인상적인 성능을 보여줍니다.

설명: LLaVA 프로젝트는 비전 기능을 언어 모델에 통합하여 텍스트 및 시각적 정보와 관련된 작업을 위한 다목적 도구를 만듭니다. 비전 인코더와 언어 모델을 결합한 아키텍처를 통해 양식 전반에 걸쳐 이해하고 생성할 수 있습니다. 다양한 시나리오에서 입증된 이 모델의 주목할만한 성능은 실제 응용 프로그램에서의 잠재적인 적용 가능성을 시사합니다. 오픈 소스 특성과 Apache 2.0 라이선스는 커뮤니티 참여와 탐색을 장려합니다.

 [LLaVA-1.5 이미지 텍스트 멀티모달hyeong9647.tistory.com](https://hyeong9647.tistory.com/entry/LLaVA-15-%EC%9D%B4%EB%AF%B8%EC%A7%80-%ED%85%8D%EC%8A%A4%ED%8A%B8-%EB%A9%80%ED%8B%B0%EB%AA%A8%EB%8B%AC)

**사용 및 데모:**  
LLaVA는 사용자가 이미지와 언어 프롬프트를 입력하여 모델의 이해도를 테스트할 수 있는 데모를 제공합니다.  
모델 가중치와 체크포인트를 다운로드할 수 있으므로 맞춤형 애플리케이션에 통합할 수 있습니다. 라바을 실행하기 위해서 콜랩을 통해서 사용해보겠습니다.

사용하기 위해서 해당 레포지토리을 참조 했습니다.

[https://github.com/camenduru/LLaVA-colab](https://github.com/camenduru/LLaVA-colab/blob/main/LLaVA_13b_4bit_vanilla_colab.ipynb)

```
# 현재 작업 디렉토리 변경 및 LLaVA 소스 코드 클론
%cd /content
!git clone -b v1.0 https://github.com/camenduru/LLaVA
%cd /content/LLaVA

# 필요한 라이브러리 설치
!pip install -q transformers==4.36.2
!pip install -q gradio .

# 필요한 패키지 및 모듈 임포트
from transformers import AutoTokenizer, BitsAndBytesConfig
from llava.model import LlavaLlamaForCausalLM
import torch

# 모델 및 토크나이저 초기화
model_path = "4bit/llava-v1.5-13b-3GB"
kwargs = {"device_map": "auto"}
kwargs['load_in_4bit'] = True
kwargs['quantization_config'] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# 모델에서 비전 타워 및 관련 구성 요소 초기화
vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda')
image_processor = vision_tower.image_processor

# 필요한 라이브러리 및 모듈 임포트
import os
import requests
from PIL import Image
from io import BytesIO
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer

# 이미지 캡션 생성 함수 정의
def caption_image(image_file, prompt):

    # 이미지 파일 불러오기
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')

    # Torch 초기화 비활성화
    disable_torch_init()

    # 대화 템플릿 및 역할 설정
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # 이미지 전처리 및 텐서로 변환
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    # 입력 생성 및 대화에 추가
    inp = f"{roles[0]}: {prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)

    # 토큰화 및 생성을 위한 입력 데이터 준비
    raw_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # 생성 중단 기준 설정
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # Torch 추론 모드에서 생성
    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=image_tensor, do_sample=True, temperature=0.2,
                                    max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])

    # 출력 디코딩 및 정리
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit('</s>', 1)[0]
    return image, o
```



사용했을 때 GPU는 bnb의 4bit 기준으로 10기가 정도 사용하는 것을 확인 할 수 있었습니다.
