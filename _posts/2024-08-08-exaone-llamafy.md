---
layout: post
title: EXAONE 3.0 7.8B 모델의 llamafied과 파인튜닝
---

EXAONE 3.0 소개: EXAONE 3.0은 LG AI Research에서 개발한 명령 조정 언어 모델로, LLM(대형 언어 모델) 시리즈 중 최초의 개방형 모델로 유명합니다. 78억 개의 매개변수 버전이 연구 및 혁신을 지원하기 위해 공개적으로 출시되었습니다.

성능 및 역량: 이 모델은 경쟁력 있는 성능을 보여줍니다. 특히 한국어 작업에서 탁월한 성능을 발휘하는 동시에 일반적인 작업과 복잡한 추론에서도 우수한 성능을 발휘합니다. 언어 기술과 도메인 전문성을 향상하기 위해 대규모 데이터 세트(8조 토큰)에 대한 교육을 받았습니다.

모델 아키텍처 및 기능: EXAONE 3.0은 RoPE(Rotary Position Embeddings) 및 GQA(Grouped Query Attention)와 같은 고급 기술을 사용합니다. 32개 레이어, 모델 차원 4,096개로 이루어져 있으며 SwiGLU 활성화 기능을 사용하여 복잡한 언어 작업을 처리하는 데 최적화되어 있습니다.

공개 가용성 및 사용: EXAONE 3.0은 주로 LG의 상업 파트너를 위한 것이지만 7.8B 모델은 비상업적 연구 목적으로 사용할 수 있습니다. 이 공개 릴리스는 더 광범위한 AI 연구 커뮤니티에 기여하는 것을 목표로 합니다.

윤리적 고려 사항 및 안전 조치: 이 모델은 오용을 방지하기 위한 조치를 통해 엄격한 윤리 및 보안 테스트를 거쳤습니다. LG AI Research는 레드팀 구성 및 기타 안전 프로토콜을 통해 편견, 차별, 유해 콘텐츠에 대한 대응을 포함하여 책임 있는 사용을 강조합니다.



```

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
 
model = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")

```


## llamafied

[maywell/EXAONE-3.0-7.8B-Instruct-Llamafied](https://huggingface.co/maywell/EXAONE-3.0-7.8B-Instruct-Llamafied)

해당 방법을 통해서 ExaoneForCausalLM구조에서 LlamaForCausalLM 구조로 변경 할 수 있습니다. 이렇게 변경된 구조를 통해 vllm과 파인튜닝에 대해서 기존의 라마로 진행했던 라이브러리나 프로젝트를 같이 사용할 수 있습니다. 

코드는 위에 링크로 들어가서 보시고 EXAONE 모델을 LLaMA 모델 형식으로 변환하는 작업으로 라마 아키텍쳐에 이 과정을 llamafy라고 부릅니다.

EXAONE 모델 로드: EXAONE 모델과 토크나이저를 불러옵니다.

LLaMA 설정 생성: EXAONE 모델의 설정을 바탕으로 LLaMA 모델 설정을 만듭니다.

가중치 복사: EXAONE 모델의 가중치를 LLaMA 모델로 복사합니다.

LLaMA 모델 저장: 변환된 LLaMA 모델과 토크나이저를 지정된 경로에 저장합니다.

이 과정을 통해 EXAONE 모델을 LLaMA 모델로 변환하고 저장할 수 있습니다.

## vllm 구동
llamafied된 모델은 라마와 아키텍쳐를 공유하기 때문에 vllm에서도 바로 사용할 수 있다. 

```
WARNING 08-07 10:22:11 config.py:1425] Casting torch.float16 to torch.bfloat16.
INFO 08-07 10:22:11 llm_engine.py:176] Initializing an LLM engine (v0.5.3.post1) with config: model='./data/EXAONE-3.0-7.8B-Instruct-llamafied', speculative_config=None, tokenizer='./data/EXAONE-3.0-7.8B-Instruct-llamafied', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None), seed=0, served_model_name=./data/EXAONE-3.0-7.8B-Instruct-llamafied, use_v2_block_manager=False, enable_prefix_caching=False)
INFO 08-07 10:22:12 model_runner.py:680] Starting to load model ./data/EXAONE-3.0-7.8B-Instruct-llamafied...
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
INFO 08-07 10:22:17 model_runner
```

## 파인튜닝 

라마팩토리로 진행, 라마 팩토리에서 파인 튜닝 성공.

https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/template.py

커스텀 코드 추가
```
_register_template(
    name="exaone",
    format_user=StringFormatter(
        slots=[
            (
                "[|user|]\n\n{{con_tent}}[|endofturn|]"
                "[|assistant|]\n\n"
            )
        ]
    ),
    format_system=StringFormatter(slots=["[|system|]\n\n{{con_tent}}[|endofturn|]"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["[|endofturn|]"],
    replace_eos=True,
)
```


