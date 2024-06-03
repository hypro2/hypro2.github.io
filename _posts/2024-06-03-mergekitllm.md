---
layout: post
title: LLM 병합 Mergekit을 사용해보자 간단
---

툴킷 소개 및 목적:

MergeKit은 매개변수를 결합하고 전이 학습의 발전을 활용하여 대규모 언어 모델(LLM)을 병합하도록 설계된 툴킷입니다.  
이는 광범위한 재교육 없이 오픈 소스 LLM의 기능을 향상하여 치명적인 할루시네이션과 같은 문제를 해결하는 것을 목표로 합니다.

병합 기술:  
Linear Mode Connectivity (LMC):모델 가중치의 선형 평균을 활용합니다.  
Task Arithmetic: 작업 벡터에 대한 산술 연산을 포함합니다.  
Permutation Symmetry: 다양한 변환을 사용하여 손실 환경의 공통 영역에 대한 가중치를 조정합니다.  
고급 기술: Fisher information 매트릭스, RegMean 및 OTFusion(Optimal Transport Fusion)과 같은 방법을 포함합니다.



실용적인 응용:  
다양하고 견고한 모델: MergeKit을 사용하면 처음부터 시작하지 않고도 여러 작업에서 탁월하거나 새로운 도메인에 적응하는 모델을 생성할 수 있습니다.  
리소스 최적화: 복잡한 실제 문제를 해결하기 위해 기존 사전 학습된 모델의 유용성을 극대화합니다.

라이브러리 디자인 및 확장성:  
확장 가능한 아키텍처: MergeKit은 메모리가 제한된 CPU와 가속 GPU 모두에서 실행을 지원합니다.  
모듈식 구조: 주요 모듈에는 아키텍처 호환성, 병합 계획 수립 및 운영 그래프 실행이 포함됩니다.  
상호 운용성: Hugging Face Transformers와 원활하게 통합되어 모델 결합이 쉬워집니다.

커뮤니티 참여 및 지원:  
지속적인 업데이트: 새로운 병합 기술을 통합하기 위한 정기적인 업데이트 및 유지 관리입니다.  
사용자 협업: 오픈 소스 커뮤니티의 피드백, 토론 및 기여를 장려하여 툴킷의 기능을 향상시킵니다.

설치
```
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit

pip install -e .  # install the package and make scripts available
```

병합에 필요한 config 설정
```
import yaml

MODEL_NAME = "llama-3-base-instruct-slerp"
yaml_config = """
 slices:
   - sources:
       - model: ./Meta-Llama-3-8B-Instruct
         layer_range: [0, 32]
       - model: ./Meta-Llama-3-8B-Base
         layer_range: [0, 32]
 merge_method: slerp
 base_model: ./Meta-Llama-3-8B-Instruct
 parameters:
   t:
     - filter: self_attn
       value: [0, 0.5, 0.3, 0.7, 1]
     - filter: mlp
       value: [1, 0.5, 0.7, 0.3, 0]
     - value: 0.9
 dtype: bfloat16

"""

print("Writing YAML config to 'config.yaml'...")

try:
   with open('config.yaml', 'w', encoding="utf-8") as f:
       f.write(yaml_config)
   print("File 'config.yaml' written successfully.")
except Exception as e:
   print(f"Error writing file: {e}")

CONFIG_YML="./config.yaml"
```

실행
```
# actually do merge
import torch
import yaml

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

OUTPUT_PATH = "./merged"  # folder to store the result in
LORA_MERGE_CACHE = "/tmp"  # change if you want to keep these for some reason
CONFIG_YML = "./examples/gradient-slerp.yml"  # merge configuration file
COPY_TOKENIZER = True  # you want a tokenizer? yeah, that's what i thought
LAZY_UNPICKLE = False  # experimental low-memory model loader
LOW_CPU_MEMORY = False  # enable if you somehow have more VRAM than RAM+swap

with open(CONFIG_YML, "r", encoding="utf-8") as fp:
    merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

run_merge(
    merge_config,
    out_path=OUTPUT_PATH,
    options=MergeOptions(
        lora_merge_cache=LORA_MERGE_CACHE,
        cuda=torch.cuda.is_available(),
        copy_tokenizer=COPY_TOKENIZER,
        lazy_unpickle=LAZY_UNPICKLE,
        low_cpu_memory=LOW_CPU_MEMORY,
    ),
)
print("Done!")
```

[https://github.com/arcee-ai/mergekit](https://github.com/arcee-ai/mergekit)
