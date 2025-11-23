---
layout: post
title: 오픈 LLM에서 중국어, 한자 안뜨게 하기 vLLM Custom Logits Processors
---


대규모 언어 모델을 쓰다 보면 특정 단어나 토큰이 나오지 않도록 막거나, 반대로 어떤 표현은 꼭 포함시키고 싶을 때가 있습니다. vLLM의 '커스텀 로짓 프로세서’입니다. 모델이 다음 토큰을 선택하기 직전 단계에 개입해서 로짓 값을 바꿀 수 있게 해주는 장치로, 모델의 동작을 원하는 방향으로 유도하는 데 매우 유용합니다.

트랜스포머의 로짓프로세스를 이용해서 예전에도 혼자 개발할때 비슷한 경우를 공유한적이 있었습니다.

 [lora finetuning 후 EOS token이 안나오는 문제 지난번에 LoRA를 학습시키고 EOS 토큰이 나오는 확률이 낮아진거같은데...](https://hyeong9647.tistory.com/entry/lora-finetuning-%ED%9B%84-EOS-token%EC%9D%B4-%EC%95%88%EB%82%98%EC%98%A4%EB%8A%94-%EB%AC%B8%EC%A0%9C)



### 커스텀 로짓 프로세서

로짓 프로세서는 vLLM 내부에서 모델이 계산한 다음 토큰의 로짓(확률 전 단계)을 받아서 수정하는 역할을 합니다.특정 토큰을 아예 나오지 못하게 막을 수도 있고 반대로 특정 토큰의 점수를 크게 올려 선택되기 쉽게 만들 수도 있습니다.vLLM은 추론을 진행할 때마다 배치 단위로 로짓 프로세서를 호출하고, 각 요청에 해당하는 행(row)에만 변환을 적용합니다. 수정된 로짓 값은 이후 소프트맥스 계산을 거쳐 실제 출력 확률로 이어집니다. 뿐만 아니라 이와 비슷하게 온도나 top-p, top-k와 같은 디코딩 전략에서도 다음 토큰의 생성이 결정나게 됩니다.

### 로짓 프로세서 구현 시 필요한 요소

직접 로짓 프로세서를 만들려면 LogitsProcessor 클래스를 상속받아 몇 가지 메서드를 구현해야 합니다.

1.  apply(self, logits)  
    가장 핵심이 되는 부분입니다. 모델이 내놓은 로짓 텐서를 받아 필요한 조작을 하고 다시 반환합니다. 성능을 위해 벡터 연산을 사용해 한 번에 처리하는 것이 좋습니다.
2.  update\_state(self, batch\_update)  
    요청이 추가되거나 삭제되거나 이동되는 등 내부 상태가 바뀔 때 동기화하는 역할을 합니다. Remove → Add → Move 순으로 처리해야 혼란이 생기지 않습니다.
3.  validate\_params(cls, sampling\_params)  
    사용자가 전달한 커스텀 인자가 유효한지 확인합니다. 잘못된 인자는 오류로 처리해 예기치 않은 동작을 막습니다.
4.  is\_argmax\_invariant(self)  
    이 프로세서가 argmax(최대 로짓 토큰)를 바꾸지 않는다면 True를 반환합니다. 이 경우 vLLM이 그리디 샘플링에서 최적화를 수행할 수 있습니다.

no\_chinese\_plugin.py라는 파일을 만들어보겠습니다.

```
import torch
import re
from typing import Optional
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import LogitsProcessor, BatchUpdate


class NoChineseLogitsProcessor(LogitsProcessor):
    """
    생성 과정에서 한자(CJK 계열 문자)를 최대한 폭넓게 포함한 토큰을 마스킹하여
    나오지 않도록 하는 커스텀 Logits Processor입니다.
    """

    @classmethod
    def validate_params(cls, params: SamplingParams):
        # 별도 커스텀 파라미터 없음
        pass

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool):
        print("NoChineseLogitsProcessor: Initializing and identifying (almost all) CJK ideograph tokens...")

        # 1. 토크나이저 로드
        model_name = vllm_config.model_config.model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # 2. 한자 및 한자 계열 문자에 해당하는 유니코드 범위를 최대한 넓게 정의
        chinese_like_pattern = re.compile(
            r'['
            r'\u2e80-\u2eff'      # CJK Radicals Supplement
            r'\u2f00-\u2fdf'      # Kangxi Radicals
            r'\u2ff0-\u2fff'      # Ideographic Description Characters
            r'\u3005-\u3007'      # Ideographic iteration mark, ideographic number zero
            r'\u3038-\u303b'      # CJK-related symbols (一~十, 〻 등)
            r'\u3400-\u4dbf'      # CJK Unified Ideographs Extension A
            r'\u4e00-\u9fff'      # CJK Unified Ideographs (기본)
            r'\uf900-\ufaff'      # CJK Compatibility Ideographs
            r'\ufe30-\ufe4f'      # CJK Compatibility Forms
            r'\U00020000-\U0002a6df'  # CJK Unified Ideographs Extension B
            r'\U0002a700-\U0002b73f'  # Extension C
            r'\U0002b740-\U0002b81f'  # Extension D
            r'\U0002b820-\U0002ceaf'  # Extension E
            r'\U0002ceb0-\U0002ebe0'  # Extension F
            r'\U00030000-\U0003134f'  # Extension G
            r'\U00031350-\U000323af'  # Extension H
            r']+',
            flags=re.UNICODE,
        )

        banned_indices = []
        vocab_size = vllm_config.model_config.get_vocab_size()

        # 3. 전체 단어장을 순회하며 "한자/한자계열" 문자를 포함한 토큰 ID 식별
        for i in range(vocab_size):
            token_str = tokenizer.decode([i])
            if chinese_like_pattern.search(token_str):
                banned_indices.append(i)

        # 4. 차단할 인덱스를 텐서로 변환하여 GPU(device)에 올림
        if banned_indices:
            self.banned_indices_tensor = torch.tensor(banned_indices, dtype=torch.long, device=device)
        else:
            self.banned_indices_tensor = None

        print(f"NoChineseLogitsProcessor: Found {len(banned_indices)} tokens containing CJK ideograph-like characters.")

    def is_argmax_invariant(self) -> bool:
        # 로짓을 수정해 argmax가 바뀔 수 있으므로 False
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        # 전역 정적 마스킹이므로 상태 추적 불필요
        pass

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits 형태: (num_requests, vocab_size)
        한자/한자 계열 문자를 포함하는 토큰의 로짓을 -inf로 설정하여 선택되지 않도록 함
        """
        if self.banned_indices_tensor is not None and self.banned_indices_tensor.numel() > 0:
            logits[:, self.banned_indices_tensor] = float('-inf')
        return logits
```

라마 3.2 기준으로 4426 토큰이 생성되지 않게 되었습니다.

```
(EngineCore_DP0 pid=249649) NoChineseLogitsProcessor: Initializing and identifying (almost all) CJK ideograph tokens...
(EngineCore_DP0 pid=249649) NoChineseLogitsProcessor: Found 4426 tokens containing CJK ideograph-like characters.


llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    model='unsloth/Llama-3.2-1B-Instruct',
    temperature=0
)

print(llm.invoke([{'role':'user','content': "대한민국을 한자로 표기해줘"}]))
print(llm.invoke([{'role':'user','content': "中国的首都是？"}]))

# 로짓프로세서 적용 전
content='대한민국은 한자로 "大韓民國"으로 표기됩니다.' additional_kwargs={} response_metadata={'finish_reason': 'stop', 'model_name': 'unsloth/Llama-3.2-1B-Instruct'} id='run--8fa1fdfa-e54d-4380-8200-ab50e54d809d-0'
content='中国的首都是北京。' additional_kwargs={} response_metadata={'finish_reason': 'stop', 'model_name': 'unsloth/Llama-3.2-1B-Instruct'} id='run--ed839711-4016-4090-828a-c131c4407f8b-0'

# 로짓프로세서 적용 후
content='대한민국은 한자로 "중국"으로 표기할 수 있습니다.' additional_kwargs={} response_metadata={'finish_reason': 'stop', 'model_name': 'unsloth/Llama-3.2-1B-Instruct'} id='run--8e46d800-5376-4606-9bc6-5ec1ccc2712a-0'
content='《《《《《《《《《《《《《《《《《《《《《《《《《》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》》' additional_kwargs={} response_metadata={'finish_reason': 'length', 'model_name': 'unsloth/Llama-3.2-1B-Instruct'} id='run--32bdda9f-3a7b-48aa-8de1-18a663ebaf43-0'
```

주요 로드 방식은 다음과 같습니다:

1.  FQCN 문자열로 전달  
    클래스 경로를 문자열 형태로 넘기는 방식입니다.  
    예: your.module.path:ClassName

실행할때는 아래와 같이 Python path를 선언해서 클래스 경로를 먼저 할당해야 원활히 사용할 수 있습니다.

export PYTHONPATH=$PYTHONPATH:$(pwd)

vllm serve unsloth/Llama-3.2-1B-Instruct --dtype half --max\_model\_len 1024 --gpu\_memory\_utilization 0.65 --logits-processors no\_chinese\_plugin:NoChineseLogitsProcessor

그외 방법으로는엔트리 포인트 등록,pyproject.toml에 vllm.logits\_processors 항목을 등록해두면 vLLM이 자동으로 인식합니다.클래스 객체 직접 전달,Python 코드에서 LLM 또는 AsyncLLM 생성 시 클래스 자체를 넘기는 방식으로, 오프라인 환경에서만 사용됩니다.

커스텀 로짓 프로세서는 모델의 성능을 떨어뜨리지 않으면서도 출력 조작을 세밀하게 할 수 있는 강력한 기능입니다. 이를 잘 활용하면 LLM의 생성 방식을 원하는 수준까지 통제할 수 있어, 애플리케이션의 품질과 사용자 경험을 한층 더 높일 수 있습니다.

**참고자료** 

[https://docs.vllm.ai/en/stable/features/custom\_logitsprocs/](https://docs.vllm.ai/en/stable/features/custom_logitsprocs/)
