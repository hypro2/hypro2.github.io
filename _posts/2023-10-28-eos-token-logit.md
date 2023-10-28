---
layout: post
title: lora finetuning 후 EOS token이 안나오는 문제
---


지난번에 LoRA를 학습시키고 EOS 토큰이 나오는 확률이 낮아진거같은데... 어떻게 해결 할 수 있는 방법이 있는가 구글링을 통해서 찾아 보았다. 


[https://towardsdatascience.com/challenges-in-stop-generation-within-llama-2-25f5fea8dea2](https://towardsdatascience.com/challenges-in-stop-generation-within-llama-2-25f5fea8dea2)

이글을 보면 EOS 토큰의 확률이 생성될 가능성을 확인하는 방법을 알려줌

아래 코드를 잘 훔쳐 쓰겠습니다. 선생님

아래 코드는 Python으로 작성된 클래스인 EosTokenRewardLogitsProcessor를 정의하고 있습니다. 이 클래스는 LogitsProcessor 클래스를 상속받아서 만들어진 것으로 보입니다.

이 클래스는 주어진 input_ids와 scores에 대해 eos_token의 보상을 문장 길이에 따라 동적으로 조절하는 역할을 합니다. 문장 길이가 길어질수록 eos_token에 대한 보상이 높아지며, 다른 토큰의 점수도 비례적으로 조정됩니다. 클래스의 생성자에서는 eos_token_id와 max_length를 설정하고, __call__ 메서드에서 보상을 계산하여 반환합니다.


```
class EosTokenRewardLogitsProcessor(LogitsProcessor):
  def __init__(self, eos_token_id: int, max_length: int):
    # `eos_token_id`와 `max_length`의 유효성을 검사하는 조건문입니다.
    if not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

    if not isinstance(max_length, int) or max_length < 1:
        raise ValueError(f"`max_length` has to be an integer bigger than 1, but is {max_length}")

    # 생성자 메서드에서 `eos_token_id`와 `max_length`를 클래스 멤버 변수로 저장합니다.
    self.eos_token_id = eos_token_id
    self.max_length = max_length

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    cur_len = input_ids.shape[-1]
    # 보상을 증가시키는 과정을 설명하는 주석입니다.
    # eos_token의 보상을 문장 길이에 따라 조절합니다.
    for cur_len in (max(0, int(self.max_length * 0.8)), self.max_length):
      ratio = cur_len / self.max_length
      num_tokens = scores.shape[1]  # 어휘 크기 (단어 집합의 크기)
      
      # eos_token이 아닌 다른 토큰의 점수를 조정합니다.
      scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]] = \
        scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]] * ratio * 10 * torch.exp(-torch.sign(scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]]))
      
      # eos_token의 보상을 특정 상수로 설정합니다.
      scores[:, self.eos_token_id] = 1e2 * ratio
    
    # 조정된 점수가 저장된 scores를 반환합니다.
    return scores



pipe = transformers.pipeline(model=model,
tokenizer=tokenizer,
return_full_text=True,  # langchain expects the full text
task='text-generation',
# we pass model parameters here too
#stopping_criteria=stopping_criteria,  # without this model rambles during chat
logits_processor=logits_process_list,
max_new_tokens=500,  # max number of tokens to generate in the output
temperature=0.1,
)

```

StoppingCriteriaList와 조합해서 쓰면 EOS 토큰 생성 문제는 해결 할 수 있을 거같습니다 감사합니다.
