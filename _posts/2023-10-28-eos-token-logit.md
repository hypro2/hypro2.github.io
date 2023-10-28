---
layout: post
title: lora finetuning 후 EOS token이 안나오는 문제
---


지난번에 LoRA를 학습시키고 EOS 토큰이 나오는 확률이 낮아진거같은데... 어떻게 해결 할 수 있는 방법이 있는가 구글링을 통해서 찾아 보았다. 


[https://towardsdatascience.com/challenges-in-stop-generation-within-llama-2-25f5fea8dea2](https://towardsdatascience.com/challenges-in-stop-generation-within-llama-2-25f5fea8dea2)

이글을 보면 EOS 토큰의 확률이 생성될 가능성을 확인하는 방법을 알려줌

아래 코드를 잘 훔쳐 쓰겠습니다. 선생

```
class EosTokenRewardLogitsProcessor(LogitsProcessor):
  def __init__(self,  eos_token_id: int, max_length: int):
    
        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        if not isinstance(max_length, int) or max_length < 1:
          raise ValueError(f"`max_length` has to be a integer bigger than 1, but is {max_length}")

        self.eos_token_id = eos_token_id
        self.max_length=max_length

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    cur_len = input_ids.shape[-1]
    # start to increese the reward of the  eos_tokekn from 80% max length  progressively on length
    for cur_len in (max(0,int(self.max_length*0.8)), self.max_length ):
      ratio = cur_len/self.max_length
      num_tokens = scores.shape[1] # size of vocab
      scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]] =\
      scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]]*ratio*10*torch.exp(-torch.sign(scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]]))
      scores[:, self.eos_token_id] = 1e2*ratio
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

StoppingCriteriaList와 조합해서 쓰면 EOS 토큰 생성 문제는 해결 할 수 있을 거같다 감사합니다.
