---
layout: post
title: llama2를 context 8k까지 확장하는 방법 RoPE, exllama
---
해당 부분의 max\_seq\_len을 늘려주고 compress\_pos\_emb 혹은 alpha\_value를 지정해준 것입니다. 여기서 원래 지원하는 max 시퀀스의 길이를 늘리고 싶은 만큼의 배수를 넣어주면 적용하면 됩니다.  이것이 어떻게 되는 것을 제가 이해하기로는 Position embedding은 Self attention의 포지션에 대한 위치를 기억 시키기 위해 사용이 되는 중요한 요소중 하나 입니다. Llama는 Rotary Position Embedding은 회전행렬을 사용하여 위치에 대한 정보를 인코딩 하는 방식으로 구현되어 있습니다.

기존의 Position Embedding 부터 알고 가야되는데, Position Embedding은 단순히 임베딩 벡터에 순서를 매겨서 위치 정보를 인코딩합니다. Rotary Position Embedding은 회전행렬을 사용하여 위치에 대한 정보를 인코딩하는 방식을 사용합니다. 회전행렬은 벡터를 회전시키는 행렬로, 벡터의 크기는 유지하면서 방향을 바꿀 수 있습니다. Rotary Position Embedding은 Transformer 모델의 Attention 메커니즘에서 입력 벡터를 회전시키는 방식으로 위치 정보를 인코딩합니다. Attention 메커니즘은 입력 벡터의 위치에 따라 가중치를 부여하는 방식으로 작동합니다. Rotary Position Embedding은 입력 벡터의 위치에 따라 회전행렬을 생성하여, 기존의 절대 위치의 Position Embedding보다 Attention 메커니즘이 입력 벡터의 위치를 더 잘 이해할 수 있도록 합니다.

그러면, 트랜스포머 기반 모델은 학습된 토큰 사이즈에서 1024라면 훈련한 다음, 1024를 넘어가는 1025 토큰을 주어서 이제 1025 위치에서 단어를 보고 마치 '1025가 뭐야? 이런 건 처음 봤어' 혼란해 한다는 것입니다.

여기서 Rotary Position Embedding의  frequency window은 RoPE에서 사용되는 매개변수로, 입력 벡터의 위치에 대한 정보를 인코딩하는 역할을 합니다. frequency window을 0.25배로 축소하면 입력 벡터의 위치에 대한 정보를 더 세분화하여 인코딩할 수 있습니다. 이로 인해 Attention 메커니즘이 입력 벡터의 위치를 더 잘 이해할 수 있게 되고, 성능이 향상됩니다. 새로운 길이로 외삽(길이를 늘려서 학습)하는 대신에, 시퀀스가 학습 한 인코딩 범위 내에있는 것처럼 보이는 한 모델이 잘 수행 되게끔 하는 겁니다.

RoPE의 frequncy window를 0.25배로 축소하여 매우 간단한 테스트를 수행했습니다. 이렇게 하면 중간에 4단계로 인코딩을 보간하는 효과가 있으므로 위치 1은 위치 0.25, 위치 40은 위치 10, 위치 2048은 위치 512처럼 보입니다. 놀랍게도 효과가 있었다고 합니다. 저자는 이점을 확인하기 위해 모델을 미세 조정할 필요조차 없었습니다. 또한, 미세 조정한 후에는, 그 효과는 더욱 두드러져 토큰 6000에서 완벽하게 작동했습니다.

```
class ExllamaHF(PreTrainedModel):
    def __init__(self, config: ExLlamaConfig):
        super().__init__(PretrainedConfig())
        self.ex_config = config
        self.ex_model = ExLlama(self.ex_config)
        self.ex_cache = ExLlamaCache(self.ex_model)
        self.generation_config = GenerationConfig()
        self.lora = None

...중략...

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):

        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        config = ExLlamaConfig(pretrained_model_name_or_path / 'config.json')

        weight_path = None
        for ext in ['.safetensors', '.pt', '.bin']:
            found = list(pretrained_model_name_or_path.glob(f"*{ext}"))
            if len(found) > 0:
                weight_path = found[-1]
                break

        config.model_path = str(weight_path)
        config.max_seq_len = 8192
        config.compress_pos_emb = 2
        # config.alpha_value = 2

        config.calculate_rotary_embedding_base()

        if torch.version.hip:
            config.rmsnorm_no_half2 = True
            config.rope_no_half2 = True
            config.matmul_no_half2 = True
            config.silu_no_half2 = True

        return ExllamaHF(config)
```

코드 상 중요한 부분은 config를 수정해주는 것이다. exllama상 RoPE가 이미 구현되어 있기 때문에 따로 해줄건 없다고 한다 ㅎ

```
config.max_seq_len = 8192
config.compress_pos_emb = 2
# config.alpha_value = 2
```


레딧에 공유된 글 

[https://www.reddit.com/r/LocalLLaMA/comments/14j4l7h/6000\_tokens\_context\_with\_exllama/](https://www.reddit.com/r/LocalLLaMA/comments/14j4l7h/6000_tokens_context_with_exllama/)

[From the LocalLLaMA community on Reddit: 6000+ tokens context with ExLlama Explore this post and more from the LocalLLaMA community www.reddit.com](https://www.reddit.com/r/LocalLLaMA/comments/14j4l7h/6000_tokens_context_with_exllama/)

제일먼저 context를 확장법을 알린 사람이 정리한 문서

[https://kaiokendev.github.io/context](https://kaiokendev.github.io/context)

[Extending Context is Hard pages kaiokendev.github.io](https://kaiokendev.github.io/context)
