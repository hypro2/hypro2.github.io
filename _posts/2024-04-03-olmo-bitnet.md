---
layout: post
title: 모델 리뷰 OLMo Bitnet 1B을 colab에서 실행해보자
---

요즘 화두에 있는 Bitnet 양자화를 직접 구현했다는 NousResearch의 OLMo-Bitnet-1B을 리뷰해볼 예정입니다.  
NousResearch에서 제시한 방식으로 실행을 하려고 합니다. 모델 및 실행에 필요한 코드는 레포지토리에 모델과 같이 trust\_remote\_code=True을 하면 실행 할 수 있습니다. NousResearch에서에서 구현한 BitLinear158 클래스 먼저 살펴 볼 예정입니다.



## NousResearch에서에서 구현한 BitLinear158 클래스

해당 코드는 PyTorch를 사용하여 비트 정밀도(1.58 비트)로 선형 레이어를 구현하는 것으로 보입니다. 코드의 구성 요소를 살펴보겠습니다:

weight\_quant 함수: 이 함수는 가중치 텐서를 1.58 비트로 양자화합니다. 텐서의 각 요소에 대해 절대값을 취한 후 평균을 구한 다음, 그 평균의 역수로 스케일링하여 양자화된 가중치를 생성합니다.

RMSLayerNorm 클래스: 이 클래스는 RMS 레이어 정규화를 구현합니다. 입력 텐서의 각 요소를 제곱하여 분산을 계산하고, 이를 제곱근으로 나누어 정규화합니다. 이 클래스는 LayerNormBase 클래스에서 상속됩니다.

BitLinear158 클래스: 이 클래스는 1.58 비트 정밀도를 사용하여 선형 레이어를 정의합니다. 입력 텐서를 정규화한 후 양자화된 가중치와 입력을 사용하여 선형 변환을 수행합니다. 이 클래스는 nn.Linear 클래스에서 상속됩니다.

이 코드의 주요 특징은 다음과 같습니다:

입력과 가중치를 양자화하여 메모리 사용량을 줄이고 연산 속도를 향상시킵니다. RMS 레이어 정규화를 사용하여 입력을 정규화합니다. Straight-Through Estimator(STE)를 사용하여 역전파 중에 가중치와 입력을 양자화합니다.

```
def weight_quant(w):
    """Per−tensor quantization to 1.58 bits. No grouping is needed for quantization.
    Args:
    w: a weight tensor with shape [d, k]
    Returns:
    u: a quantized weight with shape [d, k]
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


class BitLinear158(nn.Linear):
    """
    This is only for training, and kernel optimization is needed for efficiency.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, config=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.norm = RMSLayerNorm(config, elementwise_affine=False)

    def forward(self, x):
        """
        Args:
        x: an input tensor with shape [n, d]
        Returns:
        y: an output tensor with shape [n, d]
        """
        w = self.weight  # a weight tensor with shape [d, k]
        x_norm = self.norm(x)
        # Atrick for implementing Straight−Through−Estimator (STE) using detach()
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y
```

### STE는 무엇인가?

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbjr3f0%2FbtsGj1cQeOw%2Fmg9gbhXE0F26wnuhWMxiXK%2Fimg.png)

STE는 "Straight Through Estimator"의 약자로, 중량 양자화나 활성화 양자화와 같은 연산에서 사용되는 기술입니다. 이는 역전파(backpropagation) 중에 발생하는 그레이디언트(gradient)의 전달을 보다 쉽게 하기 위한 방법 중 하나입니다.

일반적으로 가중치나 활성화를 양자화할 때, 연속적인 값 대신에 이산적인 값으로 변환됩니다. 이 과정에서 역전파가 잘 작동하지 않을 수 있습니다. STE는 이런 문제를 해결하기 위해 개발되었습니다.

STE를 사용하면 양자화된 값으로의 전파는 그대로 이루어지지만, 역전파는 원래의 연속적인 값을 이용하여 수행됩니다. 즉, 역전파는 양자화된 값을 건너뛰고 원래의 연속적인 값을 사용하여 이루어지며, 이를 통해 역전파의 효과를 유지할 수 있습니다.

이것은 양자화된 값과 원래의 값을 동일하게 취급하는 것처럼 보이지만, 역전파 과정에서는 원래의 값을 사용하여 그레이디언트를 전파합니다. 이렇게 함으로써 양자화 과정이 역전파에 영향을 미치지 않으면서도 모델의 학습이 원활하게 이루어지도록 합니다.

STE(직선 추정기)는 신경망에서 사용되는 이산적인 함수(예: 임계값 함수)의 역전파(backpropagation)를 위한 추정기입니다. 이것은 일반적으로 활성화 함수의 출력을 이진 값으로 바꾸는 데 사용됩니다.

이러한 이산 함수는 일반적으로 연속적인 값을 이산적인 값으로 변환하지만, 역전파 과정에서 문제를 일으킬 수 있습니다. 이러한 함수는 보통 0을 기준으로 분기되며, 이는 그래디언트가 0이 되어 역전파 동안 정보를 전달하지 않는 문제를 야기할 수 있습니다.

STE는 이러한 문제를 해결하기 위해 도입되었습니다. STE는 역전파 동안 이러한 이산 함수의 그래디언트를 "통과"시켜 연속적인 함수의 그래디언트처럼 보이게 합니다. 즉, STE는 이산 함수의 역전파 동안 그래디언트를 모조리 0이 아닌 값으로 전달하여 네트워크가 학습할 수 있도록 돕습니다.

## 모델 실행

```
!pip install -q omegaconf
!pip install -q cached_path boto3 botocore
!pip install -q -U transformers peft accelerate optimum
```

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer

tokenizer = AutoTokenizer.from_pretrained("NousResearch/OLMo-Bitnet-1B")
model = AutoModelForCausalLM.from_pretrained("NousResearch/OLMo-Bitnet-1B",
                                             torch_dtype=torch.bfloat16,
                                             trust_remote_code=True,
                                             device_map="auto")

streamer = TextStreamer(tokenizer)

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.8,
                repetition_penalty=1.1,
                do_sample=True,
                streamer=streamer)

pipe("The capitol of Paris is",  max_new_tokens=256)
```

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F7AV1v%2FbtsGhUlS7Fp%2FARfLsKoyx4Nsk9BXgOIAXK%2Fimg.png)

```
[{'generated_text': 'The capitol of Paris is a historic and beautiful square that is full of symbolism, the square stands for the French capital. In 2015 in honour of the 50th anniversary of the foundation of the City Square, they decided to do away with a square known as the National Assembly-Parliament and rebrand it as the Congress of the People’s Republic of France, so called as the Foulon Constitution.\nThe new President was elected on a very short leash. He started his inauguration with an address that has been translated into several languages: “To our people—our children—the great workers who work behind the scenes to deliver the infrastructure projects we would like to see completed… we will be working together with all the countries throughout the world, working together for the realization of the projects that are important to us”.\nThat is why he made it clear that the future of politics lies not just in the construction but also in the future of the economy of a society: that after 2025, there will be no more building jobs for young workers; that between 15 and 20 million people will have no choice but to move out of their homes; that those who will be forced to stay in their houses must live on a daily basis – that the citizens who cannot pay their rent should be'}]
```

"""파리의 국회의사당은 프랑스의 수도를 상징하는 역사적이고 아름다운 광장으로, 상징성이 가득한 곳입니다. 2015년 시티 광장 건립 50주년을 기념하여 국회-의회로 알려진 광장을 없애고 프랑스 인민공화국 의회, 이른바 풀롱 헌법으로 이름을 바꾸기로 결정했습니다. 새로운 대통령은 매우 짧은 임기로 선출되었습니다. 그는 여러 언어로 번역된 취임 연설로 취임식을 시작했습니다: "우리가 바라는 인프라 프로젝트를 완수하기 위해 보이지 않는 곳에서 일하는 위대한 일꾼인 우리 국민, 우리 아이들에게... 우리는 전 세계 모든 국가와 협력하여 우리에게 중요한 프로젝트의 실현을 위해 함께 노력할 것입니다."라고 말입니다. 그렇기 때문에 그는 정치의 미래는 건설뿐만 아니라 한 사회의 경제의 미래에도 있다는 점을 분명히했습니다. 2025 년 이후에는 더 이상 젊은 노동자들을위한 건설 일자리가 없을 것입니다. 1,500 ~ 2 천만 명의 사람들이 집을 떠날 수밖에 없으며, 집에 머물 수밖에없는 사람들은 매일 생활해야하며, 집세를 지불 할 수없는 시민들은 다음과 같이해야한다고 말했습니다."""

**#한글 성능 테스트**

```
pipe("판다의 이름은 푸바오 입니다. 판다는", max_new_tokens=2048)

[{'generated_text': '판다의 이름은 푸바오 입니다. 판다는 데이트신지만 첫팬를 정도하는, 어려웨어 국개의 과정으로 그것은 경운 있습니다.\n(1) 교라고, 가장 사고 : (2) 낮아 불문해 적합니다.\n1) 교라고, 곿선을 다른 굉임하면 속에서 확응 혜즌가 있습니다. 액직한 경우, 좋은 및 경책으로 다리워요.\n1) 파-이 속깨가는 말아니 맞했습니다. 파-고들은 전편럽품 필요하는 주자 한인을 단시기 발표항읻을 보여하면 수 있습니다.\n2) 우전부탁 대장의 의퇴브를 기라면 구전 타입로 있습니다.\n1) 파-고들과 대장대읾문해 퐔트 캐부터를 만난다.\nTaiwan. 세계: TTS.|||IP_ADDRESS|||'}]
```

영문은 자잘한 이슈는 있지만 잘 생성하는 모습을 볼 수 있습니다. 한글에 경우는 어쩔수 없나봅니다. 

모델 자체가 4기가가 넘는데  2.5기가의 GPU로만으로 구동되는 모습을 확인 할 수 있었습니다. 

어쩌면 GPU를 이용하는 새로운 양자화 방식이 우후죽순 생겨날 수 있겠군요
