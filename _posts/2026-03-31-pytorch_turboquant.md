---
layout: post
title: TurboQuant 고차원 회전을 이용한 KV 캐시 양자화 및 PyTorch 실전 구현
---
## TurboQuant: 고차원 회전을 이용한 KV 캐시 양자화 및 PyTorch 실전 구현

대규모 언어 모델(LLM) 추론 시 컨텍스트 길이가 길어짐에 따라 **KV 캐시(Key-Value Cache)**가 점유하는 VRAM 용량은 기하급수적으로 증가합니다. 이는 긴 문맥을 처리해야 하는 서비스에서 가장 큰 비용 병목 구간이 됩니다. 

2026년 ICLR에서 발표된 **TurboQuant**는 고차원 벡터의 수학적 특성을 활용하여 별도의 데이터 학습(Calibration) 없이도 KV 캐시를 3~4비트 수준으로 압축하는 혁신적인 방법론을 제시했습니다. 본 포스팅에서는 `turboquant-pytorch` 구현체를 바탕으로 그 작동 원리와 실전 사용법을 분석합니다.




---

**TurboQuant의 핵심 작동 원리**

TurboQuant의 차별점은 **데이터 독립적(Data-oblivious)** 설계에 있습니다. 기존 양자화 방식이 특정 데이터셋을 통해 최적의 파라미터를 찾는 것과 달리, TurboQuant는 모든 벡터를 무작위로 회전시켜 통계적으로 예측 가능한 분포로 변환합니다.

**1. 무작위 직교 회전 (Random Orthogonal Rotation)**
입력 벡터 $x$에 무작위 직교 행렬 $\Pi$를 곱하면, 벡터의 에너지가 모든 차원에 고르게 분산됩니다. 차원이 충분히 높을 경우, 회전된 좌표값들은 0 근처에 강하게 집중되는 가우시안 분포에 근사하게 됩니다. 이를 통해 모델이나 데이터에 상관없이 사전에 계산된 **로이드-맥스(Lloyd-Max) 코드북**을 공통적으로 적용할 수 있습니다.

**2. 비편향 내적을 위한 QJL 보정**
단순 MSE(평균 제곱 오차) 최적화만으로는 어텐션 계산의 핵심인 '내적(Inner Product)'에서 미세한 편향이 발생할 수 있습니다. TurboQuant는 **QJL(Quantized Johnson-Lindenstrauss)** 변환을 통해 잔차의 부호 정보를 1비트로 추가 저장함으로써, 수학적으로 내적의 기댓값이 원본과 동일하도록 보정합니다.

---

## PyTorch 실전 구현 분석

`turboquant-pytorch` 라이브러리를 활용하여 실제 KV 캐시 시뮬레이션 환경에서 양자화를 수행하는 과정을 살펴봅니다.

### 환경 설정 및 저장소 로드
먼저 필요한 라이브러리를 설치하고 소스 코드를 가져옵니다.

```bash
# 저장소 클로닝 및 의존성 설치
!git clone https://github.com/tonbistudio/turboquant-pytorch.git
!pip install torch --index-url https://download.pytorch.org/whl/cu128
!cd turboquant-pytorch && pip install -r requirements.txt
```

### 핵심 알고리즘 사용 예시
`TurboQuantProd` 클래스는 내적 정밀도를 보장하는 2단계 양자화를 수행합니다.

```python
import torch
import sys
sys.path.insert(0, '/content/turboquant-pytorch')
from turboquant import TurboQuantProd

# 1. KV 캐시 벡터 생성 (128차원, 1000개 토큰)
d = 128
n = 1000
vectors = torch.randn(n, d)

# 2. 3비트 양자화 설정 (내적 보정 포함)
tq = TurboQuantProd(bits=3, d=d)

# 3. 압축 및 복원 수행
compressed = tq.quantize(vectors)
reconstructed = tq.dequantize(compressed)

# 4. 내적 정확도 확인 (어텐션 계산 시뮬레이션)
query = torch.randn(1, d)
true_attn = query @ vectors.T
approx_attn = tq.inner_product(query, compressed)

print(f"Original Attention: {true_attn[0, :5]}")
print(f"Approx Attention: {approx_attn[0, :5]}")
```

### KV 캐시 메모리 절감 효과 측정
`TurboQuantKVCache` 래퍼를 사용하면 실제 모델에 적용했을 때의 메모리 사용량을 확인할 수 있습니다.

```python
from turboquant import TurboQuantKVCache

# 캐시 인스턴스 생성 (Key: 4비트, Value: 2비트 권장)
cache = TurboQuantKVCache(d_key=128, d_value=128, bits=3)

# 더미 데이터 삽입
keys = torch.randn(1024, 128)
values = torch.randn(1024, 128)
cache.append(keys, values)

# 메모리 사용량 확인
usage = cache.memory_usage_bits()
print(f"Compression Ratio: {usage['compression_ratio']:.2f}x")
# 출력 결과 예시: Compression Ratio: 5.22x (FP16 대비)
```

---

## 주요 벤치마크 결과 (Tesla T4 기준)

실제 GPU 환경에서의 테스트 결과, 다음과 같은 성능 지표를 보여줍니다.

| 지표 | FP16 (Baseline) | TurboQuant (3-bit) | 비고 |
|:---|:---:|:---:|:---|
| **VRAM 점유 (8K context)** | 2048 KB | 384 KB | **약 5.3배 절감** |
| **Needle-in-Haystack** | 100% | 100% | 정보 검색 정확도 유지 |
| **내적 RMSE** | 0.000 | 0.036 | 매우 낮은 왜곡률 |

### 구현 버전별 특징 (V2 vs V3)
실제 LLM 생성 테스트 결과, 공식 논문의 QJL 보정(V2)보다 MSE 최적화에 집중한 커뮤니티 방식(V3)이 Softmax 정합성 측면에서 더 유리한 것으로 나타났습니다. 
- **V2**: 내적 편향은 없으나 분산이 커서 Softmax 이후 Top-K 정합성이 흔들릴 수 있음.
- **V3**: QJL 보정 대신 모든 비트를 MSE 품질에 할당하여 생성 품질을 극대화함.

---
업로드하신 `turboquant_pytorch.ipynb` 파일의 **Generation Test** 섹션을 바탕으로, **Qwen2.5-3B-Instruct** 모델이 실제 생성 과정에서 어떻게 TurboQuant 캐시를 사용하는지 코드를 중심으로 설명하겠습니다.

---

## Qwen2.5 모델과 TurboQuant V3의 실전 통합

본 테스트의 목적은 **"극단적인 압축(평균 3-bit) 상태에서도 모델이 긴 문맥의 정보를 정확히 기억하고 답변할 수 있는가"**를 검증하는 것입니다. 이를 위해 8,192 토큰의 긴 문맥 속에 특정 정보(Needle)를 숨겨두고 이를 인출하는 테스트를 수행합니다.

### 1. 모델 로드 및 설정
먼저 성능과 효율의 균형이 뛰어난 **Qwen2.5-3B-Instruct** 모델을 로드합니다. 모델 가중치는 4-bit(NF4)로 양자화하여 로드함으로써 베이스라인 메모리 점유율을 낮춥니다.

```python
# 모델 및 토크나이저 로드
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.float16, 
        bnb_4bit_quant_type="nf4"
    ),
    device_map="auto", 
    torch_dtype=torch.float16,
)
```

### 2. 생성 도중의 KV 캐시 작동 방식: `V3Cache`
텍스트 생성 시, 모델은 `past_key_values`라는 객체에 이전 토큰들의 정보를 저장합니다. `V3Cache`는 이 정보를 실시간으로 압축하도록 설계되었습니다.

```python
class V3Cache(DynamicCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # 1. 신규 토큰 유입
        # 2. 최근 윈도우(예: 128토큰)는 FP16 정밀도로 'fp16_recent' 버퍼에 유지
        # 3. 윈도우를 초과하는 '오래된' 토큰들은 TurboQuantV3 알고리즘으로 압축
        
        if recent_k.shape[2] > self.residual_window:
            # 윈도우를 벗어난 부분만 골라서 압축
            ck, cv = comp.compress_kv(to_compress_k, to_compress_v)
            self._chunks_k[layer_idx].append(ck)
            self._chunks_v[layer_idx].append(cv)

        # 4. 어텐션 계산 시: 압축된 청크 복원 + FP16 최근 버퍼 결합
        full_k = torch.cat([decompress(ck) for ck in chunks] + [recent_k], dim=2)
        return full_k, full_v
```

### 3. "건초더미 속 바늘" 테스트 실행
이제 긴 문장들 사이에 비밀 코드(`AURORA-7749`)를 숨긴 프롬프트를 생성하고, 압축된 캐시를 사용하여 답변을 생성합니다.

```python
# 테스트 구성 (Key 4-bit / Value 2-bit 설정)
config = {"key_bits": 4, "value_bits": 2, "residual_window": 128, "label": "V3 K4/V2"}

# 캐시 객체 생성
cache = V3Cache(
    key_bits=config["key_bits"],
    value_bits=config["value_bits"],
    residual_window=config["residual_window"],
    n_layers=model.config.num_hidden_layers
)

# 답변 생성
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        past_key_values=cache, # TurboQuant 캐시 주입
        max_new_tokens=32,
        use_cache=True
    )

# 결과 확인: "AURORA-7749"가 포함되어 있는지 확인
response = tokenizer.decode(outputs[0][input_ids.shape[1]:])
```

---

## 생성 테스트 결과 요약

노트북의 실행 결과에 따르면, Qwen2.5 모델은 TurboQuant를 적용했을 때 다음과 같은 성능을 보였습니다:

* **정확도**: 2,048개부터 8,192개 토큰까지의 모든 테스트 구간에서 **FP16 베이스라인과 동일하게** 정답(`AURORA-7749`)을 찾아냈습니다.
* **비대칭 압축의 이점**: 특히 **Key 4-bit / Value 2-bit** 설정은 전체 KV 캐시를 약 **5.22배** 압축하면서도 정보 손실이 거의 없음을 확인했습니다.
* **결론**: 생성 도중 최근 문맥(Residual Window)을 FP16으로 보호하고 나머지를 TurboQuant로 압축하는 방식은 실제 대화 품질을 유지하면서 VRAM을 획기적으로 아끼는 실전적인 전략입니다.

TurboQuant는 고차원 기하학적 원리를 이용해 LLM 추론의 가장 큰 숙제인 KV 캐시 메모리 문제를 해결합니다.

1. **학습 데이터 불필요**: 무작위 회전 덕분에 어떤 모델에도 즉시 적용 가능합니다.
2. **높은 압축률**: FP16 대비 5배 이상의 메모리 절감 효과를 제공합니다.
3. **비대칭 비트 할당 가능**: 중요도가 높은 Key 벡터에는 4비트, 데이터 비중이 큰 Value에는 2비트를 할당하여 최적의 균형을 찾을 수 있습니다.

VRAM 한계로 인해 긴 컨텍스트 처리에 어려움을 겪고 있다면, TurboQuant는 가장 수학적으로 견고하고 효율적인 대안이 될 것입니다.

---

**관련 링크:**
- TurboQuant 공식 GitHub: [https://github.com/tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)
- ICLR 2026 논문: "TurboQuant: Online Vector Quantization"