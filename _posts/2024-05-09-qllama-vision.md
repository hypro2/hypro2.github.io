---
layout: post
title: 멀티모달 리뷰 qresearch의 llama3-vision-alpha 콜랩 구동
---

LLM RnD 자료를 찾으러 Note에서 일본 LLM 동향을 검색하고 있었는데 qresearch라는 곳에서 llama3로 vision모델을 만들었다는 글을 보았습니다. 그냥 자기 것이 성능이 우수하다 이런 내용이 아닌 만들어서 코드 리뷰하는 문서 였습니다. 생각보다 유익한 내용인거 같아서 따라 구동 해봤습니다.

간단히 코드 구동이 가능합니다. 이 경우에 허깅페이스 레포지토리에서 lama-3-vision-alpha/mm\_projector.bin만 들어있는데 그 이외에 파일은 튜닝을 따로 시키지 않은 llama3와 siglip 모델을 사용해서 중간의 projection층을 만들어서 그것만으로 vision 모델을 구현한 것으로 llava와 비슷하게 구현된 것을 볼 수 있었습니다. 



```
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "qresearch/llama-3-vision-alpha-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
)

image = Image.open("/content/sdfsdfsdfdsf.jpg")

print(
    tokenizer.decode(
        model.answer_question(image, "what is this?", tokenizer),
        skip_special_tokens=True,
    )
)
```

프로젝션의 구조는 아주 간단한 시퀀셜 리니어 모델과 액티베이션이 전부이고, 임베딩 레이어에서 concat해주므로서 비전 능력을 얻을 수 있다는 것을 볼 수 있었고, 그래서 저는 기존 llama3 모델 말고 다른 튜닝된 llama3로 바꿔서 실행도 해봤습니다. 그랬을 때도 비슷하게 동작하는 것을 보아 단순해보이면서도 강력하게 구현 할 수 있다는게 놀라웠습니다. 훈련 하는 방법도 직접 코드를 보거나 구현해보는 것을 해보고 싶다고 생각들었습니다.

```
class ProjectionModule(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size):
        super(ProjectionModule, self).__init__()

        # Directly set up the sequential model
        self.model = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.model(x)
```

```
def process_tensors(input_ids, image_features, embedding_layer):
    # Find the index of -200 in input_ids
    split_index = (input_ids == -200).nonzero(as_tuple=True)[1][0]

    # Split the input_ids at the index found, excluding -200
    input_ids_1 = input_ids[:, :split_index]
    input_ids_2 = input_ids[:, split_index + 1 :]

    # Convert input_ids to embeddings
    embeddings_1 = embedding_layer(input_ids_1)
    embeddings_2 = embedding_layer(input_ids_2)

    device = image_features.device
    token_embeddings_part1 = embeddings_1.to(device)
    token_embeddings_part2 = embeddings_2.to(device)

    # Concatenate the token embeddings and image features
    concatenated_embeddings = torch.cat(
        [token_embeddings_part1, image_features, token_embeddings_part2], dim=1
    )

    # Create the corrected attention mask
    attention_mask = torch.ones(
        concatenated_embeddings.shape[:2], dtype=torch.long, device=device
    )
    return concatenated_embeddings, attention_mask
```

```
        image_features = image_forward_outs.hidden_states[-2]

        projected_embeddings = projection_module(image_features).to("cuda")

        new_embeds, attn_mask = process_tensors(
            input_ids, projected_embeddings, embedding_layer
        )
        device = model.device
        attn_mask = attn_mask.to(device)
        new_embeds = new_embeds.to(device)
```

![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdEqJde%2FbtsHkYsb3ts%2FZu9IeONkb1suwaYeJ1EGGk%2Fimg.png)

참고자료

[https://huggingface.co/qresearch/llama-3-vision-alpha/blob/main/\_\_main\_\_.py](https://huggingface.co/qresearch/llama-3-vision-alpha/blob/main/__main__.py)

[https://note.com/astropomeai/n/n89124686697f](https://note.com/astropomeai/n/n89124686697f)
