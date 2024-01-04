---
layout: post
title: Bert Lora Classification 학습기
---

오늘은 쉬운거로 Peft를 이용한 LoRA학습기를 또 해보았다. 하지만 이번에는 Bert를 이용한 분류 문제를 가져왔습니다. LoRA는 LLM의 파인튜닝으로 많이 사용되지만 BERT의 클래시피케이션에도 사용가능합니다. 

## LoRA : Low-Rank Adaptation of Large Language Models

https://github.com/microsoft/LoRA
https://github.com/huggingface/peft

![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/a1d38968-af68-4dc8-b177-ad262a01984f)

마이크로소프트에서는 LLM을 파인튜닝하기 위해 개발한 방법은 기존 모델의 가중치를 동결한 후, 학습 가능한 rank decomposition matrices(순위 분해 행렬)를 트랜스포머 아키텍처의 각 계층에 주입하는 방식입니다.

이는 GPU 메모리 요구량과 훈련 가능한 파라미터 수를 획기적으로 줄일 수 있는 장점이 있습니다.

Pretrained Weights는 동결시키고 A와 B만 학습합니다. 여러 항목 중에서 LoRA가 가장 일정하게 좋은 정확도를 유지하는 것을 알 수 있습니다.

따라서, 사전 학습된 모델을 그대로 공유하면서 작은 LoRA 모듈을 여러 개 만들 수 있습니다. 모델을 공유하면서 새로 학습시키는 부분(A, B)만 쉽게 바꿔 끼울 수 있습니다.

간단한 선형 설계 덕분에 훈련 가능한 행렬을 배포할 때, 고정 가중치와 병합할 수 있으므로 추론 지연 시간이 발생하지 않습니다.

트랜스포머 층 사이에 추가되기 때문에, 연산에 필요한 메모리 사용량이 시퀀셜하게 증가되는 단점이 존재합니다.

LoRA는 모든 dense layer에 적용 가능하며 Neural Network는 많은 dense layer를 포함하고 있으며 Matrix multiplication을 수행합니다. 이 matrices의 weight는 일반적으로 full-rank입니다.

가중치를 업데이트하려면 gradient descent를 사용하여 얼마나 많은 가중치를 업데이트할 지를 결정하는 과정이 필요합니다.

이 중 nm linear로 된 행렬들에 대한 업데이트를 low-rank decomposition을 통해 업데이트 된 가중치로 근사하여 원래 weight에 더하는 개념입니다.

W0x에 대해 B,A의 행렬의 곱을 가중치로서 더해줬기 때문에 일종의 업데이트 해주는 형태가 되었다고 볼 수 있다. 

### 훈련 과정

필요한 패키지는 3개입니다. 
```
!pip install -q transformers 
!pip install -q peft
!pip install -q evaluate
```

단순히 테스트이기 때문에 imdb 데이터를 사용합니다
```
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

Peft를 이용해서 LoRA를 정의 해줍니다. 이경우 SEQ_CLS의 테스크를 수행합니다. 
```
from peft import LoraConfig, TaskType
from peft import get_peft_model
from transformers import BertForSequenceClassification

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=8,lora_alpha=16, lora_dropout=0.1
)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-cased', 
    num_labels=2
)

model = get_peft_model(model, lora_config)
```

모델의 정의가 끝나면 학습 파라미터를 준비 해줍니다. 이번에는 Trainer를 이용해서 쉽게 할 예정입니다. 

```
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer",
                                  evaluation_strategy="epoch",
                                  num_train_epochs=25,)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
```
![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/7354398e-ae8c-4242-8987-1773a3c94a9f)

### 제일 중요한 참고자료
https://medium.com/@karkar.nizar/fine-tuning-bert-for-text-classification-with-lora-f12af7fa95e4
