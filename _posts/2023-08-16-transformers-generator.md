---
layout: post
title: Transformers를 generator로 만드는 방법
---

transformers를 쓰면 주로 model.generate()를 쓰게 되는데 이것을 쓰면 모든 토큰이 생성이 끝날때 까지 아무 것도 확인 할 수 없다. streamer 기능을 사용하면 바로바로 생성되는 토큰을 확인 할 수 있고 generator로 만들 수 있는데 이번에는 특히 TextIteratorStreamer를 이용해서 구현할 것이다. 


아래 코드는 GPT-2로 간단히 generator로 구현하는 코드이면서 SSE를 통해 서버에 토큰별로 보내는 코드이다.

```
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


# transformers model load
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")


def run_generation(user_text):
    model_inputs = tokenizer([user_text], return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(model_inputs,streamer=streamer)

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    model_output = ""
    for new_text in streamer:
        model_output += new_text
        yield f"data: {new_text}\n\n"
        
    return model_output
```
