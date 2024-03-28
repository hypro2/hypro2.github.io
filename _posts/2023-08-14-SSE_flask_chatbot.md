---
layout: post
title: SSE(서버-사이드 이벤트)를 이용한 Flask 스트리밍 실시간 챗봇을 위한 연습
---

연습용 자료 첫번째는 오늘 하던 것 간단히 구현해서 업무에 적용시킬 프로젝트에 넣기위해 테스트 겸 만든 코드 SSE(서버-사이드 이벤트)는 클라이언트와 서버 간의 실시간 통신을 위한 웹 기술 중 하나다.


SSE는 단방향 통신을 제공하며, 서버에서 클라이언트로 실시간 업데이트를 보낼 수 있는 간단한 방법을 제공함.

주로 웹 애플리케이션에서 서버로부터 실시간 이벤트나 업데이트를 받아와서 사용자에게 동적인 콘텐츠를 표시하는 데 사용함.



```
from threading import Thread

import openai
from flask import Flask, Response, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


app = Flask(__name__)

### config model load
openai.api_key = config['OPENAI']['API']

# transformers model load
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
```

### OpenAI를 Generator로 yield 하는 작업

```
# OpenAI API
def run_generation(input_text):
    generator = openai.ChatCompletion.create(model='gpt-3.5-turbo',
                                             messages=[{'role': 'user', 'content': input_text}],
                                             temperature=0,
                                             stream=True)
    for event in generator:
        try:
            event_text = event['choices'][0]['delta']["content"]
            yield f"data: {event_text}\n\n"
        except:
            pass
```

### Flask 

```
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream/<input_text>')
def stream(input_text):
    return Response(run_generation(input_text), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
```

### SSE용 HTML 간단한 예제

```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-2 Streaming</title>
</head>
<body>
    <h1>Streaming Example</h1>

    <form>
        <label for="inputText">Enter text:</label>
        <input type="text" id="inputText" name="inputText">
        <button type="button" id="startStream">Start Streaming</button>
    </form>

    <div id="output">
        <p>Generated text:</p>
        <pre id="generatedText"></pre>
    </div>

    <script>
        const startStreamButton = document.getElementById('startStream');
        let eventSource = null;

        startStreamButton.addEventListener('click', () => {
            const inputTextElement = document.getElementById('inputText');
            const inputText = inputTextElement.value;
            if (inputText) {
                startStream(inputText);
            }
        });

        function startStream(inputText) {
            const generatedTextElement = document.getElementById('generatedText');
            eventSource = new EventSource(`/stream/${inputText}`);

            eventSource.onmessage = function(event) {
                generatedTextElement.textContent += event.data;
            };

            eventSource.onerror = function(event) {
                eventSource.close();
            };
        };
    </script>
</body>
</html>
```
