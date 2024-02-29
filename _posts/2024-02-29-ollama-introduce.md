---
layout: post
title: Ollama windows로 langchain함께 쉽게 Local LLM 실행하기
---

Ollama는 로컬 LLM을 실행하기 복잡한 과정을 쉽게 줄여주는 프로그램입니다. 이제 Ollama가 Windows에서 미리보기로 제공되며, 이를 통해 최신 모델을 간편하게 테스트할 수 있습니다. Windows용 Ollama는 내장 GPU 가속화, 전체 모델 라이브러리 액세스, 그리고 OpenAI 호환성을 갖추고 있습니다. 이제 사용자들은 최신 모델을 손쉽게 활용할 수 있습니다.



## 설치
Ollama를 사용하기 위해서는 윈도우의 wsl2 설치가 되어있어야됩니다.

간단히 이전에 포스트한 wsl docker 문서를 참조해서 wsl을 설치 하실수 있으실 겁니다. 동시에 docker도 설치해두면 편하겠죠.

[이전 포스트](https://hypro2.github.io/ubuntu-docker/)

wsl2가 사용할 수 있게된다면 Ollama 사이트로 들어가서 Windows용 Ollama를 설치합니다.

[https://ollama.com/download](https://ollama.com/download)

![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/61c67298-665e-492a-9875-efc01d311c01)


빠른 설치과 완료가 완료되면 Ollama가 설치가 완료되었습니다.

Ollama는 windows 백그라운드 icons 속에서 실행되고 있는지 확인 할 수 있습니다.

![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/63a1fea6-ecdf-42e6-b065-2c9e19517c06)

## 실행

명령 프롬프트 창에서 Ollama와 같은 명령어를 입력해서 실행하면 됩니다.

Ollama를 실행하기 위해서는 'ollama serve'를 먼저 실행해줍니다. 'ollama run llama2'와 같이 원하는 모델을 'ollama run'뒤에 붙혀서 실행합니다. 지원하는 모델들은 여기서 확인 할 수 있습니다. '[https://ollama.com/library](https://ollama.com/library)'

저는 이번에 gemma:2b를 사용해보려고 합니다. 'ollama run gemma:2b'를 실행해주면 Ollama가 알아서 다운받아서 gemma를 실행해줍니다.

![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/83ed2715-8e51-454e-958b-abb0469f9c84)


이렇게 간단히 사용하는 방법도 있고 랭체인과 연동시켜서 개발에 사용하는 방법도 존재합니다.

Ollama를 실행하게된다면 model이 자동으로 http://localhost:11434에 할당 됩니다. 저희는 이것을 사용하면됩니다. 

랭체인에서 지원하는 Ollama를 임포트해서 사용 하신다면 굳이 저 포트를 사용하실 필요 없이 자동으로 연동됩니다. 

## 코드

```
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")

llm.invoke("Tell me a joke")
```
![image](https://github.com/hypro2/hypro2.github.io/assets/84513149/14cbaed7-ca3f-4930-b70b-b1874cd8ab79)



종료를 하시고 싶다면 명령어도 없이 편하게 , icon으로가서 quit ollama를 눌러주시면 됩니다. 

**자료 출처:**

[https://ollama.com/](https://ollama.com/)

[https://python.langchain.com/docs/integrations/chat/ollama](https://python.langchain.com/docs/integrations/chat/ollama)
