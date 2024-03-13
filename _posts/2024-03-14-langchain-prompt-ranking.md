---
layout: post
title: 깃허브 프로젝트 Langchain Prompt Ranking 만들었습니다.
---
이번에는 개인 프로젝트를 만들어 보았습니다. 랭체인을 통해서 LLM의 프롬프트를 평가하는 프로젝트입니다. 이것은  gpt-prompt-engineer의 클론 프로젝트로 작동합니다. 랭체인에서 적용하기 위해서 랭체인으로 바꾸어서 진행하고 있습니다. 이것으로 다른 LLM모델들의 프롬프트도 GPT4를 통해서 ELO 레이팅으로 평가하므로서 확장성을 넓혔습니다. 

[github](https://github.com/hypro2/Langchain_Ranking)



# 🌟 Langchain Ranking 프로젝트 🌟

Langchain 프롬프트 랭킹 프로젝트에 오신 것을 환영합니다! 🎉 이 프로젝트는 gpt-prompt-engineer의 혁신적인 개념을에 감명받아서 랭체인을 통해서 구현한 랭체인을 사용하여 모델을 평가하는 것에 대해 모두 이야기합니다. 🚀

## 🤖 What is gpt-prompt-engineer?

RESPCET : [https://github.com/mshumer/gpt-prompt-engineer](https://github.com/mshumer/gpt-prompt-engineer)

'gpt-prompt-engineer'는 GPT-4 및 GPT-3.5-Turbo와 같은 대규모 언어 모델의 성능을 향상시키는 탁월한 도구입니다. 마치 모델을 최적화하는 마법의 지팡이처럼 작동합니다! ✨

저는 이 프로젝트에 깊은 인상을 받아 랭체인과 같은 일을 할 수 있는 프로젝트를 만들기로 결심했습니다.

최근에 gpt-prompt-engineer GitHub 리포지토리를 발견했고, 그 프로젝트가 저에게 큰 영감을 주었습니다. 특히 gpt-prompt-engineer에서 제공하는 프롬프트를 평가하는접근 방식이 매우 인상적었습니다. 그래서 새로운 개인 프로젝트로 시작하고자 하는데, 마음에 안드시거나 불편하시면 언제든지 연락 주시기 바랍니다. 다시 한번, 감사드립니다!

## 🛠 How does it work?

'Langchain Ranking'는 특정 사용 사례를 기반으로 다양한 프롬프트를 생성하고 엄격하게 테스트한 후 Elo 등급 시스템을 사용하여 순위를 매깁니다. 💡

## 🏆 Why did you choose Langchain?

Langchain을 사용하면 언어 학습 모델 (LLM) 체인을 구축할 수 있으며, 이는 프로젝트의 기반을 형성합니다. Langchain을 사용하여 모델을 심사하는 데 집중할 수 있습니다. 💪

## 💡 Ranking process

1.  **프롬프트 생성:** 🤔 다양한 시나리오에 맞는 다양한 프롬프트를 생성합니다.
2.  **테스트 및 비교:** 🧪 각 프롬프트는 엄격한 테스트를 거쳐 성능을 비교합니다.
3.  **Elo 등급 시스템:** 📈 프롬프트의 효과에 따라 Elo 등급 시스템을 사용하여 프롬프트를 순위로 매깁니다.

# 작동 방법

```
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

from gpt_ranking import GPTRanking

# 사용 방법
# custom llm chain 정의
prompt = ChatPromptTemplate.from_messages([("system", "Answer in {A}"), ("user", "{input}")])
result_model = ChatOpenAI(model_name="gpt-3.5-turbo")
custom_chain = prompt | result_model | StrOutputParser()

# 하고 싶은 작업에 대한 설명을 입력합니다.
description1 = "Create a landing page headline."

# 결과 값으로 받고 싶은 예시를 몇 가지 입력합니다.(적당한 개수가 있어야 합니다.)
test_cases1 = [
    "Promote your innovative new fitness app smartly",
    "Why a vegan diet is good for your health",
    "Introducing a new online course on digital marketing.",
]

# 직접 작성한 프롬프트의 성능도 비교하려면 여기에 입력합니다. 입력하지 않아도 됩니다.
user_prompt_list1 = ["example1"]

gpt_rank = GPTRanking(prompt_chain=custom_chain,
                      description=description1,
                      test_cases=test_cases1,
                      user_prompt_list=user_prompt_list1,
                      ranking_model_name="gpt-3.5-turbo",
                      use_rar=False,
                      use_csv=False,
                      n=3,
                      )

gpt_rank.generate_and_compare_prompts(A="Korean")
```

# 실행 결과

```
['Discover the top 10 secrets to unlocking your full potential.', "Craft an enticing headline that captivates your audience's attention.", 'Craft compelling headlines that entice and engage your target audience.']

11%|███▊                              | 2/18 [00:09<01:15,  4.73s/it]
 Winner: Discover the top 10 secrets to unlocking your full potential.
17%|█████▋                            | 3/18 [00:11<00:52,  3.49s/it]
 Draw
22%|███████▌                          | 4/18 [00:14<00:48,  3.44s/it]
Winner: Discover the top 10 secrets to unlocking your full potential.
28%|█████████▍                        | 5/18 [00:17<00:43,  3.35s/it]
Winner: example1
 33%|███████████▎                      | 6/18 [00:20<00:38,  3.19s/it
Winner: Craft an enticing headline that captivates your audience's attention.
39%|█████████████▏                    | 7/18 [00:22<00:31,  2.88s/it]
Winner: Craft an enticing headline that captivates your audience's attention.
 44%|███████████████                   | 8/18 [00:27<00:34,  3.44s/it]
Winner: Craft compelling headlines that entice and engage your target audience.
Winner: Craft compelling headlines that entice and engage your target audience.
 56%|██████████████████▎              | 10/18 [00:31<00:21,  2.66s/it]
 Winner: Craft compelling headlines that entice and engage your target audience.
 61%|████████████████████▏            | 11/18 [00:33<00:16,  2.42s/it]
 Draw
 67%|██████████████████████           | 12/18 [00:35<00:13,  2.22s/it]
 Winner: Discover the top 10 secrets to unlocking your full potential.
Draw
 78%|█████████████████████████▋       | 14/18 [00:39<00:08,  2.20s/it]
 Draw
 83%|███████████████████████████▌     | 15/18 [00:41<00:06,  2.22s/it]
 Draw
Winner: Discover the top 10 secrets to unlocking your full potential.
 94%|███████████████████████████████▏ | 17/18 [00:45<00:02,  2.16s/it]
 Draw
100%|█████████████████████████████████| 18/18 [00:47<00:00,  2.13s/it]
Draw
100%|█████████████████████████████████| 18/18 [00:49<00:00,  2.76s/it]
Draw


{
'example1': 1118.6540526700558,

'Discover the top 10 secrets to unlocking your full potential.': 1257.36515660952,

"Craft an enticing headline that captivates your audience's attention.": 1203.8262872651787,

'Craft compelling headlines that entice and engage your target audience.': 1220.1545034552455
}
```

# GPTRanking Args

```
GPTRanking 클래스의 생성자입니다.
prompt_chain에 의해서 생성되는 결과는 문자열로 반환이 되어야 됩니다.

:param prompt_chain: 프롬프트 체인을 나타내는 LangChain 객체입니다.
:param description: 프롬프트 생성을 위한 작업 또는 문맥을 설명하는 문자열입니다.
:param test_cases: 프롬프트를 생성하는 데 사용되는 다양한 시나리오나 입력을 나타내는 문자열 목록입니다.
:param user_prompt_list: (선택 사항) 비교에 포함할 사용자 정의 프롬프트의 목록입니다.
:param use_rar: (선택 사항) 프롬프트 생성에 RAR 기법을 사용할지 여부를 나타내는 부울 값입니다. 기본값은 False입니다.
:param use_csv: (선택 사항) 프롬프트 평가를 CSV 파일에 저장할지 여부를 나타내는 부울 값입니다. 기본값은 False입니다.
:param ranking_model_name: (선택 사항) 기본값은 "gpt-3.5-turbo" 입니다.
:param judge_prompt: (선택 사항) 기본값은 None 입니다.
:param n: (선택 사항) 생성할 프롬프트의 수를 지정하는 정수입니다. 기본값은 5입니다.
```
