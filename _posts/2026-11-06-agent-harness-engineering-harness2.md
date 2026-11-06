---
layout: post
title: "에이전트 실패의 90%는 모델 지능이 아니라 '하네스(Harness)' 문제다: Google Research Harness²와 에이전트 엔지니어링"
tags: [AI, LLM, Agent, Harness, GoogleResearch, Harness2, ClaudeCode, PromptEngineering, SoftwareEngineering]
---

## 에이전트 실패의 90%는 모델 지능이 아니라 '하네스(Harness)' 문제다: Google Research Harness²와 에이전트 엔지니어링

최근 Claude 3.7 Sonnet, OpenAI o3, Gemini 2.5 등 대규모 언어 모델(LLM)의 추론 능력과 코딩 벤치마크 점수는 인간 시니어 엔지니어를 방불케 할 정도로 치솟았습니다. 

하지만 실제로 터미널이나 IDE에서 자율 코딩 에이전트(Claude Code, Codex CLI, Cursor 등)에게 복잡한 실무 과제를 맡겨보면, 우리는 여전히 다음과 같은 뼈아픈 상황을 마주합니다:

- 테스트 코드가 실패하거나 런타임 에러가 명백히 남아있는데도, 혼자서 당당하게 **"모든 기능이 완벽하게 구현되었습니다!"**라며 작업을 마감해 버림 (**조기 승리 선언 / Illusion of Victory**).
- 요청하지도 않은 인접 파일들을 마구잡이로 수정하거나 불필요한 리팩토링을 감행해 기존 코드를 망가뜨림 (**스코프 이탈 / Scope Creep**).
- 수십 번의 도구 호출을 거치며 컨텍스트가 토큰 한도에 도달해, 자신이 원래 무엇을 하려 했는지조차 잊어버림 (**상태 유실 / State Amnesia**).

많은 개발자들은 이 현상을 보고 *"아직 모델의 지능(IQ)이 부족하다"*, *"더 똑똑한 차세대 파운데이션 모델이 나와야 해결된다"*라고 생각합니다.

하지만 Google Research가 2026년 10월 발표한 연구 **[Harness²: Recursive Agent Harnessing for an Open World (arXiv:2610.14224)](https://arxiv.org/abs/2610.14224)**와 최근의 에이전트 하네스 엔지니어링(Agent Harness Engineering) 진영은 전혀 다른 진단을 내립니다:

> **"에이전트가 실패하는 진짜 원인은 모델의 지능 부족이 아니라, 모델을 둘러싼 '하네스(Harness)'의 결핍 때문이다."**

모델의 가중치(Weights)를 단 1밀리그램도 건드리지 않고, 오직 모델 주변의 **하네스(Harness)**를 엔지니어링하고 재귀적으로 최적화하는 것만으로 복합 벤치마크 점수를 대폭 끌어올린 Google Research의 연구와 최신 에이전트 하네스 설계 원리를 깊이 있게 파헤쳐 보겠습니다.

---

### 1. 하네스(Harness)란 무엇인가?

'하네스(Harness)'의 원래 뜻은 말이나 썰매견을 마차에 연결하는 마구(馬具), 혹은 암벽 등반가의 몸을 안전 로프에 묶어주는 안전벨트입니다. 말의 힘이 아무리 천하장사라 해도 마구가 엉성하면 짐수레는 엉뚱한 도랑으로 처박히고 맙니다.

AI 시스템에서 **하네스(Agent Harness)**란 **"파운데이션 모델을 감싸고 있는 실행 런타임 환경, 상태 관리 체계, 도구 인터페이스, 그리고 비타협적 외부 제어 장치 일체"**를 의미합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENT HARNESS (하네스 계층)               │
│                                                             │
│  [Context Construction]                                     │
│   - System Prompt, Guardrails, Memory/State, Skills         │
│                                                             │
│  [Tool & Execution Interface]                               │
│   - Tool Guidance, Scripts, Dynamic Tool Selection (SLM)    │
│                                                             │
│  [Verification & Anti-Premature Victory]                    │
│   - Objective Test Runner, Regression Gate, Linter          │
│                                                             │
│                     ┌────────────────┐                      │
│                     │   LLM Engine   │                      │
│                     │  (Frozen Model)│                      │
│                     └────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

에이전트는 LLM 단독으로 동작하는 생명체가 아닙니다. 모델은 확률적으로 다음 토큰을 예측하는 순수 함수일 뿐이며, **에이전트의 완성도와 안정성을 결정짓는 90%의 지분은 모델 밖의 하네스**에 있습니다.

---

### 2. 에이전트 실패의 주범: '조기 승리 선언 (Illusion of Victory)'

하네스가 부실한 에이전트 시스템에서 가장 빈번하게 발생하는 재앙이 바로 **조기 승리 선언(Premature Victory Declaration)**입니다.

에이전트에게 구현 태스크를 주면, 모델은 자신의 생성물을 스스로 검증(Self-Evaluation)하는 환각에 빠집니다:

1. 코드를 작성함 (내부에 문법 에러나 논리 버그 존재).
2. 스스로 생각함: *"나는 프롬프트가 요구한 클래스와 메서드를 모두 정의했다. 따라서 작업은 성공적이다."*
3. 사용자에게 출력함: *"요청하신 기능 구현을 100% 완료했습니다."*

```
[자가 채점 환각 (Self-Evaluation Hallucination)]
에이전트 생성 ───> 에이전트 자가 판단 ───> "완벽합니다!" (조기 승리 선언)
                      ▲
             (자기 편향 & 환각 발생)
```

이 환각을 깨부수는 유일한 방법은 **인간이나 독립된 외부 시스템이 정의한 '비타협적 검증 하네스(Hands-on Verification Harness)'**입니다.

에이전트가 완성을 선언하기 전에, 하네스가 강제로 인터셉트하여:
1. 실제 린터(Linter)와 타입 체커(`mypy`, `tsc`)를 통과했는가?
2. 사전에 정의된 단위/통합 테스트 스위트를 100% 패스했는가?
3. 기존 기능의 회귀(Regression)가 없는가?

이 세 가지 외부 물리적 증거(Empirical Evidence)가 확인되지 않으면, 하네스는 에이전트의 종료 호출을 가로막고 에러 로그와 함께 제어권을 강제로 다시 모델에게 반환해야 합니다.

---

### 3. Google Research의 해법: Harness² (arXiv:2610.14224)

그렇다면 이 하네스를 어떻게 구성하고 최적화해야 할까요?

수작업으로 프롬프트를 깎고 도구 인터페이스를 일일이 수정하는 방식은 도메인이 바뀌거나 새로운 모델이 나올 때마다 막대한 인력 낭비를 초래합니다. 

Google Research는 [google-research/harness2](https://github.com/google-research/harness2) 오픈소스를 통해, **모델 파인튜닝 없이 에이전트가 자체 실행 결과의 행동 대조(Behavioral Contrasts)를 분석하여 하네스 구조를 스스로 교정·발전시키는 2계층 재귀 최적화 프레임워크인 'Harness²'**를 제안했습니다.

```mermaid
flowchart TD
    subgraph TaskLevel [1계층: Task-level Recursion (Propose-Probe-Compose)]
        E0["초기 하네스 실행"] --> P1["Harness Editor: 수정 가설 제안"]
        P1 --> P2["Candidate Probes: 병렬/순차 실행 검증"]
        P2 --> P3["Harness Composer: 대조 합성 및 최적 하네스 도출"]
    end

    subgraph HarnessingLevel [2계층: Harnessing-level Recursion (Persistent Reflection)]
        R1["Experience Reflector: 실행 이력 반추"]
        R2["Improvement Harness 원장 누적 갱신"]
        P3 --> R1 --> R2
        R2 -.->|다음 태스크의 초기 하네스 가이드| TaskLevel
    end
```

Harness²는 하네스를 구성하는 핵심 요소를 **8대 수정 가능 컴포넌트(8 Editable Components)**로 모듈화했습니다:

| 범주 | 컴포넌트 | 설명 |
| :--- | :--- | :--- |
| **Context Construction** | **System Prompt** | 에이전트의 역할, 성격, 전반적 목표 정의 |
| | **Guardrails** | 스코프 이탈 방지 및 보안/품질 불변 조건 |
| | **Memory** | 장기 실행 간 보존되는 진행 상태 및 히스토리 |
| | **Skills** | 도메인별 작업 절차 및 가이드라인 (SOP) |
| **Tool Interface** | **Tool Guidance** | 도구의 스키마 명세 및 올바른 호출 예시 |
| | **Scripts** | 복잡한 로직을 대신 수행하는 결정론적 헬퍼 스크립트 |
| **Delegation & Middleware** | **Subagents** | 특정 서브태스크를 전담하는 위임 에이전트 구조 |
| | **Plugins** | 런타임에 주입되는 필터 및 미들웨어 |

#### Propose-Probe-Compose (PPC) 루프
Harness²의 1계층(Task-level)은 정답 레이블(Ground-truth)이 없는 오픈월드 태스크에서도 다음과 같이 작동합니다:

1. **Propose (제안)**: 현재 하네스에서 실패하거나 아쉬웠던 점을 파악하고, 하네스 컴포넌트를 변형한 여러 수정 가설을 생성.
2. **Probe (탐침)**: 변형된 하네스 후보군들을 병렬로 실행하여 실행 궤적(Trajectory)과 중간 부산물을 수집.
3. **Compose (합성)**: 서로 다른 프로브들의 실행 결과를 대조(Contrasting)하여, 가장 오류가 적고 강건한 하네스 파라미터를 최종 병합.

그 결과, Gemini 3.5 Flash와 GLM-5를 대상으로 복합 실무 벤치마크인 **WorkBuddy-Bench에서 베이스라인 대비 +11.5점**, **JobBench에서 +4.5점**, **LAB(Language Agent Benchmark)에서 부분 통과율 +11.8%p**라는 경이로운 성능 향상을 달성했습니다. 가중치 학습 없이 순수 하네스 튜닝만으로 일궈낸 결과입니다.

```bash
# Harness² 환경 설치 및 JobBench 병렬 재귀 최적화 실행
uv pip install -e ".[render]"
harness2-setup jb
harness2 --bench jb --k 1 --mode parallel
```

---

### 4. 실무 에이전트를 위한 4대 하네스 엔지니어링 패턴

Google의 Harness² 연구와 최신 오픈소스 실무 사례([MemoHarness, arXiv:2607.14159](https://arxiv.org/abs/2607.14159), `learn-harness-engineering`)를 종합해 볼 때, 프로덕션 에이전트를 구축하는 엔지니어가 반드시 갖춰야 할 4가지 실전 하네스 패턴은 다음과 같습니다.

#### 패턴 1: 상태의 외부 영속화 (State Externalization)
에이전트의 단기 기억(Context Window)을 믿지 마세요. 컨텍스트가 길어지면 Attention Sink 현상으로 초반 지시사항이 증발합니다.
- `feature_list.json`: 구현해야 할 기능들의 완료 여부(`pending`, `in_progress`, `completed`)를 외부 파일로 영속화.
- `claude-progress.md` / `task-progress.md`: 에이전트가 매 턴마다 현재 위치와 다음 할 일을 기록하는 영속 일지. 
에이전트가 크래시되거나 토큰 한계로 재시작하더라도, 하네스가 이 외부 파일을 읽어 컨텍스트를 복원해야 합니다.

#### 패턴 2: 동적 도구 선택 미들웨어 (Dynamic Tool Selection)
에이전트에게 50개의 도구 스키마를 한꺼번에 프롬프트에 주입하면(Context Bloat), 모델의 도구 선택 정확도는 곤두박질치고 토큰 비용은 수 배로 치솟습니다 (Sebastian Raschka의 실측에 따르면, Claude Code 계열 하네스는 Codex CLI 대비 동일 작업에서 약 2배의 토큰을 소비함).
- 가벼운 경량 보조 모델(SLM)이나 시맨틱 라우터(`LLMToolSelectorMiddleware`)를 하네스 앞단에 배치하여, 현재 단계에 필요한 상위 3~5개 도구만 동적으로 주입합니다.

#### 패턴 3: 비타협적 외부 게이트 (Strict External Gates)
에이전트의 "작업 완료" 도구 호출을 무조건 신뢰하지 마세요.
- 완료 도구(`finish_task`)가 호출되면 하네스가 즉시 터미널에서 `pytest`나 빌드 스크립트를 백그라운드로 실행합니다.
- 실패 시 성공 응답 대신 `Exit Code 1: 3 tests failed`를 에러 메시지로 반환하여 수정을 강제합니다.

#### 패턴 4: 2계층 경험 은행 (Dual-layer Experience Bank)
MemoHarness가 제시한 방식으로, 에이전트의 실패와 성공 경험을 휘발시키지 않고:
- **1계층 (Per-case Diagnoses)**: 개별 태스크에서 발생한 구체적 에러와 해결책.
- **2계층 (Distilled Global Patterns)**: 반복되는 실수로부터 증류된 전역 원칙.
이를 누적 기록하여 새로운 태스크 실행 시 시스템 프롬프트의 룰셋으로 자동 주입합니다.

---

### 5. 마치며: 모델 중심에서 '하네스 중심'으로

우리는 오랫동안 더 거대한 파라미터, 더 뛰어난 벤치마크 점수를 가진 "초지능 파운데이션 모델"만을 기다려왔습니다.

하지만 자율 에이전트의 시대에 들어서면서 승부처는 완전히 바뀌었습니다. **이미 출시된 프론티어 모델들의 지능은 충분히 훌륭합니다. 문제는 그 지능을 현실 세계의 코드베이스와 소프트웨어 파이프라인에 안전하게 연결해 줄 하네스가 부실하다는 점입니다.**

더 똑똑한 모델을 기다리기 전에, 여러분의 에이전트 시스템에 이렇게 질문해 보아야 합니다:

- *"우리 에이전트는 거짓 완료를 선언하지 못하도록 막는 객관적 하네스를 갖추고 있는가?"*
- *"컨텍스트 오염을 막기 위한 동적 미들웨어가 작동하고 있는가?"*
- *"실패로부터 스스로 학습하는 경험 원장이 영속화되어 있는가?"*

에이전트 엔지니어링의 본질은 프롬프트 몇 줄을 수정하는 것이 아니라, **강건하고 비타협적인 하네스를 구축하는 소프트웨어 엔지니어링**입니다.
