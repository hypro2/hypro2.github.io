---
layout: post
title: MCP Langgraph AI 에이전트 아키텍처와 LangChain MCP Adapters의 FastMCP SSE 예제
---
최근 AI 에이전트를 구축할 때, 다양한 외부 도구를 손쉽게 연결하고 확장할 수 있는 **MCP (Modular Command Protocol)** 아키텍처가 주목받고 있습니다.

🧩 MCP란?

**MCP(Modular Command Protocol)**는 LLM 기반 AI 에이전트가 다양한 외부 도구를 유연하게 호출할 수 있도록 설계된 **범용 인터페이스**입니다.

✔️ MCP의 주요 특징

-   **유연한 통신**: 다양한 클라이언트/서버 구성에서 사용 가능
-   **빠른 도구 연동**: 도구를 데코레이터 한 줄로 노출 가능
-   **빠른 개발 및 프로토타이핑**에 최적
-   **낮은 진입 장벽**: LangChain Adapter + FastMCP로 바로 시작 가능




---

## 🧱 아키텍처 구성 개요

![image](https://github.com/user-attachments/assets/a5eed240-9134-4ee7-88c9-40d0b8401d6f)


MCP 시스템은 크게 **3가지 요소**로 구성됩니다.

1.  **MCP Host**
    -   유저가 사용하는 인터페이스 (예: ChatGPT, LangGraph, 웹 UI 등)
    -   MCP 도구를 호출하는 클라이언트이자 실행 환경
2.  **MCP Client**
    -   도구 목록을 MCP 서버에 요청하고, 메시지를 전송하는 중계자
    -   LangChain MCP Adapter로 쉽게 구현 가능
3.  **MCP Server**
    -   실제 도구를 실행하는 서버
    -   도구들은 함수 기반으로 정의되어 있으며, FastMCP로 간단하게 구성 가능

---

## 🌐 SSE 방식이란?

우리는 **SSE(Server-Sent Events)** 방식으로 MCP 서버를 구동합니다. 이 방식은 **HTTP 기반의 실시간 이벤트 스트리밍**을 지원하며, 다음과 같은 장점이 있습니다:

-   **실시간 업데이트**: 서버에서 클라이언트로 실시간 데이터 푸시
-   **HTTP 기반**: 방화벽/프록시 우회가 쉬워 운영 환경에 적합
-   **경량 통신**: WebSocket보다 단순하고 빠르게 구성 가능

---

## 🚀 FastMCP로 MCP 서버 만들기

 [MCP SSE Remote사용하기](https://hyeong9647.tistory.com/entry/MCP-Remote-SSE-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0)

이전 글쓰기를 참조해서 MCP를 Remote SSE환경에서 띄어서 사용해보자 

```
from typing import List
from mcp.server.fastmcp import FastMCP

# FastMCP 인스턴스 생성 및 'Weather' 도구 추가
mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """지정된 위치의 날씨 정보를 반환"""
    return f"It's always sunny in {location}"

# SSE 방식으로 MCP 서버 실행
if __name__ == "__main__":
    mcp.run(transport="sse")
```

위 코드를 app.py로 저장한 후, 다음 명령어를 실행하여 SSE 서버를 시작할 수 있다.

## 🎯 랭체인 코드 실행

간단한 create\_react\_agent를 만들어서 똑같은 동작을 구현하려고 한다.

MCP는 전부 비동기를 사용해야되서 디버그가 쉽지 않지만 나쁘지 않게 해볼만하다.

```
import asyncio
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from langchain_mcp_adapters.client import SingleServerMCPClient


async def main():
    # OpenAI 모델 초기화
    model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)

    # FastMCP에서 제공하는 Weather 도구에 연결 (SSE)
    async with SingleServerMCPClient(
        url="http://localhost:8000/sse",  # FastMCP 서버 포트와 일치해야 함
        transport="sse"
    ) as client:
        tools = client.get_tools()
        print("Available tools:", tools)

        # MCP 도구로 React Agent 생성
        agent = create_react_agent(model, tools)

        # 메시지를 통한 도구 호출
        response = await agent.ainvoke({"messages": "서울의 날씨를 알려줘"})
        print("Agent response:", response)

# 비동기 실행
asyncio.run(main())
```

## 🌐 랭그래프 코드 실행

연결하는 방법은 어렵지 않다 기존의 코드에서 tool만 client에서 받아와서 사용하면 똑같이 기존코드를 사용할수 있다. 

```
import asyncio
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import SingleServerMCPClient

# ✅ 상태 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]

memory = MemorySaver()


# ✅ FastMCP용 클라이언트 생성 함수
async def create_client():
    return SingleServerMCPClient(
        url="http://localhost:8000/sse",  # FastMCP 서버 포트에 맞춰 설정
        transport="sse"
    )


# ✅ MCP Graph 생성 함수
def mcp_graph(client):
    tools = client.get_tools()
    print("🔧 MCP Tools:", tools)

    # LLM 설정
    api_key = openai_api_key()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    llm_with_tools = llm.bind_tools(tools)

    # 챗봇 노드 정의
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # 상태 그래프 정의
    graph_builder = StateGraph(State)

    # 노드 구성
    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")

    # 시작과 종료 정의
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # 그래프 컴파일
    return graph_builder.compile(checkpointer=memory)


# ✅ 메인 함수
async def main():
    config = RunnableConfig(
        recursion_limit=10,
        configurable={"thread_id": "1"},
        tags=["my-tag"]
    )

    async with await create_client() as client:
        agent = mcp_graph(client)
        response = await agent.ainvoke(
            {"messages": "서울 날씨 알려줘"},  # 메시지를 MCP 도구에 맞게 조정
            config=config
        )
        print("📨 Agent Response:", response)


# ✅ 실행
asyncio.run(main())
```

아시아 나가 나쁘다
