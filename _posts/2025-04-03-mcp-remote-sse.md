---
layout: post
title: MCP Remote SSE 사용하기
---
최근 AI 업계에서는 **모델 컨텍스트 프로토콜(MCP)** 이 주요 이슈로 떠오르고 있다. MCP는 애플리케이션이 대형 언어 모델(LLM)에 컨텍스트를 제공하는 방식을 표준화하는 **개방형 프로토콜**이며, 여러 기업 및 개발자들이 이를 활용한 프로젝트를 진행 중이다.

앤트로픽 페이지에서는 MCP를 다음과 같이 정의하고 있다:

> **"모델 컨텍스트 프로토콜 (MCP)는 애플리케이션이 LLM에 컨텍스트를 제공하는 방법을 표준화하는 개방형 프로토콜입니다."**

MCP는 다양한 방식으로 구현할 수 있으며, `uv`를 사용하여 **STDIO 방식**으로 운영할 수도 있지만, 확장성과 유지보수를 고려하면 **클라이언트-서버 구조**를 구성하는 것이 보다 효율적이다.

본 문서에서는 **MCP를 리모트 서버에서 실행하고, SSE(Server-Sent Events) 방식으로 데이터를 스트리밍하는 방법**을 설명한다. 또한, **클로드 데스크탑(Claude Desktop)에서 MCP 서버를 설정하는 방법**도 포함한다.




오늘은 그냥 아시아 나가 나쁘다. 힝

---

## 1. MCP SSE 서버 구성

MCP를 SSE 방식으로 실행하기 위해 `FastMCP` 라이브러리를 사용한다. 이를 통해 MCP 툴을 정의하고 서버를 실행할 수 있다.

### FastMCP를 이용한 SSE 서버 설정
```python
from typing import List
from mcp.server.fastmcp import FastMCP

# FastMCP 인스턴스 생성 및 'Weather' 도구 추가
mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """지정된 위치의 날씨 정보를 반환"""
    return "It's always sunny in New York"

# SSE 방식으로 MCP 서버 실행
if __name__ == "__main__":
    mcp.run(transport="sse")
```

위 코드를 `app.py`로 저장한 후, 다음 명령어를 실행하여 **SSE 서버를 시작**할 수 있다.

```sh
python app.py
```

이제 클라이언트는 **SSE를 통해 데이터 스트리밍이 가능**하다.

---

## 2. MCP Inspector를 활용한 테스트

서버가 정상적으로 동작하는지 확인하기 위해 **MCP Inspector**를 사용한다. MCP Inspector는 MCP 서버와 연결을 테스트하고, 등록된 MCP 툴이 정상적으로 작동하는지 검증하는 도구이다.

### MCP Inspector 사용 절차

1. MCP Inspector 실행
2. 연결(Connection) 버튼 클릭
3. 등록된 MCP 툴이 정상적으로 동작하는지 테스트

아래는 MCP Inspector의 예시 화면이다.

![MCP Inspector](https://github.com/user-attachments/assets/5bebba6d-bf90-4d38-b7a1-e868d5d5c00c)

MCP 툴이 정상적으로 작동하면, 리모트 서버 설정을 진행할 수 있다.

---

## 3. `mcp-remote` 설치

MCP를 리모트 서버에서 실행하려면 **`mcp-remote` 패키지를 설치**해야 한다.

설치는 [npm 패키지 페이지](https://www.npmjs.com/package/mcp-remote)에서 확인 가능하며, 다음 명령어를 실행하여 설치할 수 있다.

```sh
npm install -g mcp-remote
```

설치가 완료되면, 클로드 데스크탑에서 MCP 서버를 등록할 수 있다.

---

## 4. 클로드 데스크탑 설정

MCP를 클로드 데스크탑에서 사용하려면, **설정 파일(`claude_desktop_config.json`)을 수정**하여 리모트 MCP 서버를 등록해야 한다.

### 설정 파일 수정 절차

1. `claude_desktop_config.json` 파일의 위치는 다음과 같다. 존재하지 않는 경우 새로 생성해야 한다.

   ```
   C:\Users\<사용자이름>\AppData\Roaming\Claude\claude_desktop_config.json
   ```

2. 다음 JSON 설정을 추가하여 `mcp-remote`를 등록한다.

   ```json
   {
     "mcpServers": {
       "remote-example": {
         "command": "npx",
         "args": [
           "mcp-remote",
           "https://remote.mcp.server/sse"
         ]
       }
     }
   }
   ```

3. 설정을 저장한 후 **클로드 데스크탑을 재시작**한다.

이제 클로드 데스크탑에서 MCP 리모트 서버를 사용할 수 있다.

---

## 5. 윈도우 환경에서의 설정

리눅스 및 Mac 환경에서는 앞서 설명한 설정을 그대로 사용하면 되지만, **윈도우 환경에서는 설정이 다소 다르다.**

### 윈도우 환경에서 `claude_desktop_config.json` 수정

윈도우에서는 `cmd`를 이용하여 MCP 리모트 서버를 실행해야 하므로, 다음과 같이 설정해야 한다.

```json
{
  "mcpServers": {
    "remote-example": {
      "command": "cmd",
      "args": [
        "/c",
        "npx",
        "https://remote.mcp.server/sse"
      ]
    }
  }
}
```

이제 윈도우에서도 MCP 리모트 서버를 등록하고 사용할 수 있다.

---

## 결론

본 문서에서는 **MCP를 SSE 방식으로 실행하는 방법**, **리모트 MCP 서버 설정 방법**, 그리고 **클로드 데스크탑에서 MCP 서버를 등록하는 방법**을 설명하였다.

MCP Inspector를 활용하여 정상 작동을 확인한 후, 필요에 따라 다양한 MCP 툴을 추가하여 확장할 수 있다.�️

이제 여러분도 **MCP 기반의 AI 애플리케이션을 손쉽게 구축**할 수 있답니다! 😊

✨ **MCP Inspector를 활용해서 정상 작동을 확인**하고, 다양한 MCP 툴을 추가해서 더 멋진 기능을 만들어 보세요! 🚀
