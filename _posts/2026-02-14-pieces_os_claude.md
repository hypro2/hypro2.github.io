---
layout: post
title:  Pieces OS Claude Desktop과 연동하여 개발 워크플로우 기억 시스템 구축하기
---

Pieces OS: Claude Desktop과 연동하여 개발 워크플로우 기억 시스템 구축하기

Pieces OS 개요
Pieces OS는 개발자의 작업 흐름을 OS 레벨에서 자동으로 캡처하고 저장하는 시스템입니다. 일반적인 AI 어시스턴트가 대화 세션이 종료되면 컨텍스트를 잃어버리는 것과 달리, Pieces OS는 최대 9개월간 작업 히스토리를 로컬에 보관합니다.

기본적인 동작 방식은 다음과 같습니다. IDE에서 작성한 코드, 브라우저에서 참고한 문서, 복사한 스니펫, 팀과의 대화 등 개발 과정에서 발생하는 모든 활동을 백그라운드에서 수집합니다. 수집된 데이터는 전부 로컬 디바이스에 저장되며, 외부로 전송되지 않습니다.




## LTM-2.7 엔진의 작동 원리
Long-Term Memory 엔진은 Pieces OS의 핵심 구성요소입니다. 이 엔진이 수행하는 주요 작업은 세 가지로 요약됩니다.

첫째, 20분 간격으로 워크플로우를 자동 요약합니다. 이 과정을 **Roll-Up**이라고 부르며, 각 Roll-Up에는 해당 시간대의 주요 작업, 결정사항, 참고 자료가 구조화되어 저장됩니다.

둘째, 시간 기반 컨텍스트 검색을 지원합니다. "어제 디버깅한 오류"나 "지난주 화요일에 참고한 Firebase 문서" 같은 자연어 쿼리를 처리할 수 있습니다. 이는 단순한 키워드 검색이 아니라 시간적 맥락을 이해하는 검색입니다.

셋째, 프로젝트 맥락을 장기간 보존합니다. 몇 주 후에 프로젝트를 다시 열어도 당시의 코드, 노트, 대화 내용이 모두 연결된 상태로 복원됩니다.

처리 과정의 90% 이상이 온디바이스에서 진행됩니다. 클라우드 LLM이 필요한 일부 고급 기능만 선택적으로 외부 모델을 사용하며, 이는 사용자 설정으로 제어 가능합니다.

## Workstream Activity 구조
Workstream Activity는 LTM 엔진이 생성한 데이터를 시각화하는 인터페이스입니다. Pieces Desktop 앱에서 제공되는 이 뷰는 시간순으로 정렬된 Roll-Up 목록을 보여줍니다.

각 Roll-Up은 다음 정보를 포함합니다:
* 작업한 파일 경로와 코드 변경사항
* 열람한 문서와 웹페이지 URL
* 해결한 이슈 번호와 관련 커밋
* 팀원과의 대화 내용
* 내린 기술적 결정과 그 근거

실전 활용 예시를 들면, 비동기 스탠드업을 준비할 때 "이번 주 월요일부터 금요일까지 작업 내용"을 조회하면 해당 기간의 모든 Roll-Up이 취합되어 마크다운 형식의 보고서로 생성됩니다. 프로젝트 인수인계 시에는 특정 기간을 필터링하여 컨텍스트를 추출하고, 이를 동료에게 공유할 수 있습니다.

## MCP (Model Context Protocol) 통합
MCP는 Anthropic이 개발한 오픈 표준 프로토콜입니다. AI 애플리케이션이 외부 도구와 데이터 소스에 표준화된 방식으로 접근할 수 있도록 설계되었습니다.

Pieces OS는 MCP 서버로 동작할 수 있으며, 이를 통해 Claude Desktop이 Pieces의 LTM 데이터에 직접 쿼리를 날릴 수 있습니다. 사용자가 Claude에게 "어제 작업 내용"을 물으면, Claude는 `ask_pieces_ltm` 도구를 호출하여 해당 시점의 Roll-Up을 검색하고 응답을 생성합니다.

이 구조의 장점은 매번 AI에게 맥락을 설명할 필요가 없다는 점입니다. Claude는 필요할 때마다 Pieces LTM에서 관련 정보를 가져와 응답에 활용합니다.

## 설치 및 연동 절차

### 사전 요구사항
1. **Pieces OS 설치**가 선행되어야 합니다. 공식 사이트([pieces.app/download](https://pieces.app/download))에서 OS에 맞는 버전을 다운로드합니다. 설치 후 시스템 트레이에 Pieces 아이콘이 표시되는지 확인합니다.
2. **Long-Term Memory 엔진을 활성화**해야 합니다. Pieces Desktop App의 설정에서 LTM-2.7 옵션을 활성화하거나, 시스템 트레이의 Quick Menu에서 활성화할 수 있습니다.
3. **Claude Desktop 최신 버전**을 설치합니다. Claude 메뉴에서 "Check for Updates"로 버전을 확인합니다.

### Pieces CLI 설치
Windows 환경에서는 PowerShell을 관리자 권한으로 실행하여 다음 명령을 실행합니다:
```powershell
py -m pip install --upgrade pip
py -m pip install pieces-cli
pieces --version

```

macOS/Linux 환경:

```bash
pip install --upgrade pip
pip install pieces-cli
pieces --version

```

설치 후 Pieces CLI 실행 파일의 정확한 경로를 확인해야 합니다. 이 경로는 Python 설치 방식에 따라 다릅니다.

* **Windows 일반적인 경로:** `C:\Users\<사용자명>\AppData\Local\Programs\Python\Python3XX\Scripts\pieces`
* **macOS:** `/usr/local/bin/pieces`
* **Linux:** `/home/<사용자명>/.local/bin/pieces`

정확한 경로는 PowerShell에서 `where.exe pieces`, macOS/Linux에서 `which pieces` 명령으로 확인할 수 있습니다.

## Claude Desktop 설정

Claude Desktop의 MCP 설정 파일을 편집해야 합니다. 설정 파일 위치는 OS마다 다릅니다:

* **Windows:** `%AppData%\Claude\claude_desktop_config.json`
* **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
* **Linux:** `~/.config/Claude/claude_desktop_config.json`

Claude Desktop 인터페이스에서 **Settings → Developer → Edit Config**를 클릭하면 에디터가 열립니다. 설정 파일에 다음 내용을 작성합니다.

**Windows 예시:**

```json
{
  "mcpServers": {
    "pieces": {
      "command": "C:\\Users\\YourUsername\\AppData\\Local\\Programs\\Python\\Python313\\Scripts\\pieces",
      "args": ["--ignore-onboarding", "mcp", "start"]
    }
  }
}

```

**macOS/Linux 예시:**

```json
{
  "mcpServers": {
    "pieces": {
      "command": "/usr/local/bin/pieces",
      "args": ["--ignore-onboarding", "mcp", "start"]
    }
  }
}

```

> **주의사항:** Windows 경로는 백슬래시를 이스케이프 처리해야 하므로 `\\`를 사용합니다. 본인의 실제 Python 버전과 사용자명으로 경로를 수정해야 합니다. JSON 문법 오류가 없는지 확인합니다.

## 연동 확인

Claude Desktop을 완전히 종료한 후 재실행합니다. 채팅 입력창 하단에 MCP 연결 아이콘이 표시되는지 확인합니다. "+" 버튼을 클릭하고 "Connectors"를 선택하면 연결된 MCP 서버 목록이 표시됩니다. Pieces 서버가 활성화되어 있어야 합니다.

## 실제 활용 사례

과거 작업 조회 시나리오를 살펴보겠습니다. Claude에게 **"어제 작업한 내용 요약해줘"**라고 요청하면, Claude는 자동으로 `ask_pieces_ltm` 도구를 호출합니다. Pieces LTM이 해당 시점의 Roll-Up을 반환하면, Claude는 이를 분석하여 작업한 프로젝트, 수정한 파일, 해결한 이슈, 참고한 문서를 정리해서 응답합니다.

특정 문제 추적의 경우, **"지난주 Redis 연결 오류 어떻게 해결했지?"**라는 질문에 대해 Claude는 해당 키워드와 시간대로 검색하여 오류 발생 시점, 참고한 문서나 Stack Overflow 링크, 최종 해결 코드를 찾아 제시합니다.

스탠드업 보고서 자동 생성 시에는 **"월요일부터 금요일까지 작업 내용으로 스탠드업 보고서 작성"**을 요청하면, 해당 기간의 모든 Roll-Up을 취합하여 구조화된 마크다운 문서를 생성합니다.

## 문제 해결

MCP 아이콘이 표시되지 않는 경우, 다음을 순서대로 확인합니다.

1. **Pieces OS 실행 여부**를 시스템 트레이에서 확인합니다. 실행 중이 아니면 Pieces Desktop App을 실행합니다.
2. **LTM 엔진이 활성화**되어 있는지 Pieces Desktop App 설정에서 확인합니다.
3. **Pieces CLI 경로**가 정확한지 재확인합니다. `where.exe pieces` 또는 `which pieces`로 실제 경로를 확인하고, `claude_desktop_config.json`의 경로와 일치하는지 검사합니다.
4. **JSON 문법 오류**가 없는지 검증합니다. 온라인 JSON validator를 사용하거나, 백슬래시 이스케이프, 쉼표, 괄호를 점검합니다.
5. **MCP 로그**를 확인합니다. Claude Desktop 메뉴에서 **Help → Enable Developer Mode**를 활성화한 후, **Developer → Open MCP Log File**로 상세 오류 메시지를 확인할 수 있습니다.

"Could not attach to MCP" 오류는 대부분 Python 환경 변수 문제입니다. 관리자 권한으로 PowerShell을 실행하여 `pieces --version` 명령으로 CLI가 정상 작동하는지 테스트합니다. 작동하지 않으면 시스템 환경 변수의 Path에 Python Scripts 폴더가 포함되어 있는지 확인합니다.

## 워크플로우 최적화

Pieces는 워크스페이스 단위로 컨텍스트를 분리할 수 있습니다. 프로젝트별로 워크스페이스를 나누면 서로 다른 프로젝트의 맥락이 섞이지 않아 검색 정확도가 향상됩니다.

시간 필터를 적극 활용하면 효율이 높아집니다. "1월 한 달간 작업", "오늘 오전에 본 문서", "지난주 화요일 작업" 같은 시간 기반 쿼리가 모두 지원됩니다.

Workstream Activity의 키워드 검색 기능으로 프로젝트명, 기술 스택, 버그 번호, 팀원 이름 등으로 필터링할 수 있습니다.

## 데이터 프라이버시

전체 처리 과정의 90%는 오프라인에서 진행됩니다. 온디바이스 머신러닝 모델이 비밀번호, API 키 같은 민감 정보를 자동으로 필터링합니다.

데이터 수집을 세밀하게 제어할 수 있습니다. 특정 애플리케이션을 수집 대상에서 제외하거나, 시간대별로 수집을 일시정지하거나, 주제별로 저장된 데이터를 삭제할 수 있습니다.

모든 데이터는 로컬 디바이스에만 존재하며, Pieces 팀을 포함한 외부에서는 접근할 수 없습니다. 사용자가 명시적으로 공유를 선택하지 않는 한 데이터는 절대 외부로 전송되지 않습니다.

## 참고 자료

* Pieces 다운로드: https://pieces.app/download
* Pieces 공식 문서: https://docs.pieces.app
* MCP 문서: https://modelcontextprotocol.io
* Claude Desktop: https://claude.ai/download
