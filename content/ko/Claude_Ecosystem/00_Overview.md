# Claude 생태계

Claude AI 생태계에 대한 종합 가이드 — Claude Code(CLI 코딩 도구), Claude Desktop, Cowork, 모델 컨텍스트 프로토콜(MCP, Model Context Protocol), 에이전트 SDK(Agent SDK), Claude API를 다룹니다. 이 토픽은 대화형 코딩 세션부터 프로덕션급 AI 에이전트 구축에 이르기까지, Claude를 AI 지원 개발의 전 영역에 걸쳐 활용하기 위한 개념적 이해와 실무 역량을 모두 제공합니다.

## 학습 내용

- **기초**: Claude란 무엇인지, 모델 패밀리(Opus, Sonnet, Haiku), 제품 생태계
- **Claude Code**: 설치, 프로젝트 설정(CLAUDE.md), 권한 모드(permission modes), 일상적 워크플로우
- **자동화**: 이벤트 기반 자동화를 위한 훅(Hooks), 커스텀 스킬(Skills) 및 슬래시 명령어
- **에이전트**: 작업 위임을 위한 서브에이전트(Subagents), 다중 에이전트 협업을 위한 에이전트 팀(Agent Teams)
- **IDE 및 데스크탑**: VS Code / JetBrains 통합, Claude Desktop 기능, Cowork
- **MCP**: 모델 컨텍스트 프로토콜 — Claude를 외부 도구 및 데이터 소스에 연결
- **API 및 SDK**: Messages API, 도구 사용(tool use) / 함수 호출(function calling), 커스텀 에이전트 구축을 위한 에이전트 SDK
- **심화**: 비용 최적화, 개발 워크플로우, 모범 사례, 문제 해결

> **참고**: 이 토픽은 Claude 생태계를 **사용하고 그 위에서 개발하는 것**에 초점을 맞춥니다. 대규모 언어 모델 이면의 이론은 **LLM_and_NLP**와 **Foundation_Models**를 참조하세요. 일반적인 프롬프트 엔지니어링 개념은 **LLM_and_NLP/08_Prompt_Engineering.md**를 참조하세요.

## 레슨 목록

| # | 제목 | 난이도 | 설명 |
|---|------|--------|------|
| 01 | [Claude 소개](01_Introduction_to_Claude.md) | ⭐ | 모델 패밀리, 핵심 기능, 제품 생태계 개요 |
| 02 | [Claude Code: 시작하기](02_Claude_Code_Getting_Started.md) | ⭐ | 설치, 첫 번째 세션, 기본 워크플로우 (읽기 → 편집 → 테스트 → 커밋) |
| 03 | [CLAUDE.md와 프로젝트 설정](03_CLAUDE_md_and_Project_Setup.md) | ⭐ | 프로젝트 지침, .claude/ 디렉토리, 설정 계층 구조 |
| 04 | [권한 모드와 보안](04_Permission_Modes.md) | ⭐⭐ | 다섯 가지 권한 모드, 허용/거부 규칙, 샌드박싱 |
| 05 | [훅과 이벤트 기반 자동화](05_Hooks.md) | ⭐⭐ | 훅 라이프사이클, 설정, 실용적 자동화 예제 |
| 06 | [스킬과 슬래시 명령어](06_Skills_and_Slash_Commands.md) | ⭐⭐ | SKILL.md, 커스텀 스킬 생성, 내장 명령어 |
| 07 | [서브에이전트와 작업 위임](07_Subagents.md) | ⭐⭐ | 탐색/계획/범용 서브에이전트, 커스텀 정의 |
| 08 | [에이전트 팀](08_Agent_Teams.md) | ⭐⭐⭐ | 다중 에이전트 협업, 공유 작업 목록, 병렬 작업 스트림 |
| 09 | [IDE 통합](09_IDE_Integration.md) | ⭐ | VS Code 확장, JetBrains 플러그인, 키보드 단축키 |
| 10 | [Claude 데스크탑 애플리케이션](10_Claude_Desktop.md) | ⭐ | 데스크탑 기능, 앱 미리보기, GitHub 통합 |
| 11 | [Cowork: AI 디지털 동료](11_Cowork.md) | ⭐⭐ | 다단계 작업 실행, 플러그인, MCP 커넥터 |
| 12 | [모델 컨텍스트 프로토콜(MCP)](12_Model_Context_Protocol.md) | ⭐⭐ | MCP 아키텍처, 사전 구축된 서버, 도구 연결 |
| 13 | [커스텀 MCP 서버 구축](13_Building_MCP_Servers.md) | ⭐⭐⭐ | 리소스/도구/프롬프트 정의, TypeScript/Python 구현 |
| 14 | [Claude 프로젝트와 아티팩트](14_Claude_Projects_and_Artifacts.md) | ⭐ | 프로젝트 구성, 지식 기반, 아티팩트 유형 |
| 15 | [Claude API 기초](15_Claude_API_Fundamentals.md) | ⭐⭐ | 인증, Messages API, 스트리밍, 클라이언트 SDK |
| 16 | [도구 사용과 함수 호출](16_Tool_Use_and_Function_Calling.md) | ⭐⭐ | 도구 정의, 호출 패턴, 병렬 실행 |
| 17 | [Claude 에이전트 SDK](17_Claude_Agent_SDK.md) | ⭐⭐⭐ | SDK 개요, 에이전트 루프, 내장 도구, 컨텍스트 관리 |
| 18 | [커스텀 에이전트 구축](18_Building_Custom_Agents.md) | ⭐⭐⭐ | 커스텀 도구, 시스템 프롬프트, 프로덕션 에이전트 패턴 |
| 19 | [모델, 가격, 최적화](19_Models_and_Pricing.md) | ⭐⭐ | 모델 비교, 가격 책정, 프롬프트 캐싱, 배치 API |
| 20 | [고급 개발 워크플로우](20_Advanced_Workflows.md) | ⭐⭐⭐ | 다중 파일 리팩토링, TDD, CI/CD 통합, 코드베이스 탐색 |
| 21 | [모범 사례와 패턴](21_Best_Practices.md) | ⭐⭐ | 효과적인 프롬프트, 컨텍스트 관리, 보안, 팀 패턴 |
| 22 | [문제 해결과 디버깅](22_Troubleshooting.md) | ⭐⭐ | 권한 오류, 훅 실패, 컨텍스트 한계, MCP 문제 |

## 선행 조건

- **기본 프로그래밍 경험**: 최소 하나의 프로그래밍 언어에 익숙할 것 (Python 또는 TypeScript 권장)
- **커맨드라인 친숙도**: 기본 터미널 사용법 (필요 시 **Shell_Script** 토픽 참조)
- **Git 기초**: 커밋, 브랜치, 풀 리퀘스트(pull request) 이해 (**Git** 토픽 참조)

AI 도구나 API에 대한 사전 경험은 필요하지 않습니다 — 이 토픽은 기초부터 시작합니다.

## 학습 경로

**1단계 — 기초 (레슨 1–3)**
Claude 시작하기: 모델 패밀리와 제품 생태계를 이해하고, Claude Code를 설치하고, CLAUDE.md로 프로젝트를 설정하는 방법을 배웁니다. 이 레슨들을 마치면 일상적인 개발에 Claude Code를 사용할 수 있습니다.

**2단계 — 핵심 기능 (레슨 4–9)**
Claude Code의 강력한 기능을 마스터합니다: 보안을 위한 권한 모드, 자동화를 위한 훅, 커스텀 명령어를 위한 스킬, 복잡한 작업을 위한 서브에이전트, 협업을 위한 에이전트 팀, 원활한 개발을 위한 IDE 통합.

**3단계 — 플랫폼 & 통합 (레슨 10–14)**
더 넓은 Claude 플랫폼을 탐색합니다: 데스크탑 애플리케이션, 자율적 작업 실행을 위한 Cowork, 외부 도구 및 데이터 소스 연결을 위한 MCP, 지식 구성 및 아티팩트 생성을 위한 프로젝트.

**4단계 — API 및 심화 (레슨 15–22)**
프로덕션급 AI 애플리케이션 구축: Claude API와 에이전트 SDK 활용, 커스텀 도구 및 에이전트 생성, 비용 최적화, 고급 개발 워크플로우 마스터, 팀 모범 사례 및 문제 해결 학습.

## 관련 토픽

- **LLM_and_NLP**: 대규모 언어 모델의 이론, BERT, GPT 아키텍처, 프롬프트 엔지니어링
- **Foundation_Models**: 스케일링 법칙, 모델 학습, PEFT, RAG 아키텍처
- **Programming**: 클린 코드, 디자인 패턴, 테스팅 — AI 지원 개발을 위한 기초 역량
- **Git**: Claude Code가 깊이 통합된 버전 관리 워크플로우
- **Docker**: 안전한 Claude Code 실행을 위한 컨테이너 환경 (Bypass 모드)
- **Web_Development**: Claude가 생성하고 반복 개선할 수 있는 애플리케이션 구축

**라이선스**: 콘텐츠는 CC BY-NC 4.0 라이선스 하에 제공됩니다
