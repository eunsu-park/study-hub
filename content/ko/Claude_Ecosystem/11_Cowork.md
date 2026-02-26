# Cowork: AI 디지털 동료

**이전**: [10. Claude 데스크톱 애플리케이션](./10_Claude_Desktop.md) | **다음**: [12. 모델 컨텍스트 프로토콜 (MCP)](./12_Model_Context_Protocol.md)

---

Cowork는 Claude를 자율적인 **디지털 동료(digital colleague)**로 배포하는 Anthropic의 제품으로, 코딩을 넘어 더 넓은 작업을 수행합니다. Claude Code가 터미널이나 IDE에서 코딩에 집중하는 반면, Cowork는 더 독립적으로 운영되어 문서 생성, 연구 종합, 프로젝트 관리 작업, 플러그인 생태계와 MCP 커넥터를 통한 서비스 간 통합과 같은 다단계 워크플로를 처리합니다. 이 레슨에서는 Cowork가 무엇인지, 어떻게 작동하는지, 언제 사용해야 하는지를 다룹니다.

**난이도**: ⭐⭐

**전제 조건**:
- 레슨 01: Claude 소개 (제품 생태계 개요)
- 레슨 10: Claude Desktop (데스크톱 환경 이해)
- 워크플로 자동화 개념에 대한 기본 이해

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Cowork가 무엇이며 Claude Code와 어떻게 다른지 이해
2. 워크플로에 맞게 Cowork 설정 및 구성
3. 다양한 도메인에서 다단계 작업 실행에 Cowork 사용
4. 플러그인 생태계 탐색 및 관련 플러그인 활성화
5. MCP 커넥터를 통해 외부 서비스 연결
6. 실용적인 사용 사례에 Cowork 적용 (문서화, 연구, 프로젝트 관리)
7. 개인 정보 보호, 데이터 처리, 한계 이해

---

## 목차

1. [Cowork란?](#1-cowork란)
2. [Cowork vs. Claude Code](#2-cowork-vs-claude-code)
3. [시작하기](#3-시작하기)
4. [다단계 작업 실행](#4-다단계-작업-실행)
5. [플러그인 생태계](#5-플러그인-생태계)
6. [MCP 커넥터 통합](#6-mcp-커넥터-통합)
7. [실용적인 사용 사례](#7-실용적인-사용-사례)
8. [자율 운영](#8-자율-운영)
9. [한계 및 모범 사례](#9-한계-및-모범-사례)
10. [개인 정보 보호 및 데이터 처리](#10-개인-정보-보호-및-데이터-처리)
11. [연습 문제](#11-연습-문제)
12. [참고 자료](#12-참고-자료)

---

## 1. Cowork란?

Cowork는 Claude를 **디지털 동료**로 자리매김합니다 — 챗봇보다 더 자율적으로 작동하지만 잘 정의된 경계 내에서 운영되는 AI입니다. 한 번에 하나의 프롬프트에 응답하는 대신, Cowork는 고수준의 목표를 받아 단계별로 분해하고, 사용 가능한 도구와 통합을 사용하여 해당 단계를 실행하며, 포괄적인 결과를 생성할 수 있습니다.

```
Traditional chatbot:
  You → Question → AI → Answer → You → Follow-up → AI → Answer → ...

Claude Code:
  You → "Fix this bug" → Claude → [reads code, edits, tests] → Result

Cowork:
  You → "Prepare the Q1 engineering report" → Cowork →
    [reads project docs]
    [queries issue tracker]
    [pulls metrics from dashboards]
    [drafts report sections]
    [formats and polishes]
    → Complete Q1 Report
```

### 핵심 특성

- **자율 실행**: 목표가 주어지면 Cowork가 지속적인 인간의 입력 없이 여러 단계를 계획하고 실행
- **서비스 간 연결**: 플러그인과 MCP를 통해 다양한 도구 및 서비스에 연결
- **더 넓은 범위**: 코딩에 제한되지 않음 — 문서, 데이터, 커뮤니케이션, 프로젝트 관리 처리
- **체크포인트 기반**: 주요 체크포인트에서 진행 상황을 보고하여 필요 시 방향 수정 가능
- **감사 가능**: 수행된 작업과 내린 결정에 대한 완전한 로그 제공

---

## 2. Cowork vs. Claude Code

이 제품들 사이의 경계를 이해하면 올바른 도구를 선택하는 데 도움이 됩니다.

| 차원 | Claude Code | Cowork |
|-----------|-------------|--------|
| **주요 도메인** | 소프트웨어 개발 | 광범위한 지식 작업 |
| **입력** | 코드에 관한 자연어 | 모든 작업에 관한 자연어 |
| **도구** | 파일 시스템, 터미널, git, 웹 | 플러그인, MCP 커넥터, 문서 |
| **자율성** | 반자율 (권한 요청) | 더 자율적 (체크포인트 기반) |
| **환경** | 터미널 / IDE | 전용 인터페이스 |
| **출력** | 코드 변경, 테스트 결과, 커밋 | 문서, 보고서, 처리된 데이터 |
| **일반적인 작업** | 리팩토링, 디버깅, 구현, 테스트 | 연구, 초안 작성, 정리, 분석 |
| **컨텍스트** | 코드베이스 (파일, git 이력) | 더 넓은 워크스페이스 (문서, 서비스) |
| **사용자** | 소프트웨어 개발자 | 팀의 누구나 |

### 중복 및 상호 보완성

두 제품 사이에는 중복이 있습니다. 둘 다 파일을 편집하고, 웹을 검색하고, 작업을 자동화할 수 있습니다. 차이점은 **강조점과 설계**에 있습니다:

```
Claude Code:
  Optimized for → code editing → testing → committing
  Context → repository structure, file contents, build systems

Cowork:
  Optimized for → multi-step workflows → service integration → document production
  Context → organizational tools, documents, communications
```

실제로 소프트웨어 엔지니어는 두 가지를 모두 사용할 수 있습니다:
- **Claude Code**: 기능 구현, 버그 수정, 테스트 작성
- **Cowork**: 설계 문서 초안 작성, 회의 노트 처리, 스프린트 보고서 준비, 서비스 전반의 의존성 업데이트 관리

---

## 3. 시작하기

### 접근 방법

Cowork는 다음을 통해 사용할 수 있습니다:
1. Claude Desktop 애플리케이션 (통합 모드로)
2. Claude.ai 웹 인터페이스 (Team 및 Enterprise 플랜)
3. 프로그래밍 방식의 통합을 위한 API 접근

### 초기 설정

```
┌──────────────────────────────────────────────────────────────┐
│  Cowork Setup                                                │
│                                                              │
│  Step 1: Select your workspace                               │
│  ┌──────────────────────────────────────────────────────────┐│
│  │  ○ Personal workspace                                    ││
│  │  ● Team workspace: "Engineering Team"                    ││
│  │  ○ Enterprise workspace: "Acme Corp"                     ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
│  Step 2: Connect services (optional)                         │
│  ☑ GitHub        Connected (acme-org)                        │
│  ☑ Google Drive  Connected (engineering@acme.com)            │
│  ☐ Slack         Not connected  [Connect]                    │
│  ☐ Jira          Not connected  [Connect]                    │
│  ☐ Confluence    Not connected  [Connect]                    │
│                                                              │
│  Step 3: Configure permissions                               │
│  ☑ Read files and documents                                  │
│  ☑ Create and edit documents                                 │
│  ☐ Send messages on your behalf                              │
│  ☐ Create issues/tickets on your behalf                      │
│                                                              │
│  [Get Started]                                               │
└──────────────────────────────────────────────────────────────┘
```

### 첫 번째 작업

```
You: Prepare a summary of all open issues in the payments
     repository, grouped by severity, with suggested priorities
     for the next sprint.

Cowork: I'll prepare that summary. Here's my plan:

     1. Fetch open issues from github.com/acme/payments
     2. Categorize by severity labels (critical, high, medium, low)
     3. Analyze each issue for sprint-readiness (clear scope? dependencies?)
     4. Draft priority recommendations based on severity + effort
     5. Format as a structured report

     Starting now. I'll check in after step 2.

     [Checkpoint 1]
     Found 34 open issues:
     - Critical: 2
     - High: 8
     - Medium: 15
     - Low: 9

     Continuing to analysis...

     [Final Result]
     Sprint Planning Summary attached.
     See: payments-sprint-summary.md
```

---

## 4. 다단계 작업 실행

Cowork는 서로 다른 도구와 데이터 소스를 아우르는 여러 단계가 필요한 작업에서 탁월합니다.

### 4.1 파일 정리 및 관리

```
You: Organize the Q4 reports folder. Group reports by department,
     rename files to follow our naming convention (YYYY-MM_Department_Type.pdf),
     and create an index document.

Cowork executes:
  1. Reads all files in Q4-reports/
  2. Identifies departments from file contents
  3. Creates department subdirectories
  4. Renames files following the convention
  5. Moves files to appropriate directories
  6. Creates INDEX.md with a table of all reports
```

### 4.2 문서 생성 및 편집

```
You: Draft a technical design document for the new caching layer.
     Use our design doc template, reference the existing database
     architecture in docs/architecture.md, and include performance
     benchmarks from last sprint's analysis.

Cowork executes:
  1. Reads the design doc template
  2. Reads docs/architecture.md for context
  3. Searches for performance benchmark data
  4. Drafts the document following the template structure
  5. Fills in architecture diagrams (ASCII/Mermaid)
  6. Adds performance data tables
  7. Produces: docs/designs/caching-layer-design.md
```

### 4.3 연구 및 종합

```
You: Research the top 5 options for real-time analytics databases.
     Compare them on: performance, cost, ease of integration with
     our Python/PostgreSQL stack, community support, and managed
     service availability. Produce a comparison matrix.

Cowork executes:
  1. Searches for real-time analytics databases (web search)
  2. Reads documentation for top candidates
  3. Checks each for Python client libraries
  4. Checks PostgreSQL integration capabilities
  5. Gathers pricing information
  6. Assesses community activity (GitHub stars, recent commits)
  7. Creates comparison matrix with scoring
  8. Writes recommendation with rationale
```

### 4.4 프로젝트 자동화

```
You: Set up the new microservice project. Create the directory
     structure following our standard template, initialize git,
     configure CI/CD, add the standard linting and testing setup,
     and create the initial README with our boilerplate.

Cowork executes:
  1. Creates directory structure from template
  2. Initializes git repository
  3. Creates .github/workflows/ with CI/CD configs
  4. Sets up linting (eslint/prettier or ruff/black)
  5. Configures testing framework
  6. Creates README.md from boilerplate
  7. Creates initial commit
  8. Pushes to GitHub (if authorized)
```

---

## 5. 플러그인 생태계

Cowork는 외부 서비스 및 도구에 연결하는 사전 구축된 통합인 **플러그인**을 통해 기능을 확장합니다.

### 사용 가능한 플러그인

| 플러그인 | 기능 | 사용 사례 |
|--------|-------------|-----------|
| **GitHub** | 이슈, PR, 저장소, 액션 | 코드 리뷰, 이슈 분류, CI 모니터링 |
| **Google Workspace** | 문서, 시트, 슬라이드, 드라이브 | 문서 생성, 데이터 분석, 프레젠테이션 |
| **Slack** | 채널, 메시지, 스레드 | 커뮤니케이션 요약, 응답 초안 작성 |
| **Jira** | 이슈, 스프린트, 보드 | 스프린트 계획, 이슈 관리 |
| **Confluence** | 페이지, 스페이스, 검색 | 문서화, 지식 베이스 |
| **Linear** | 이슈, 프로젝트, 사이클 | 작업 추적, 프로젝트 관리 |
| **Notion** | 페이지, 데이터베이스 | 지식 관리, 작업 추적 |
| **Calendar** | 이벤트, 일정 | 회의 준비, 일정 분석 |

### 플러그인 기능

각 플러그인은 특정 액션을 제공합니다:

```
GitHub Plugin:
  READ:
  - List repositories
  - Get issue details
  - Read PR diffs
  - Check CI status
  - Read file contents

  WRITE (if authorized):
  - Create issues
  - Comment on PRs
  - Create/update files
  - Trigger workflows
  - Approve/request changes
```

### 플러그인 관리

```
┌──────────────────────────────────────────────────────────────┐
│  Plugin Management                                           │
│                                                              │
│  Installed:                                                  │
│  ✓ GitHub        v2.1.0   [Settings] [Disable]              │
│  ✓ Google Drive  v1.8.0   [Settings] [Disable]              │
│  ✓ Slack         v1.5.0   [Settings] [Disable]              │
│                                                              │
│  Available:                                                  │
│  ○ Jira          v2.0.0   [Install]                         │
│  ○ Confluence    v1.3.0   [Install]                         │
│  ○ Linear        v1.1.0   [Install]                         │
│  ○ Notion        v1.6.0   [Install]                         │
│  ○ Calendar      v1.0.0   [Install]                         │
│                                                              │
│  [Browse All Plugins]                                        │
└──────────────────────────────────────────────────────────────┘
```

### 플러그인 권한

각 플러그인에는 구성 가능한 권한 수준이 있습니다:

```
GitHub Plugin Settings:
  ☑ Read repository contents
  ☑ Read issues and pull requests
  ☑ Read CI/CD check status
  ☐ Create and edit issues
  ☐ Comment on pull requests
  ☐ Push commits to branches
  ☐ Trigger GitHub Actions

  Scope: [Selected repos ▼]
  - acme/payments
  - acme/frontend
  - acme/infrastructure
```

---

## 6. MCP 커넥터 통합

사전 구축된 플러그인 외에도, Cowork는 MCP 표준을 구현하는 모든 외부 서비스에 연결하기 위한 **MCP(모델 컨텍스트 프로토콜, Model Context Protocol) 커넥터**를 지원합니다.

### MCP 커넥터란?

MCP 커넥터는 Cowork와 외부 데이터 소스 또는 도구 사이의 표준화된 브릿지입니다. 레슨 12(모델 컨텍스트 프로토콜)에서 설명하는 것과 동일한 프로토콜을 따르지만, Claude Code의 설정이 아닌 Cowork의 인터페이스 내에서 구성됩니다.

```
┌──────────────────────────────────────────────────────────────┐
│                     Cowork                                    │
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │ Plugin:  │    │  MCP:    │    │  MCP:    │             │
│   │ GitHub   │    │ Postgres │    │ Internal │             │
│   │          │    │ DB       │    │ API      │             │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘             │
│        │               │               │                    │
└────────┼───────────────┼───────────────┼────────────────────┘
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │ GitHub  │    │Postgres │    │Internal │
    │ API     │    │Database │    │  REST   │
    └─────────┘    └─────────┘    └─────────┘
```

### MCP 서버 연결

```
┌──────────────────────────────────────────────────────────────┐
│  MCP Connector Setup                                         │
│                                                              │
│  Add MCP Server:                                             │
│                                                              │
│  Name:      [Production Database      ]                      │
│  Type:      [stdio ▼]                                        │
│  Command:   [npx @anthropic/mcp-postgres                ]    │
│  Args:      [postgresql://read-only@db.acme.com/prod    ]   │
│                                                              │
│  [Test Connection]  [Save]                                   │
│                                                              │
│  Connected MCP Servers:                                      │
│  ✓ Production Database   (3 tools, 5 resources)             │
│  ✓ Internal Wiki         (2 tools, 12 resources)            │
│  ✗ Analytics API         (disconnected)  [Reconnect]        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 작업에서 MCP 리소스 사용

연결 후 MCP 리소스는 사용 가능한 데이터 소스로 표시됩니다:

```
You: What were the top 10 most common errors in production
     last week? Cross-reference with our open GitHub issues
     to identify any that are already being tracked.

Cowork:
  1. [MCP: Production Database] Query error logs for last 7 days
  2. [MCP: Production Database] Aggregate by error type and count
  3. [Plugin: GitHub] Fetch open issues with "bug" label
  4. Cross-reference error messages with issue titles/descriptions
  5. Produce report with matched and unmatched errors

Result:
  Top 10 Errors (Last 7 Days):

  | # | Error | Count | GitHub Issue |
  |---|-------|-------|-------------|
  | 1 | NullPointerException in PaymentService | 342 | #234 (open) |
  | 2 | TimeoutError: Redis connection | 218 | #256 (open) |
  | 3 | ValidationError: invalid email format | 156 | Not tracked |
  | ... | ... | ... | ... |

  3 errors are not currently tracked in GitHub.
  Shall I create issues for them?
```

---

## 7. 실용적인 사용 사례

### 7.1 프로젝트 문서 생성

```
You: Generate comprehensive API documentation for the payments
     service. Include endpoints, request/response schemas,
     authentication requirements, and example curl commands.
     Publish to our Confluence space.

Cowork workflow:
  1. Read source code (via GitHub or filesystem)
  2. Extract route definitions and handler logic
  3. Identify request/response types
  4. Generate endpoint documentation
  5. Create example requests
  6. Format as Confluence-compatible markup
  7. Publish to "Engineering > API Docs" space
```

### 7.2 회의 노트 처리

```
You: Process the meeting notes from yesterday's architecture review.
     Extract: decisions made, action items (with assignees), open
     questions, and follow-up meeting needs. Post the summary to
     #engineering in Slack and create Jira tickets for action items.

Cowork workflow:
  1. Read meeting notes from Google Drive
  2. Extract structured information:
     - Decisions: 3 items
     - Action items: 5 items with assignees
     - Open questions: 2 items
     - Follow-ups: 1 meeting needed
  3. Format summary for Slack
  4. Post to #engineering channel
  5. Create 5 Jira tickets (one per action item)
  6. Link Jira tickets back to the Slack summary
```

### 7.3 코드 리뷰 준비

```
You: Prepare me for reviewing PR #342 in the payments repo.
     Summarize the changes, identify areas of concern, check
     if tests cover the modified code, and note any files that
     typically cause issues.

Cowork workflow:
  1. Fetch PR #342 details and diff
  2. Summarize changes by module
  3. Analyze code for potential issues:
     - Security: input validation, auth checks
     - Performance: query efficiency, caching
     - Correctness: edge cases, error handling
  4. Check test coverage for modified files
  5. Query git history for frequently-reverted files
  6. Compile review preparation document
```

### 7.4 의존성 분석 및 업데이트

```
You: Analyze all our JavaScript projects for outdated dependencies.
     Identify security vulnerabilities, available updates, and
     breaking change risks. Prioritize updates by security impact.

Cowork workflow:
  1. List all JS repositories in the organization
  2. Read package.json from each
  3. Check each dependency against npm registry
  4. Query vulnerability databases (npm audit, Snyk)
  5. Assess breaking change risk (major version bumps)
  6. Create prioritized update plan:
     Priority 1: Security vulnerabilities (CVE)
     Priority 2: Major updates with migration guides
     Priority 3: Minor/patch updates
  7. Generate update report with effort estimates
```

---

## 8. 자율 운영

Cowork의 특징적인 기능은 전통적인 챗봇보다 더 큰 자율성으로 운영할 수 있는 능력입니다.

### 체크포인트 모델

매 단계마다 권한을 요청하는 대신, Cowork는 체크포인트를 사용합니다 — 진행 상황을 보고하고 선택적으로 확인을 기다리기 위해 일시 중지하는 순간:

```
Task: "Reorganize the engineering wiki"

Checkpoint 1 (after analysis):
  "I've analyzed the wiki. It has 234 pages across 12 spaces.
   I propose reorganizing into 5 main areas:
   1. Architecture (42 pages)
   2. Processes (38 pages)
   3. Runbooks (56 pages)
   4. Onboarding (31 pages)
   5. Reference (67 pages)

   Should I proceed with this structure?"

You: Yes, but merge Runbooks into Processes.

Checkpoint 2 (after reorganization):
  "Reorganization complete. I've moved 234 pages into 4 areas.
   Updated 89 internal links. Found 12 broken links that were
   pre-existing — shall I fix those too?"

You: Yes, fix them.

Final: "Done. All pages reorganized, all links updated and fixed.
        See the new structure at wiki.acme.com/engineering"
```

### 자율성 수준

Cowork가 얼마나 자율적으로 운영할지 구성할 수 있습니다:

```
Autonomy Settings:
  ○ Supervised:  Ask before every action
  ● Checkpoint:  Execute steps, report at key points (recommended)
  ○ Autonomous:  Execute entire task, report at completion
```

### 가드레일(Guardrails)

자율 모드에서도 Cowork는 가드레일을 적용합니다:

- **명시적 확인 없이 데이터를 절대 삭제하지 않음**
- **권한 없이 대신 메시지를 절대 보내지 않음**
- **권한 없이 코드를 절대 커밋하거나 푸시하지 않음**
- **감사 가능성을 위해 모든 작업을 로그에 기록**
- **오류 발생 시 중단하고 인간의 결정을 위해 보고**

---

## 9. 한계 및 모범 사례

### 한계

**1. 실시간이 아님**

Cowork는 세션 내에서 순차적으로 작업을 처리합니다. 지속적으로 실행되는 백그라운드 서비스가 아닙니다:

```
Cowork is NOT:
  - A monitoring system that alerts you to issues
  - A daemon that runs 24/7
  - A real-time event processor

Cowork IS:
  - A task executor that you invoke with a goal
  - A batch processor that handles multi-step workflows
  - A research assistant that synthesizes information
```

**2. 컨텍스트 창 제한**

모든 Claude 제품과 마찬가지로 Cowork는 유한한 컨텍스트 창을 가집니다. 매우 큰 작업은 더 작은 부분으로 나눠야 할 수 있습니다:

```
Too large: "Process all 5,000 customer support tickets from Q4"
Better:    "Process the 50 highest-priority customer tickets from Q4"
           (then iterate for remaining batches)
```

**3. 서비스 가용성**

Cowork는 연결된 서비스가 사용 가능한 상태에 의존합니다. GitHub, Slack 또는 데이터베이스가 다운되면 관련 작업이 실패합니다:

```
Cowork: "Unable to complete step 3 — GitHub API returned 503.
         I've saved my progress. You can retry when GitHub
         is back online."
```

**4. 세션 간 학습 없음**

각 Cowork 세션은 새로 시작됩니다. 이전 세션에서 학습하거나 조직에 대한 지속적인 이해를 구축하지 않습니다:

```
Session 1: "Where are the architecture docs?"
Cowork: [searches and finds them in Confluence/Engineering/Architecture]

Session 2: "Update the architecture docs"
Cowork: [must search again — does not remember from Session 1]
```

### 모범 사례

**1. 범위를 구체적으로 지정**

```
# Vague (leads to broad, unfocused work)
"Clean up the project documentation"

# Specific (clear deliverable)
"Update the API documentation in docs/api/ to reflect the
 new pagination parameters added in PR #345. Also add curl
 examples for the three new endpoints."
```

**2. 출력 형식 지정**

```
"Create a dependency audit report in Markdown format with:
 - Table of outdated packages (name, current, latest, severity)
 - Summary statistics at the top
 - Recommended update order at the bottom"
```

**3. 경계 설정**

```
"Analyze the test coverage gaps but do NOT create any new test files.
 Only produce a report listing which functions lack tests."
```

**4. 위험한 작업에는 체크포인트 사용**

```
"Reorganize the shared drive. IMPORTANT: Show me the proposed
 new structure before making any moves. I want to approve the
 plan first."
```

**5. 작게 시작하고 확장**

```
# Start with a focused task
"Summarize the 5 most recent PRs in the backend repo"

# If satisfied, expand
"Now do the same for the frontend and infrastructure repos"
```

---

## 10. 개인 정보 보호 및 데이터 처리

### Cowork가 접근하는 데이터는?

Cowork는 명시적으로 승인한 것만 접근합니다:

- **파일**: 접근 권한을 부여한 디렉토리에 있는 것만
- **서비스**: 설치하고 승인한 플러그인을 통해서만
- **MCP**: 구성한 커넥터를 통해서만

### 데이터 처리

```
Your data flow:
  Service (GitHub, Slack, etc.)
    │
    ▼
  Plugin/MCP Connector
    │
    ▼
  Cowork session (processes data, generates output)
    │
    ▼
  Result (to you) + Action (to service, if authorized)
```

### 데이터 보존

- **세션 데이터**: 세션 기간 동안 보존
- **출력**: 지정한 곳에 저장 (로컬 파일, 서비스)
- **로그**: 조직의 정책에 따라 액션 로그 보존
- **학습 없음**: 사용자의 데이터는 Claude 모델 훈련에 사용되지 않음 (Anthropic의 상업 고객을 위한 데이터 정책에 따라)

### 엔터프라이즈 제어

Team 및 Enterprise 플랜의 경우:

```
Admin Settings:
  ☑ Require SSO authentication
  ☑ Log all Cowork actions to audit trail
  ☑ Restrict plugin installation to admin-approved list
  ☑ Limit MCP connections to approved servers
  ☐ Allow autonomous mode (disabled by default)
  ☑ Data residency: US only
```

### 민감한 데이터 처리

```
Best practices:
  - Do NOT paste API keys, passwords, or tokens into Cowork
  - Use environment variables or secrets managers for credentials
  - Review plugin permissions carefully
  - Use read-only database connections for MCP
  - Enable audit logging for compliance
```

---

## 11. 연습 문제

### 연습 1: 첫 번째 Cowork 작업

Cowork 세션을 시작하고 다단계 작업을 수행하세요:
1. 주요 언어에 대한 상위 3개 테스트 프레임워크 조사
2. 성능, 커뮤니티 지원, 학습 곡선 비교
3. Cowork에 비교 표를 생성해 달라고 요청

### 연습 2: 플러그인 통합

1. Cowork에 GitHub 플러그인 연결
2. Cowork에 저장소의 모든 오픈 이슈를 요약해 달라고 요청
3. Cowork가 이슈를 유형별로 분류하도록 요청 (버그, 기능, 문서)
4. 출력의 정확성 검토

### 연습 3: 문서 생성

1. Cowork에 코드베이스 디렉토리 제공
2. 다음을 포함하는 개발자 온보딩 가이드 생성을 요청:
   - 프로젝트 구조
   - 설정 지침
   - 주요 패턴 및 관례
   - 일반적인 디버깅 팁
3. 생성된 문서의 완성도 검토

### 연습 4: 자율성 수준

두 가지 다른 자율성 수준에서 동일한 작업 수행:
1. **감독(Supervised)**: "이 회의 노트를 정리하세요" (각 액션을 승인)
2. **체크포인트(Checkpoint)**: "이 회의 노트를 정리하세요" (Cowork가 작업하도록 두고, 체크포인트에서 검토)
3. 경험 비교: 소요 시간, 품질, 제어 수준

### 연습 5: MCP 커넥터

1. 로컬 PostgreSQL 데이터베이스에 MCP 커넥터 설정
2. Cowork에 스키마를 분석하고 다음을 식별해 달라고 요청:
   - 인덱스가 누락된 테이블
   - 제약 조건이 없는 컬럼
   - 잠재적인 정규화 문제
3. Cowork가 스키마 개선 계획을 작성하도록 요청

---

## 12. 참고 자료

- [Cowork Documentation](https://docs.anthropic.com/en/docs/cowork)
- [Anthropic Blog: Introducing Cowork](https://www.anthropic.com/news)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Claude Enterprise Features](https://www.anthropic.com/enterprise)
- [Anthropic Privacy Policy](https://www.anthropic.com/privacy)

---

## 다음 단계

다음 레슨 [모델 컨텍스트 프로토콜 (MCP)](./12_Model_Context_Protocol.md)에서는 Cowork의 MCP 커넥터와 Claude Code의 도구 통합을 구동하는 프로토콜을 깊이 살펴봅니다. MCP 아키텍처, 세 가지 기본 요소(리소스, 도구, 프롬프트), 사전 구축된 MCP 서버를 구성하는 방법, 그리고 AI를 외부 시스템에 연결하는 보안 고려사항을 배웁니다.
