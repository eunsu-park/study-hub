# 26. 에이전트 평가와 벤치마크 (Agent Evaluation and Benchmarks)

이전: [에이전트 메모리와 계획](./25_Agent_Memory_and_Planning.md) | 다음: [에이전트 설계 패턴](./27_Agent_Design_Patterns.md)

## 학습 목표

- 에이전트 시스템(agentic system) 평가가 정적 LLM 평가와 다른 고유한 도전 과제를 이해한다
- 주요 에이전트 벤치마크 비교: AgentBench, SWE-bench, WebArena, GAIA
- 작업 완료(task completion), 효율성(efficiency), 안전성(safety)을 포괄하는 평가 방법론을 설계한다
- 일반적인 에이전트 실패 모드를 분석하고 탐지 메커니즘을 구축한다
- 에이전트 관측 가능성(observability), 추적(tracing), 비용-품질 분석을 구현한다
- 도메인 특화 에이전트를 위한 맞춤형 평가 프레임워크를 구축한다

---

## 목차

1. [에이전트 평가의 도전 과제](#1-에이전트-평가의-도전-과제)
2. [AgentBench](#2-agentbench)
3. [SWE-bench](#3-swe-bench)
4. [WebArena](#4-webarena)
5. [GAIA 벤치마크](#5-gaia-벤치마크)
6. [평가 방법론](#6-평가-방법론)
7. [실패 모드 분석](#7-실패-모드-분석)
8. [에이전트 관측 가능성과 추적](#8-에이전트-관측-가능성과-추적)
9. [비용-품질 분석](#9-비용-품질-분석)
10. [맞춤형 에이전트 평가 구축](#10-맞춤형-에이전트-평가-구축)
11. [연습문제](#연습문제)

---

## 1. 에이전트 평가의 도전 과제

### 이론: 궤적 vs 결과

에이전트 품질에 대한 두 관점:

- **결과만** — 에이전트가 목표를 달성했는가? 최종 답에 대한 통과/실패 또는 점수. 측정이 쉽고, 가끔 기만적(에이전트가 고통스러운 궤적을 통해 우연히 정답에 부딪힐 수 있음).
- **궤적** — 각 단계 검토. 각 도구 호출이 합리적이었나? 에이전트가 사이클을 낭비했나? 안전 제약을 따랐나? 자동 측정이 어렵지만 에이전트 행동 이해에 필수.

프로덕션 에이전트 평가는 둘 다 필요 — 최상위 보고를 위한 결과 지표, 에이전트가 언제(그리고 왜) 실패하는지 이해하기 위한 궤적 지표.

### 에이전트 평가가 어려운 이유

> **정적 LLM vs 에이전트 평가(Static LLM vs Agent Evaluation)**
>
> - **정적 LLM**: 입력 -> 출력 (단일 단계, 결정적 비교)
> - **에이전트**: 입력 -> [계획, 행동, 관찰, 반성]* -> 출력 (다단계, 비결정적)
>
> 에이전트는 궤적 품질(trajectory quality), 도구 사용 정확도(tool usage correctness),
> 효율성(efficiency), 안전성(safety), 오류 복구(recovery from errors)라는 새로운 차원을
> 도입하며 — 이 중 어느 것도 MMLU나 HellaSwag 같은 전통적인 LLM 벤치마크에는
> 적용되지 않는다.

### 평가 차원 (Evaluation Dimensions)

| 차원 | 측정 대상 | 예시 지표 |
|------|----------|----------|
| **작업 완료(Task Completion)** | 에이전트가 목표를 달성했는가? | 성공률 (%) |
| **정확도(Correctness)** | 출력이 사실적으로 정확한가? | F1 점수, 정확 일치(exact match) |
| **효율성(Efficiency)** | 완료까지 몇 단계/토큰/비용이 필요한가? | 완료까지 단계 수, 총 토큰 수 |
| **궤적 품질(Trajectory Quality)** | 에이전트가 합리적인 경로를 택했는가? | 최적 단계 비율 |
| **도구 사용 정확도(Tool Use Accuracy)** | 도구를 올바르게 호출했는가? | 도구 정밀도/재현율(precision/recall) |
| **안전성(Safety)** | 에이전트가 유해한 행동을 피했는가? | 안전 위반율 |
| **견고성(Robustness)** | 엣지 케이스와 오류를 처리하는가? | 오류 복구율 |
| **지연 시간(Latency)** | 전체 작업에 얼마나 걸리는가? | 종단간(end-to-end) 초 단위 |

### 비결정성 문제 (The Non-Determinism Problem)

```python
from dataclasses import dataclass, field
import time
import hashlib
import json


@dataclass
class EvalRun:
    """A single evaluation run with metadata."""
    run_id: str
    task_id: str
    model: str
    success: bool
    steps: int
    total_tokens: int
    total_cost: float
    latency_seconds: float
    trajectory: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class AgentEvalRunner:
    """Run evaluations with statistical significance in mind."""

    def __init__(self, n_runs: int = 5):
        self.n_runs = n_runs
        self.results: list[EvalRun] = []

    def run_task(self, task_id: str, agent_fn, task_input: str,
                 expected_output: str, model: str = "claude-sonnet-4-20250514") -> list[EvalRun]:
        """Run the same task multiple times to account for non-determinism."""
        runs = []
        for i in range(self.n_runs):
            run_id = hashlib.md5(f"{task_id}-{i}-{time.time()}".encode()).hexdigest()[:8]

            start = time.time()
            try:
                result = agent_fn(task_input)
                success = self._check_success(result, expected_output)
                latency = time.time() - start

                run = EvalRun(
                    run_id=run_id,
                    task_id=task_id,
                    model=model,
                    success=success,
                    steps=result.get("steps", 0),
                    total_tokens=result.get("total_tokens", 0),
                    total_cost=result.get("total_cost", 0.0),
                    latency_seconds=latency,
                    trajectory=result.get("trajectory", []),
                )
            except Exception as e:
                run = EvalRun(
                    run_id=run_id,
                    task_id=task_id,
                    model=model,
                    success=False,
                    steps=0,
                    total_tokens=0,
                    total_cost=0.0,
                    latency_seconds=time.time() - start,
                    errors=[str(e)],
                )

            runs.append(run)
            self.results.append(run)

        return runs

    def _check_success(self, result: dict, expected: str) -> bool:
        """Check if the agent output matches expected output."""
        actual = result.get("output", "").strip().lower()
        return expected.strip().lower() in actual

    def aggregate(self, task_id: str) -> dict:
        """Aggregate results for a specific task across runs."""
        task_runs = [r for r in self.results if r.task_id == task_id]
        if not task_runs:
            return {}

        successes = sum(1 for r in task_runs if r.success)
        return {
            "task_id": task_id,
            "n_runs": len(task_runs),
            "success_rate": successes / len(task_runs),
            "avg_steps": sum(r.steps for r in task_runs) / len(task_runs),
            "avg_tokens": sum(r.total_tokens for r in task_runs) / len(task_runs),
            "avg_cost": sum(r.total_cost for r in task_runs) / len(task_runs),
            "avg_latency": sum(r.latency_seconds for r in task_runs) / len(task_runs),
            "error_count": sum(len(r.errors) for r in task_runs),
        }
```

---

## 2. AgentBench

### 이론: 주요 벤치마크

**C.1 AgentBench** (Liu 등, 2023). 8개 다양한 환경(OS, DB, 지식 그래프 등). 에이전트가 각각에서 작업 완료해야 함. 일반성을 스트레스 테스트 — 한 환경에서 이기는 에이전트가 다른 곳에서 심하게 실패할 수 있음.

**C.2 SWE-bench** (Jimenez 등, 2023). 인기 Python 저장소의 실제 GitHub 이슈. 에이전트가 이슈를 해결하고 프로젝트의 기존 테스트를 통과하는 패치를 만들어야 함. 어렵고, 현실적, 자동 검증. 코드 에이전트 능력의 참조 벤치마크.

**C.3 WebArena** (Zhou 등, 2023). 로컬에서 실행되는 실제 웹사이트의 현실적 웹 작업(쇼핑, GitLab, 콘텐츠 관리). 웹 브라우징, 폼 채우기, 다중 페이지 추론 테스트. 웹사이트가 컨테이너화되어 재현 가능.

**C.4 GAIA** (Mialon 등, 2023). 일반 어시스턴트 벤치마크 — 다단계 추론, 도구 사용, 웹 검색, 파일 처리를 요구하는 질문. 인간에게 쉬움(대부분 >90% 점수); 현재 LLM(도구가 있는 GPT-4)은 30-50% 점수. 능력 격차를 노출하도록 설계.

각 벤치마크가 다른 측면을 조사 — AgentBench는 폭에, SWE-bench는 코드에, WebArena는 웹에, GAIA는 일반 어시스턴스에. 단일 벤치마크가 모든 것을 포착하지 못합니다.

### 개요

> **AgentBench**
>
> 8개 환경에 걸쳐 LLM 에이전트를 평가하는 종합 벤치마크:
>
> | 환경 | 작업 유형 | 예시 |
> |------|----------|------|
> | 운영체제(Operating System) | 셸 명령어 | "1MB보다 큰 모든 .py 파일 찾기" |
> | 데이터베이스(Database) | SQL 쿼리 | "매출 상위 5명의 고객 찾기" |
> | 지식 그래프(Knowledge Graph) | SPARQL/Cypher | "놀란 감독의 모든 영화 찾기" |
> | 디지털 카드 게임(Digital Card Game) | 전략 | "카드 게임에서 한 턴 플레이" |
> | 수평적 사고(Lateral Thinking) | 퍼즐 | "스무고개" 스타일 추론 |
> | 가사 관리(House-Holding) | 시뮬레이션 작업 | "사과를 냉장고에 넣기" |
> | 웹 쇼핑(Web Shopping) | 전자상거래 | "30달러 미만의 빨간 셔츠 찾기" |
> | 웹 브라우징(Web Browsing) | 탐색 | "NYC에서 LA로 항공편 예약" |

### AgentBench 평가 구조

```python
from abc import ABC, abstractmethod
from enum import Enum


class EnvironmentType(Enum):
    OS = "operating_system"
    DATABASE = "database"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    WEB_SHOPPING = "web_shopping"
    WEB_BROWSING = "web_browsing"


@dataclass
class BenchmarkTask:
    """A single benchmark task definition."""
    task_id: str
    environment: EnvironmentType
    instruction: str
    expected_output: str
    max_steps: int = 20
    time_limit_seconds: float = 300.0
    difficulty: str = "medium"


class BenchmarkEnvironment(ABC):
    """Abstract environment for agent evaluation."""

    @abstractmethod
    def reset(self) -> str:
        """Reset environment and return initial observation."""
        ...

    @abstractmethod
    def step(self, action: str) -> tuple[str, bool, dict]:
        """Execute action. Returns (observation, done, info)."""
        ...

    @abstractmethod
    def evaluate(self, task: BenchmarkTask) -> dict:
        """Evaluate agent performance on the task."""
        ...


class SimulatedOSEnvironment(BenchmarkEnvironment):
    """Simulated OS environment for safe evaluation."""

    def __init__(self):
        self.filesystem: dict[str, str] = {
            "/home/user/data.csv": "id,name,value\n1,alpha,100\n2,beta,200",
            "/home/user/notes.txt": "Meeting at 3pm\nProject deadline Friday",
            "/home/user/scripts/process.py": "import pandas as pd\ndf = pd.read_csv('data.csv')",
        }
        self.cwd = "/home/user"
        self.history: list[str] = []

    def reset(self) -> str:
        self.history.clear()
        return f"Current directory: {self.cwd}"

    def step(self, action: str) -> tuple[str, bool, dict]:
        """Simulate shell command execution."""
        self.history.append(action)

        if action.startswith("ls"):
            path = action.split()[-1] if len(action.split()) > 1 else self.cwd
            files = [
                f.split("/")[-1] for f in self.filesystem
                if f.startswith(path)
            ]
            return "\n".join(files) if files else "No files found", False, {}

        elif action.startswith("cat"):
            filename = action.split()[-1]
            full_path = f"{self.cwd}/{filename}" if not filename.startswith("/") else filename
            content = self.filesystem.get(full_path, "File not found")
            return content, False, {}

        elif action.startswith("grep"):
            parts = action.split()
            pattern = parts[1] if len(parts) > 1 else ""
            results = []
            for path, content in self.filesystem.items():
                if pattern.lower() in content.lower():
                    results.append(f"{path}: {content.split(chr(10))[0]}")
            return "\n".join(results) if results else "No matches", False, {}

        return f"Command executed: {action}", False, {}

    def evaluate(self, task: BenchmarkTask) -> dict:
        return {
            "steps_taken": len(self.history),
            "commands_used": self.history,
        }
```

---

## 3. SWE-bench

### 소프트웨어 엔지니어링 벤치마크 (Software Engineering Benchmark)

> **SWE-bench**
>
> 실제 GitHub 이슈를 기반으로 에이전트를 평가한다. 에이전트는 다음을 수행해야 한다:
> 1. 이슈 설명 읽기
> 2. 코드베이스 이해하기
> 3. 이슈를 수정하는 패치 생성하기
> 4. 기존 테스트 스위트 통과하기
>
> **주요 통계:**
> - 12개의 Python 저장소에서 2,294개의 작업 인스턴스
> - SWE-bench Lite: 빠른 평가를 위한 300개의 선별된 인스턴스
> - SWE-bench Verified: 모호하지 않은 솔루션을 가진 인간 검증 하위 집합
> - 최고 에이전트가 SWE-bench Verified의 ~50%를 해결 (2025년 초 기준)

### SWE-bench 평가 프레임워크

```python
import subprocess
from pathlib import Path


@dataclass
class SWEBenchTask:
    """A SWE-bench task definition."""
    instance_id: str
    repo: str
    base_commit: str
    issue_text: str
    hints_text: str
    test_patch: str
    expected_patch: str | None = None


class SWEBenchEvaluator:
    """Evaluate patches generated by an agent for SWE-bench tasks."""

    def __init__(self, workspace_dir: str = "/tmp/swe-bench"):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def evaluate_patch(self, task: SWEBenchTask, agent_patch: str) -> dict:
        """Evaluate an agent-generated patch."""
        result = {
            "instance_id": task.instance_id,
            "patch_generated": bool(agent_patch),
            "patch_applies": False,
            "tests_pass": False,
            "exact_match": False,
        }

        if not agent_patch:
            return result

        # Check if patch is syntactically valid
        result["patch_applies"] = self._validate_patch_syntax(agent_patch)

        # Check test passage (simulated)
        if result["patch_applies"]:
            result["tests_pass"] = self._run_tests(task, agent_patch)

        # Check exact match with expected patch
        if task.expected_patch:
            result["exact_match"] = (
                self._normalize_patch(agent_patch)
                == self._normalize_patch(task.expected_patch)
            )

        return result

    def _validate_patch_syntax(self, patch: str) -> bool:
        """Check if the patch has valid diff syntax."""
        lines = patch.strip().split("\n")
        has_diff_header = any(l.startswith("diff ") or l.startswith("---") for l in lines)
        has_hunks = any(l.startswith("@@") for l in lines)
        return has_diff_header and has_hunks

    def _run_tests(self, task: SWEBenchTask, patch: str) -> bool:
        """Run the test suite against the patched code (simulated)."""
        # In real SWE-bench, this would:
        # 1. Clone the repo at base_commit
        # 2. Apply the agent's patch
        # 3. Apply the test patch
        # 4. Run pytest and check exit code
        return True  # Simulated

    def _normalize_patch(self, patch: str) -> str:
        """Normalize a patch for comparison."""
        lines = patch.strip().split("\n")
        # Remove timestamps and line numbers that may differ
        normalized = []
        for line in lines:
            if line.startswith("index ") or line.startswith("diff --git"):
                continue
            normalized.append(line.rstrip())
        return "\n".join(normalized)

    def compute_metrics(self, results: list[dict]) -> dict:
        """Compute aggregate SWE-bench metrics."""
        total = len(results)
        if total == 0:
            return {}

        resolved = sum(1 for r in results if r["tests_pass"])
        applied = sum(1 for r in results if r["patch_applies"])

        return {
            "total_instances": total,
            "patches_generated": sum(1 for r in results if r["patch_generated"]),
            "patches_apply": applied,
            "tests_pass": resolved,
            "resolve_rate": resolved / total,
            "apply_rate": applied / total,
        }
```

---

## 4. WebArena

### 웹 기반 에이전트 평가 (Web-Based Agent Evaluation)

> **WebArena**
>
> 자체 호스팅 웹사이트를 활용한 사실적인 웹 환경 벤치마크:
>
> | 사이트 유형 | 플랫폼 | 예시 작업 |
> |-----------|--------|----------|
> | 전자상거래(E-commerce) | 원스톱 쇼핑몰 | "16GB RAM 노트북 중 가장 저렴한 것 찾기" |
> | 포럼(Forum) | Reddit 유사 | "r/cooking의 인기 스레드에 답글 작성" |
> | CMS | GitLab 유사 | "프론트엔드 저장소에 이슈 생성" |
> | 지도(Maps) | OpenStreetMap | "A에서 B까지 운전 경로 찾기" |
> | 위키(Wiki) | Wikipedia 유사 | "머신러닝 문서 편집" |
>
> **주요 특징:**
> - 5개 웹사이트에 걸친 812개의 인간 검증 작업
> - 기능적 정확성 평가 (단순 행동 매칭이 아님)
> - 계획이 필요한 다중 탭, 다단계 작업

### 웹 에이전트 평가 (Web Agent Evaluation)

```python
from dataclasses import dataclass, field


@dataclass
class WebAction:
    """A web interaction action."""
    action_type: str  # "click", "type", "scroll", "navigate", "select"
    target: str       # CSS selector or element description
    value: str = ""   # For type/select actions
    timestamp: float = field(default_factory=time.time)


@dataclass
class WebTrajectory:
    """Complete trajectory of web interactions."""
    task_id: str
    actions: list[WebAction] = field(default_factory=list)
    pages_visited: list[str] = field(default_factory=list)
    final_state: dict = field(default_factory=dict)

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    @property
    def unique_pages(self) -> int:
        return len(set(self.pages_visited))


class WebArenaEvaluator:
    """Evaluate web agent trajectories."""

    def __init__(self):
        self.task_registry: dict[str, dict] = {}

    def register_task(self, task_id: str, description: str,
                      success_criteria: dict, max_steps: int = 30):
        """Register a WebArena task."""
        self.task_registry[task_id] = {
            "description": description,
            "success_criteria": success_criteria,
            "max_steps": max_steps,
        }

    def evaluate_trajectory(self, trajectory: WebTrajectory) -> dict:
        """Evaluate a web agent's trajectory."""
        task = self.task_registry.get(trajectory.task_id, {})
        criteria = task.get("success_criteria", {})

        result = {
            "task_id": trajectory.task_id,
            "num_actions": trajectory.num_actions,
            "unique_pages": trajectory.unique_pages,
            "within_step_limit": (
                trajectory.num_actions <= task.get("max_steps", 30)
            ),
        }

        # Check success criteria
        result["criteria_met"] = {}
        all_met = True

        for criterion, expected in criteria.items():
            if criterion == "final_url":
                met = trajectory.pages_visited[-1] == expected if trajectory.pages_visited else False
            elif criterion == "element_present":
                met = expected in str(trajectory.final_state)
            elif criterion == "form_submitted":
                met = any(
                    a.action_type == "click" and "submit" in a.target.lower()
                    for a in trajectory.actions
                )
            else:
                met = False

            result["criteria_met"][criterion] = met
            if not met:
                all_met = False

        result["success"] = all_met
        result["efficiency"] = self._compute_efficiency(trajectory, task)
        return result

    def _compute_efficiency(self, trajectory: WebTrajectory, task: dict) -> float:
        """Compute efficiency score (0 to 1). Fewer steps = better."""
        max_steps = task.get("max_steps", 30)
        if trajectory.num_actions == 0:
            return 0.0
        # Optimal is ~3-5 steps; penalize excessive actions
        optimal_steps = 5
        ratio = optimal_steps / max(trajectory.num_actions, 1)
        return min(1.0, ratio)


# Usage
evaluator = WebArenaEvaluator()
evaluator.register_task(
    "shop-001",
    "Find the cheapest laptop with 16GB RAM on the e-commerce site",
    success_criteria={
        "final_url": "/product/laptop-budget-16gb",
        "element_present": "Add to Cart",
    },
    max_steps=15,
)

# Simulate a trajectory
trajectory = WebTrajectory(
    task_id="shop-001",
    actions=[
        WebAction("navigate", "/", ""),
        WebAction("click", ".category-laptops", ""),
        WebAction("select", "#ram-filter", "16GB"),
        WebAction("click", ".sort-by-price", ""),
        WebAction("click", ".product-card:first-child", ""),
    ],
    pages_visited=["/", "/laptops", "/laptops?ram=16gb", "/laptops?ram=16gb&sort=price",
                   "/product/laptop-budget-16gb"],
    final_state={"page_title": "Budget Laptop 16GB", "buttons": ["Add to Cart", "Wishlist"]},
)

result = evaluator.evaluate_trajectory(trajectory)
print(f"Task: {result['task_id']}")
print(f"Success: {result['success']}")
print(f"Actions: {result['num_actions']}")
print(f"Efficiency: {result['efficiency']:.2f}")
```

---

## 5. GAIA 벤치마크

### 범용 AI 어시스턴트 벤치마크 (General AI Assistants Benchmark)

> **GAIA**
>
> 여러 능력을 요구하는 실세계 작업에서 에이전트를 테스트한다:
>
> | 레벨 | 복잡도 | 예시 |
> |------|--------|------|
> | 레벨 1 | 단일 도구, 1-3 단계 | "프랑스의 인구는?" |
> | 레벨 2 | 다중 도구, 3-8 단계 | "최근 5년간 미국과 중국의 GDP 성장률 비교" |
> | 레벨 3 | 복합 추론 + 도구 | "예산과 리뷰를 최적화하여 3일 여행 계획하기" |
>
> **GAIA가 중요한 이유:**
> - 명확한 답이 있는 466개의 인간 작성 질문
> - 합성 벤치마크가 아닌 실세계 능력 테스트
> - 인간 점수 ~92%; 최고 에이전트 점수 ~75% (레벨 1) 및 ~40% (레벨 3)
> - "텍스트를 생성할 수 있다"와 "실제로 도움이 된다" 사이의 격차를 측정

### GAIA 스타일 평가

```python
@dataclass
class GAIATask:
    """A GAIA benchmark task."""
    task_id: str
    question: str
    expected_answer: str
    level: int  # 1, 2, or 3
    required_tools: list[str]  # Tools needed to solve the task
    file_attachment: str | None = None


class GAIAEvaluator:
    """Evaluate agent responses on GAIA-style tasks."""

    def __init__(self):
        self.tasks: list[GAIATask] = []

    def add_task(self, task: GAIATask):
        self.tasks.append(task)

    def evaluate_answer(self, task: GAIATask, agent_answer: str) -> dict:
        """Evaluate correctness of the agent's answer."""
        expected = task.expected_answer.strip().lower()
        actual = agent_answer.strip().lower()

        # Exact match
        exact = actual == expected

        # Contains match (for numeric or short answers embedded in text)
        contains = expected in actual

        # Numeric tolerance (for numerical answers)
        numeric_match = False
        try:
            expected_num = float(expected.replace(",", "").replace("$", ""))
            actual_num = float(actual.replace(",", "").replace("$", ""))
            numeric_match = abs(expected_num - actual_num) / max(abs(expected_num), 1e-10) < 0.05
        except (ValueError, ZeroDivisionError):
            pass

        return {
            "task_id": task.task_id,
            "level": task.level,
            "correct": exact or contains or numeric_match,
            "exact_match": exact,
            "contains_match": contains,
            "numeric_match": numeric_match,
            "expected": task.expected_answer,
            "actual": agent_answer[:200],
        }

    def compute_metrics(self, results: list[dict]) -> dict:
        """Compute GAIA metrics by level."""
        by_level = {1: [], 2: [], 3: []}
        for r in results:
            by_level[r["level"]].append(r)

        metrics = {"overall": {}, "by_level": {}}

        # Overall
        total_correct = sum(1 for r in results if r["correct"])
        metrics["overall"] = {
            "total": len(results),
            "correct": total_correct,
            "accuracy": total_correct / len(results) if results else 0,
        }

        # By level
        for level, level_results in by_level.items():
            if not level_results:
                continue
            correct = sum(1 for r in level_results if r["correct"])
            metrics["by_level"][f"level_{level}"] = {
                "total": len(level_results),
                "correct": correct,
                "accuracy": correct / len(level_results),
            }

        return metrics
```

---

## 6. 평가 방법론

### 이론: 작업 성공과 부분 점수

개방형 작업에 대해 "성공"을 정의하는 것은 그 자체가 연구 문제입니다. 세 패턴:

- **이진** — 통과 또는 실패. 명확하고 모호하지 않은 정답이 있을 때 적합(코드 패치, SQL 쿼리, 수학 답).
- **등급** — 루브릭에서 0-1 점수(LLM-as-judge 또는 자동 검사). 미묘한 작업(쓰기, 요약)에 적합.
- **다중 목적** — 정확성, 효율성, 안전성 등에 대한 별도 점수. 단일 점수로 결합은 신중히 — 평균이 트레이드오프를 숨김.

벤치마크에 대해 — **자동 검증성**이 벤치마크를 확장 가능하게 만드는 것입니다. SWE-bench가 작동하는 이유는 패치를 테스트할 수 있기 때문. WebArena가 작동하는 이유는 최종 웹 상태가 검사 가능하기 때문. 성공이 인간 판단을 요구하는 작업은 확장이 빈약합니다.

### 작업 완료 평가 (Task Completion Evaluation)

```python
class TaskCompletionEvaluator:
    """Evaluate whether an agent successfully completed its task."""

    def __init__(self):
        self.evaluations: list[dict] = []

    def evaluate(self, task_description: str, agent_output: dict,
                 ground_truth: dict) -> dict:
        """Multi-dimensional task completion evaluation."""
        result = {
            "task": task_description,
            "dimensions": {},
        }

        # 1. Output correctness
        result["dimensions"]["correctness"] = self._check_correctness(
            agent_output.get("answer", ""),
            ground_truth.get("answer", ""),
        )

        # 2. Step efficiency
        actual_steps = agent_output.get("steps", 0)
        optimal_steps = ground_truth.get("optimal_steps", actual_steps)
        result["dimensions"]["efficiency"] = {
            "actual_steps": actual_steps,
            "optimal_steps": optimal_steps,
            "ratio": optimal_steps / max(actual_steps, 1),
            "grade": "optimal" if actual_steps <= optimal_steps
                     else "acceptable" if actual_steps <= optimal_steps * 2
                     else "inefficient",
        }

        # 3. Tool usage accuracy
        expected_tools = set(ground_truth.get("tools_used", []))
        actual_tools = set(agent_output.get("tools_used", []))
        result["dimensions"]["tool_accuracy"] = {
            "precision": (
                len(expected_tools & actual_tools) / len(actual_tools)
                if actual_tools else 0
            ),
            "recall": (
                len(expected_tools & actual_tools) / len(expected_tools)
                if expected_tools else 1.0
            ),
            "unnecessary_tools": list(actual_tools - expected_tools),
            "missed_tools": list(expected_tools - actual_tools),
        }

        # 4. Safety check
        result["dimensions"]["safety"] = self._check_safety(
            agent_output.get("trajectory", [])
        )

        # Overall score
        scores = []
        if result["dimensions"]["correctness"]["correct"]:
            scores.append(1.0)
        else:
            scores.append(0.0)
        scores.append(result["dimensions"]["efficiency"]["ratio"])
        tool_f1 = (
            2 * result["dimensions"]["tool_accuracy"]["precision"]
            * result["dimensions"]["tool_accuracy"]["recall"]
            / max(
                result["dimensions"]["tool_accuracy"]["precision"]
                + result["dimensions"]["tool_accuracy"]["recall"],
                1e-10,
            )
        )
        scores.append(tool_f1)
        scores.append(1.0 if result["dimensions"]["safety"]["safe"] else 0.0)

        result["overall_score"] = sum(scores) / len(scores)
        self.evaluations.append(result)
        return result

    def _check_correctness(self, actual: str, expected: str) -> dict:
        actual_lower = actual.strip().lower()
        expected_lower = expected.strip().lower()
        return {
            "correct": expected_lower in actual_lower or actual_lower == expected_lower,
            "expected": expected[:100],
            "actual": actual[:100],
        }

    def _check_safety(self, trajectory: list[dict]) -> dict:
        """Check trajectory for safety violations."""
        unsafe_patterns = [
            "rm -rf", "DROP TABLE", "DELETE FROM",
            "sudo", "chmod 777", "format",
            "password", "credit_card", "ssn",
        ]

        violations = []
        for step in trajectory:
            action = str(step.get("action", "")).lower()
            for pattern in unsafe_patterns:
                if pattern.lower() in action:
                    violations.append({
                        "step": step.get("step_number", "?"),
                        "action": action[:100],
                        "pattern": pattern,
                    })

        return {
            "safe": len(violations) == 0,
            "violation_count": len(violations),
            "violations": violations[:5],
        }
```

---

## 7. 실패 모드 분석

### 이론: 실패 모드 분류

에이전트가 실패하는 흔한 방식:

**D.1 무한 루프.** 에이전트가 같은 도구를 같은 인자로 반복 호출. 감지 — 과거 호출을 해시 비교, 반복에 경고.

**D.2 환각된 도구 또는 인자.** 에이전트가 도구 이름을 발명, 실제 도구를 가상의 파라미터로 호출. 감지 — 스키마 검증(function calling이 API 수준에서 이를 거의 불가능하게 함).

**D.3 조기 중단.** 에이전트가 실제로 작업을 완료하기 전 "작업 완료"를 선언. 감지 — 결과 검증, 에이전트가 완료를 정당화하도록 강제.

**D.4 도구 오용.** 에이전트가 옳은 도구를 잘못된 인자로 호출 — 사용자가 "Python the language"를 물었을 때 "Python"을 검색. 감지 — 더 어려움; 보통 궤적에 LLM-as-judge 필요.

**D.5 비용 통제 불능.** 에이전트가 비생산적 탐색에 갇혀 토큰과 도구 호출을 쌓음. 감지 — 단계와 예산에 단단한 상한.

**D.6 안전 위반.** 에이전트가 적절한 권한 없이 파괴적 도구(삭제, 전송, 결제) 호출. 감지 — 위험이 큰 도구에 human-in-the-loop, 나머지에 출력 필터링.

견고한 에이전트 평가는 각 실패를 이(또는 도메인 특화) 클래스로 분류 — 수정 우선순위를 매길 수 있도록.

### 일반적인 에이전트 실패 모드 (Common Agent Failure Modes)

> **에이전트 실패 분류 체계 (Agent Failure Taxonomy)**
>
> | 실패 모드 | 설명 | 빈도 |
> |----------|------|------|
> | **무한 루프(Infinite Loop)** | 에이전트가 같은 행동을 반복 | 높음 |
> | **환각된 행동(Hallucinated Action)** | 존재하지 않는 도구/API를 만들어냄 | 높음 |
> | **잘못된 도구 선택(Wrong Tool Selection)** | 작업에 부적절한 도구를 사용 | 중간 |
> | **조기 종료(Premature Termination)** | 작업 완료 전에 중단 | 중간 |
> | **컨텍스트 손실(Context Loss)** | 작업 중 이전 정보를 잊어버림 | 중간 |
> | **오류 연쇄(Error Cascade)** | 하나의 실패가 후속 실패를 유발 | 중간 |
> | **목표 이탈(Goal Drift)** | 의도와 다른 목표를 추구 | 낮음 |
> | **안전 위반(Safety Violation)** | 유해하거나 비인가된 행동을 수행 | 낮음 |

### 실패 탐지기 (Failure Detector)

```python
from collections import Counter


class AgentFailureDetector:
    """Detect common failure modes in agent trajectories."""

    def __init__(self, max_repeated_actions: int = 3,
                 max_steps: int = 50):
        self.max_repeated_actions = max_repeated_actions
        self.max_steps = max_steps

    def analyze(self, trajectory: list[dict]) -> dict:
        """Analyze a trajectory for failure modes."""
        failures = {
            "loop_detected": self._detect_loop(trajectory),
            "hallucinated_actions": self._detect_hallucinations(trajectory),
            "wrong_tool_usage": self._detect_wrong_tools(trajectory),
            "premature_termination": self._detect_premature_stop(trajectory),
            "context_loss": self._detect_context_loss(trajectory),
            "error_cascade": self._detect_error_cascade(trajectory),
        }

        failures["total_issues"] = sum(
            1 for v in failures.values()
            if isinstance(v, dict) and v.get("detected", False)
        )
        failures["severity"] = (
            "critical" if failures["total_issues"] >= 3
            else "warning" if failures["total_issues"] >= 1
            else "clean"
        )

        return failures

    def _detect_loop(self, trajectory: list[dict]) -> dict:
        """Detect repeated action sequences."""
        actions = [str(step.get("action", "")) for step in trajectory]

        # Check for exact repeated actions
        action_counts = Counter(actions)
        repeated = {
            action: count for action, count in action_counts.items()
            if count >= self.max_repeated_actions
        }

        # Check for repeated action patterns (bigrams)
        bigrams = [
            f"{actions[i]}|{actions[i+1]}"
            for i in range(len(actions) - 1)
        ]
        bigram_counts = Counter(bigrams)
        repeated_patterns = {
            bg: count for bg, count in bigram_counts.items()
            if count >= self.max_repeated_actions
        }

        detected = bool(repeated or repeated_patterns)
        return {
            "detected": detected,
            "repeated_actions": repeated,
            "repeated_patterns": repeated_patterns,
        }

    def _detect_hallucinations(self, trajectory: list[dict]) -> dict:
        """Detect actions referencing non-existent tools or APIs."""
        available_tools = set()
        hallucinated = []

        for step in trajectory:
            tool = step.get("tool", "")
            available = step.get("available_tools", [])
            if available:
                available_tools.update(available)

            if tool and available_tools and tool not in available_tools:
                hallucinated.append({
                    "step": step.get("step_number", "?"),
                    "tool": tool,
                })

        return {
            "detected": len(hallucinated) > 0,
            "hallucinated_tools": hallucinated,
        }

    def _detect_wrong_tools(self, trajectory: list[dict]) -> dict:
        """Detect tool usage that doesn't match the subtask."""
        mismatches = []
        for step in trajectory:
            tool = step.get("tool", "")
            subtask = step.get("subtask", "").lower()
            error = step.get("error", "")

            # Heuristic: if a tool call resulted in an error about wrong input
            if error and ("invalid" in error.lower() or "unexpected" in error.lower()):
                mismatches.append({
                    "step": step.get("step_number", "?"),
                    "tool": tool,
                    "error": error[:100],
                })

        return {
            "detected": len(mismatches) > 0,
            "mismatches": mismatches,
        }

    def _detect_premature_stop(self, trajectory: list[dict]) -> dict:
        """Detect if the agent stopped without completing the task."""
        if not trajectory:
            return {"detected": True, "reason": "empty trajectory"}

        last_step = trajectory[-1]
        has_final_answer = "final_answer" in last_step or last_step.get("action") == "finish"

        return {
            "detected": not has_final_answer and len(trajectory) < self.max_steps,
            "last_action": str(last_step.get("action", ""))[:100],
            "total_steps": len(trajectory),
        }

    def _detect_context_loss(self, trajectory: list[dict]) -> dict:
        """Detect if the agent re-asks for information it already has."""
        seen_info = set()
        re_requests = []

        for step in trajectory:
            # Track information the agent received
            observation = str(step.get("observation", "")).lower()
            for word in observation.split():
                if len(word) > 5:
                    seen_info.add(word)

            # Check if the agent is asking for something it already knows
            action_text = str(step.get("action", "")).lower()
            if "what is" in action_text or "find" in action_text:
                query_words = set(action_text.split())
                overlap = query_words & seen_info
                if len(overlap) > 3:
                    re_requests.append({
                        "step": step.get("step_number", "?"),
                        "repeated_info": list(overlap)[:5],
                    })

        return {
            "detected": len(re_requests) > 0,
            "re_requests": re_requests,
        }

    def _detect_error_cascade(self, trajectory: list[dict]) -> dict:
        """Detect consecutive errors (error cascade)."""
        consecutive_errors = 0
        max_consecutive = 0
        cascade_start = -1

        for i, step in enumerate(trajectory):
            if step.get("error"):
                consecutive_errors += 1
                if consecutive_errors > max_consecutive:
                    max_consecutive = consecutive_errors
                    cascade_start = i - consecutive_errors + 1
            else:
                consecutive_errors = 0

        return {
            "detected": max_consecutive >= 3,
            "max_consecutive_errors": max_consecutive,
            "cascade_start_step": cascade_start if max_consecutive >= 3 else None,
        }
```

---

## 8. 에이전트 관측 가능성과 추적

### 이론: 궤적 분석

로깅 구조가 중요합니다. 유용한 에이전트 로그는 단계당 다음을 포함:

- 단계 번호와 타임스탬프.
- 생각(LLM 추론).
- 행동(도구 이름 + 인자).
- 관측(도구 결과, 가능하게 잘림).
- 소비된 토큰, 비용, 지연.
- 선택적으로 — "이 단계가 생산적이었나?"에 대한 단계별 LLM-as-judge 점수.

LangSmith, Phoenix, Langfuse, Helicone(레슨 24) 같은 도구가 이 구조로 에이전트를 자동 계측. 궤적 로그 없이 실패한 에이전트 실행을 디버그하는 것은 본질적으로 불가능.

### 구조화된 추적 수집 (Structured Trace Collection)

```python
from contextlib import contextmanager
import uuid


@dataclass
class TraceSpan:
    """A single span in a distributed trace."""
    span_id: str
    parent_id: str | None
    operation: str
    start_time: float
    end_time: float | None = None
    status: str = "in_progress"
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0
        return (self.end_time - self.start_time) * 1000


class AgentTracer:
    """Collect structured traces from agent execution."""

    def __init__(self):
        self.trace_id = str(uuid.uuid4())
        self.spans: list[TraceSpan] = []
        self._span_stack: list[str] = []

    @contextmanager
    def span(self, operation: str, **attributes):
        """Context manager for tracing a span."""
        span_id = str(uuid.uuid4())[:8]
        parent_id = self._span_stack[-1] if self._span_stack else None

        trace_span = TraceSpan(
            span_id=span_id,
            parent_id=parent_id,
            operation=operation,
            start_time=time.time(),
            attributes=attributes,
        )

        self._span_stack.append(span_id)
        self.spans.append(trace_span)

        try:
            yield trace_span
        except Exception as e:
            trace_span.status = "error"
            trace_span.events.append({
                "type": "exception",
                "message": str(e),
                "timestamp": time.time(),
            })
            raise
        finally:
            trace_span.end_time = time.time()
            trace_span.status = trace_span.status if trace_span.status == "error" else "ok"
            self._span_stack.pop()

    def add_event(self, name: str, **attributes):
        """Add an event to the current span."""
        if self._span_stack:
            current_span_id = self._span_stack[-1]
            for span in self.spans:
                if span.span_id == current_span_id:
                    span.events.append({
                        "name": name,
                        "timestamp": time.time(),
                        **attributes,
                    })
                    break

    def get_trace_summary(self) -> dict:
        """Summarize the trace."""
        total_duration = sum(s.duration_ms for s in self.spans)
        errors = [s for s in self.spans if s.status == "error"]

        return {
            "trace_id": self.trace_id,
            "total_spans": len(self.spans),
            "total_duration_ms": round(total_duration, 2),
            "error_spans": len(errors),
            "operations": [s.operation for s in self.spans],
            "span_tree": self._build_tree(),
        }

    def _build_tree(self, parent_id: str | None = None, depth: int = 0) -> list[dict]:
        """Build a hierarchical trace tree."""
        children = [s for s in self.spans if s.parent_id == parent_id]
        tree = []
        for span in children:
            node = {
                "indent": "  " * depth,
                "operation": span.operation,
                "duration_ms": round(span.duration_ms, 1),
                "status": span.status,
                "children": self._build_tree(span.span_id, depth + 1),
            }
            tree.append(node)
        return tree


# Usage example
tracer = AgentTracer()

with tracer.span("agent_run", goal="Answer user question"):
    with tracer.span("planning", strategy="task_decomposition"):
        time.sleep(0.01)  # Simulated planning
        tracer.add_event("plan_created", steps=3)

    with tracer.span("tool_call", tool="search", query="population of France"):
        time.sleep(0.02)  # Simulated tool call
        tracer.add_event("tool_result", tokens=150)

    with tracer.span("generation", model="claude-sonnet-4-20250514"):
        time.sleep(0.01)  # Simulated generation

summary = tracer.get_trace_summary()
print(f"Trace: {summary['trace_id'][:8]}...")
print(f"Spans: {summary['total_spans']}")
print(f"Duration: {summary['total_duration_ms']:.1f}ms")
print(f"Errors: {summary['error_spans']}")
```

### OpenTelemetry 통합

```python
# Production observability setup with OpenTelemetry
# (This shows the integration pattern; requires opentelemetry packages)

def setup_agent_observability():
    """Configure OpenTelemetry for agent tracing."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    provider = TracerProvider()
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    return trace.get_tracer("agent-service")


def traced_agent_call(tracer, model: str, messages: list[dict]) -> str:
    """Make an LLM call with OpenTelemetry tracing."""
    with tracer.start_as_current_span("llm_call") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.message_count", len(messages))

        import anthropic
        client = anthropic.Anthropic()

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=messages,
        )

        span.set_attribute("llm.input_tokens", response.usage.input_tokens)
        span.set_attribute("llm.output_tokens", response.usage.output_tokens)
        span.set_attribute("llm.stop_reason", response.stop_reason)

        return response.content[0].text
```

---

## 9. 비용-품질 분석

### 이론: 비용-품질 프런티어

같은 성공률을 가진 두 에이전트가 매우 다른 비용을 가질 수 있습니다:

- 에이전트 A — 80% 성공, 작업당 $0.10.
- 에이전트 B — 80% 성공, 작업당 $1.50.

같은 결과 지표, 15배 비용 차이. B가 다른 이점(지연, 안전성 등)을 가지지 않는 한 A가 엄밀히 더 낫습니다. 에이전트 평가는 품질과 함께 비용을 보고해야 합니다.

각 에이전트에 대해 성공 vs 비용을 그리고 파레토 프런티어를 찾으세요. 어느 에이전트를 배포할지에 대한 결정은 애플리케이션의 비용 민감도에 의존합니다. 백오피스 데이터 추출 도구는 비싼 에이전트를 감당할 수 있지만, 실시간 채팅 동반자는 그렇지 못합니다.

### 비용 추적 (Cost Tracking)

```python
from collections import defaultdict


class AgentCostTracker:
    """Track and analyze agent costs per task."""

    PRICING = {
        # Per 1M tokens (input, output)
        "claude-sonnet-4-20250514": (3.00, 15.00),
        "claude-haiku-4-20250514": (0.25, 1.25),
        "gpt-4o": (2.50, 10.00),
        "gpt-4o-mini": (0.15, 0.60),
    }

    def __init__(self):
        self.tasks: list[dict] = []

    def record(self, task_id: str, model: str,
               input_tokens: int, output_tokens: int,
               success: bool, quality_score: float):
        """Record a task execution with cost and quality."""
        input_rate, output_rate = self.PRICING.get(model, (5.0, 15.0))
        cost = (
            input_tokens * input_rate / 1_000_000
            + output_tokens * output_rate / 1_000_000
        )

        self.tasks.append({
            "task_id": task_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "success": success,
            "quality_score": quality_score,
        })

    def cost_quality_report(self) -> dict:
        """Generate a cost-quality analysis report."""
        if not self.tasks:
            return {}

        total_cost = sum(t["cost_usd"] for t in self.tasks)
        total_tasks = len(self.tasks)
        successes = sum(1 for t in self.tasks if t["success"])
        avg_quality = sum(t["quality_score"] for t in self.tasks) / total_tasks

        # Cost per successful task
        cost_per_success = total_cost / max(successes, 1)

        # Cost by model
        by_model = defaultdict(lambda: {"cost": 0, "tasks": 0, "quality_sum": 0})
        for t in self.tasks:
            by_model[t["model"]]["cost"] += t["cost_usd"]
            by_model[t["model"]]["tasks"] += 1
            by_model[t["model"]]["quality_sum"] += t["quality_score"]

        model_breakdown = {}
        for model, stats in by_model.items():
            model_breakdown[model] = {
                "total_cost": round(stats["cost"], 4),
                "task_count": stats["tasks"],
                "avg_quality": round(stats["quality_sum"] / stats["tasks"], 2),
                "cost_per_task": round(stats["cost"] / stats["tasks"], 4),
            }

        return {
            "total_cost_usd": round(total_cost, 4),
            "total_tasks": total_tasks,
            "success_rate": successes / total_tasks,
            "avg_quality": round(avg_quality, 2),
            "cost_per_success": round(cost_per_success, 4),
            "quality_per_dollar": round(avg_quality / max(total_cost, 0.0001), 2),
            "by_model": model_breakdown,
        }


# Usage
tracker = AgentCostTracker()

# Simulate tasks with different models
tasks = [
    ("task-1", "claude-sonnet-4-20250514", 2000, 800, True, 0.9),
    ("task-2", "claude-sonnet-4-20250514", 3000, 1200, True, 0.85),
    ("task-3", "claude-haiku-4-20250514", 1500, 600, True, 0.7),
    ("task-4", "claude-haiku-4-20250514", 1000, 400, False, 0.3),
    ("task-5", "gpt-4o-mini", 800, 300, True, 0.75),
]

for task_id, model, inp, out, success, quality in tasks:
    tracker.record(task_id, model, inp, out, success, quality)

report = tracker.cost_quality_report()
print(f"Total cost: ${report['total_cost_usd']:.4f}")
print(f"Success rate: {report['success_rate']:.0%}")
print(f"Avg quality: {report['avg_quality']}")
print(f"Cost per success: ${report['cost_per_success']:.4f}")
print(f"Quality per dollar: {report['quality_per_dollar']}")
print("\nBy model:")
for model, stats in report["by_model"].items():
    print(f"  {model}: ${stats['total_cost']:.4f} "
          f"({stats['task_count']} tasks, quality={stats['avg_quality']})")
```

---

## 10. 맞춤형 에이전트 평가 구축

### 이론: 커스텀 평가 구축

특정 애플리케이션에 대해 공개 벤치마크는 필수지만 충분하지 않습니다. 커스텀 평가:

1. **실제 프로덕션 트래픽을 대표하는 작업 정의.** 실제 사용자 쿼리(프라이버시 제어와 함께) 표본 추출 또는 사용 사례 기반 합성 작성.
2. **정답 확립.** 각 작업에 대해 정확한 결과는 무엇인가? 수동 라벨링(비싸지만 황금 표준) 또는 LLM-as-judge(저렴, 대부분에 충분).
3. **평가 셋에 후보 에이전트 실행.**
4. **지표 계산** — 성공률, 비용, 지연, 실패 모드 분포.
5. **시간에 따라 추적.** 회귀를 잡기 위해 모든 릴리스에서 재실행.

표준 패턴 — 모든 커밋에 대한 작은 "smoke test" 평가(10-50 작업, 분 단위 실행), 릴리스에 대한 더 큰 "포괄적" 평가(200-1000 작업, 시간 단위 실행).

### 평가 프레임워크 (Evaluation Framework)

```python
from abc import ABC, abstractmethod
from typing import Callable
import json


class EvalCase:
    """A single evaluation test case."""

    def __init__(self, case_id: str, input_data: dict,
                 expected: dict, tags: list[str] | None = None):
        self.case_id = case_id
        self.input_data = input_data
        self.expected = expected
        self.tags = tags or []


class EvalMetric(ABC):
    """Abstract evaluation metric."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def compute(self, predicted: dict, expected: dict) -> float:
        """Return a score between 0 and 1."""
        ...


class ExactMatchMetric(EvalMetric):
    @property
    def name(self) -> str:
        return "exact_match"

    def compute(self, predicted: dict, expected: dict) -> float:
        pred_answer = str(predicted.get("answer", "")).strip().lower()
        exp_answer = str(expected.get("answer", "")).strip().lower()
        return 1.0 if pred_answer == exp_answer else 0.0


class ContainsMetric(EvalMetric):
    @property
    def name(self) -> str:
        return "contains"

    def compute(self, predicted: dict, expected: dict) -> float:
        pred_answer = str(predicted.get("answer", "")).lower()
        exp_answer = str(expected.get("answer", "")).lower()
        return 1.0 if exp_answer in pred_answer else 0.0


class StepEfficiencyMetric(EvalMetric):
    @property
    def name(self) -> str:
        return "step_efficiency"

    def compute(self, predicted: dict, expected: dict) -> float:
        actual = predicted.get("steps", 1)
        optimal = expected.get("optimal_steps", actual)
        if actual == 0:
            return 0.0
        return min(1.0, optimal / actual)


class ToolAccuracyMetric(EvalMetric):
    @property
    def name(self) -> str:
        return "tool_accuracy"

    def compute(self, predicted: dict, expected: dict) -> float:
        pred_tools = set(predicted.get("tools_used", []))
        exp_tools = set(expected.get("tools_used", []))
        if not exp_tools:
            return 1.0 if not pred_tools else 0.5
        return len(pred_tools & exp_tools) / len(pred_tools | exp_tools)


class AgentEvalSuite:
    """Complete evaluation suite for agent systems."""

    def __init__(self, name: str):
        self.name = name
        self.cases: list[EvalCase] = []
        self.metrics: list[EvalMetric] = []
        self.results: list[dict] = []

    def add_case(self, case: EvalCase):
        self.cases.append(case)

    def add_metric(self, metric: EvalMetric):
        self.metrics.append(metric)

    def run(self, agent_fn: Callable[[dict], dict],
            verbose: bool = False) -> dict:
        """Run the eval suite against an agent function."""
        self.results = []

        for case in self.cases:
            try:
                predicted = agent_fn(case.input_data)
            except Exception as e:
                predicted = {"error": str(e)}

            case_result = {
                "case_id": case.case_id,
                "tags": case.tags,
                "scores": {},
            }

            for metric in self.metrics:
                score = metric.compute(predicted, case.expected)
                case_result["scores"][metric.name] = round(score, 4)

            self.results.append(case_result)

            if verbose:
                avg_score = sum(case_result["scores"].values()) / max(len(case_result["scores"]), 1)
                status = "PASS" if avg_score >= 0.7 else "FAIL"
                print(f"  [{status}] {case.case_id}: {case_result['scores']}")

        return self._aggregate()

    def _aggregate(self) -> dict:
        """Aggregate results across all cases."""
        if not self.results:
            return {}

        # Average score per metric
        metric_scores = defaultdict(list)
        for r in self.results:
            for metric_name, score in r["scores"].items():
                metric_scores[metric_name].append(score)

        avg_scores = {
            name: round(sum(scores) / len(scores), 4)
            for name, scores in metric_scores.items()
        }

        # Pass rate (>= 0.7 average across all metrics)
        pass_count = 0
        for r in self.results:
            avg = sum(r["scores"].values()) / max(len(r["scores"]), 1)
            if avg >= 0.7:
                pass_count += 1

        # By tag
        tag_results = defaultdict(list)
        for r in self.results:
            for tag in r.get("tags", []):
                avg = sum(r["scores"].values()) / max(len(r["scores"]), 1)
                tag_results[tag].append(avg)

        by_tag = {
            tag: round(sum(scores) / len(scores), 4)
            for tag, scores in tag_results.items()
        }

        return {
            "suite": self.name,
            "total_cases": len(self.results),
            "pass_rate": pass_count / len(self.results),
            "avg_scores": avg_scores,
            "by_tag": by_tag,
        }


# Build a custom eval suite
suite = AgentEvalSuite("customer-support-agent")

# Add metrics
suite.add_metric(ExactMatchMetric())
suite.add_metric(ContainsMetric())
suite.add_metric(StepEfficiencyMetric())
suite.add_metric(ToolAccuracyMetric())

# Add test cases
suite.add_case(EvalCase(
    "cs-001",
    input_data={"question": "What is the refund policy?"},
    expected={
        "answer": "30-day refund policy",
        "optimal_steps": 2,
        "tools_used": ["knowledge_base"],
    },
    tags=["policy", "easy"],
))

suite.add_case(EvalCase(
    "cs-002",
    input_data={"question": "I was charged twice for order #12345"},
    expected={
        "answer": "refund processed",
        "optimal_steps": 4,
        "tools_used": ["order_lookup", "payment_system"],
    },
    tags=["billing", "medium"],
))

# Simulate running against an agent
def mock_agent(input_data: dict) -> dict:
    """Simulated agent for testing."""
    q = input_data.get("question", "").lower()
    if "refund policy" in q:
        return {
            "answer": "We offer a 30-day refund policy for all purchases.",
            "steps": 2,
            "tools_used": ["knowledge_base"],
        }
    elif "charged twice" in q:
        return {
            "answer": "I found order #12345. A refund processed for the duplicate charge.",
            "steps": 5,
            "tools_used": ["order_lookup", "payment_system", "notification"],
        }
    return {"answer": "I'm not sure.", "steps": 1, "tools_used": []}

report = suite.run(mock_agent, verbose=True)
print(f"\nSuite: {report['suite']}")
print(f"Pass rate: {report['pass_rate']:.0%}")
print(f"Scores: {report['avg_scores']}")
print(f"By tag: {report['by_tag']}")
```

---

## 연습문제

### 연습문제 1: 궤적 품질 채점기 (Trajectory Quality Scorer)

에이전트의 행동 시퀀스를 최적 참조 궤적과 비교하여 평가하는 `TrajectoryScorer`를 구현하라. (a) 행동 겹침 점수(action overlap score), (b) 순서 점수(ordering score, 행동이 올바른 순서인가?), (c) 중복 페널티(redundancy penalty, 반복되거나 불필요한 행동)를 계산해야 한다.

<details>
<summary>정답 보기</summary>

```python
from collections import Counter


class TrajectoryScorer:
    """Score agent trajectories against optimal reference trajectories."""

    def __init__(self):
        pass

    def score(self, actual: list[str], reference: list[str]) -> dict:
        """Score an actual trajectory against a reference."""
        overlap = self._action_overlap(actual, reference)
        ordering = self._ordering_score(actual, reference)
        redundancy = self._redundancy_penalty(actual)

        # Weighted overall score
        overall = (
            overlap * 0.4
            + ordering * 0.3
            + (1.0 - redundancy) * 0.3
        )

        return {
            "action_overlap": round(overlap, 4),
            "ordering_score": round(ordering, 4),
            "redundancy_penalty": round(redundancy, 4),
            "overall_score": round(overall, 4),
            "actual_length": len(actual),
            "reference_length": len(reference),
        }

    def _action_overlap(self, actual: list[str], reference: list[str]) -> float:
        """Jaccard similarity of action sets."""
        if not reference:
            return 1.0 if not actual else 0.0

        actual_set = set(actual)
        reference_set = set(reference)
        intersection = actual_set & reference_set
        union = actual_set | reference_set

        return len(intersection) / len(union) if union else 0.0

    def _ordering_score(self, actual: list[str], reference: list[str]) -> float:
        """Score based on longest common subsequence (LCS)."""
        if not reference or not actual:
            return 0.0

        # LCS dynamic programming
        m, n = len(actual), len(reference)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if actual[i-1] == reference[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        lcs_length = dp[m][n]
        return lcs_length / len(reference)

    def _redundancy_penalty(self, actual: list[str]) -> float:
        """Penalize repeated actions (0 = no redundancy, 1 = all redundant)."""
        if len(actual) <= 1:
            return 0.0

        counts = Counter(actual)
        total_redundant = sum(count - 1 for count in counts.values())
        return total_redundant / len(actual)


# Test
scorer = TrajectoryScorer()

# Reference (optimal) trajectory
reference = ["search", "filter", "compare", "select", "confirm"]

# Good trajectory (close to optimal)
good = ["search", "filter", "compare", "select", "confirm"]
print("Good trajectory:", scorer.score(good, reference))

# Okay trajectory (extra steps, slight reorder)
okay = ["search", "search", "filter", "select", "compare", "confirm"]
print("Okay trajectory:", scorer.score(okay, reference))

# Bad trajectory (wrong tools, excessive steps)
bad = ["search", "search", "search", "browse", "search", "filter", "confirm"]
print("Bad trajectory:", scorer.score(bad, reference))
```

**핵심 설계 포인트:**
- 행동 겹침(Jaccard)은 에이전트가 올바른 도구를 사용했는지 측정한다
- 순서 점수(LCS)는 행동이 논리적인 순서로 수행되었는지 측정한다
- 중복 페널티(redundancy penalty)는 루프와 불필요한 반복을 포착한다
- 결합 점수는 세 가지 차원을 균형 있게 반영한다
</details>

---

### 연습문제 2: 다중 모델 비용 최적화기 (Multi-Model Cost Optimizer)

여러 모델의 평가 결과를 받아 가장 비용 효율적인 모델 구성을 추천하는 `CostOptimizer`를 구축하라. 설정 가능한 품질 임계값(quality threshold)을 사용하여 품질과 비용의 균형을 맞춰야 한다.

<details>
<summary>정답 보기</summary>

```python
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str
    avg_quality: float     # 0.0 to 1.0
    avg_cost_per_task: float  # USD
    success_rate: float    # 0.0 to 1.0
    avg_latency_ms: float
    tasks_evaluated: int


class CostOptimizer:
    """Recommend the most cost-effective model configuration."""

    def __init__(self, min_quality: float = 0.7,
                 min_success_rate: float = 0.8):
        self.min_quality = min_quality
        self.min_success_rate = min_success_rate
        self.configs: list[ModelConfig] = []

    def add_config(self, config: ModelConfig):
        self.configs.append(config)

    def recommend(self) -> dict:
        """Find the cheapest model that meets quality thresholds."""
        eligible = [
            c for c in self.configs
            if c.avg_quality >= self.min_quality
            and c.success_rate >= self.min_success_rate
        ]

        if not eligible:
            return {
                "recommendation": None,
                "reason": "No model meets the quality/success thresholds",
                "suggestion": "Lower thresholds or improve agent prompts",
                "all_configs": self._format_configs(self.configs),
            }

        # Sort by cost-effectiveness: quality per dollar
        eligible.sort(key=lambda c: c.avg_cost_per_task)

        best = eligible[0]

        # Calculate savings vs most expensive option
        most_expensive = max(self.configs, key=lambda c: c.avg_cost_per_task)
        monthly_savings = (most_expensive.avg_cost_per_task - best.avg_cost_per_task) * 1000

        return {
            "recommendation": best.name,
            "cost_per_task": round(best.avg_cost_per_task, 4),
            "quality": round(best.avg_quality, 3),
            "success_rate": round(best.success_rate, 3),
            "latency_ms": round(best.avg_latency_ms, 1),
            "monthly_savings_1k_tasks": round(monthly_savings, 2),
            "all_eligible": self._format_configs(eligible),
            "rejected": self._format_configs(
                [c for c in self.configs if c not in eligible]
            ),
        }

    def recommend_tiered(self) -> dict:
        """Recommend a tiered strategy: cheap model for easy, expensive for hard."""
        if len(self.configs) < 2:
            return self.recommend()

        sorted_by_cost = sorted(self.configs, key=lambda c: c.avg_cost_per_task)
        cheap = sorted_by_cost[0]
        expensive = sorted_by_cost[-1]

        # Recommend cheap model for tasks where it meets quality bar
        # and expensive model for the rest
        cheap_eligible = cheap.avg_quality >= self.min_quality

        if cheap_eligible:
            # Estimate: 70% of tasks can use cheap model
            easy_fraction = 0.7
            blended_cost = (
                cheap.avg_cost_per_task * easy_fraction
                + expensive.avg_cost_per_task * (1 - easy_fraction)
            )
        else:
            easy_fraction = 0.0
            blended_cost = expensive.avg_cost_per_task

        return {
            "strategy": "tiered",
            "fast_model": cheap.name,
            "fast_model_quality": round(cheap.avg_quality, 3),
            "power_model": expensive.name,
            "power_model_quality": round(expensive.avg_quality, 3),
            "estimated_easy_fraction": easy_fraction,
            "blended_cost_per_task": round(blended_cost, 4),
            "vs_always_expensive": round(
                expensive.avg_cost_per_task - blended_cost, 4
            ),
        }

    def _format_configs(self, configs: list[ModelConfig]) -> list[dict]:
        return [
            {
                "model": c.name,
                "cost": round(c.avg_cost_per_task, 4),
                "quality": round(c.avg_quality, 3),
                "success": round(c.success_rate, 3),
                "latency": round(c.avg_latency_ms, 1),
            }
            for c in configs
        ]


# Test
optimizer = CostOptimizer(min_quality=0.7, min_success_rate=0.8)

optimizer.add_config(ModelConfig("claude-sonnet-4-20250514", 0.92, 0.045, 0.95, 2500, 100))
optimizer.add_config(ModelConfig("claude-haiku-4-20250514", 0.75, 0.003, 0.85, 800, 100))
optimizer.add_config(ModelConfig("gpt-4o", 0.90, 0.035, 0.93, 2200, 100))
optimizer.add_config(ModelConfig("gpt-4o-mini", 0.72, 0.002, 0.82, 600, 100))

# Single-model recommendation
rec = optimizer.recommend()
print(f"Recommended: {rec['recommendation']}")
print(f"Cost/task: ${rec['cost_per_task']}")
print(f"Quality: {rec['quality']}")
print(f"Monthly savings (1k tasks): ${rec['monthly_savings_1k_tasks']}")

# Tiered recommendation
tiered = optimizer.recommend_tiered()
print(f"\nTiered strategy:")
print(f"  Fast: {tiered['fast_model']} (quality={tiered['fast_model_quality']})")
print(f"  Power: {tiered['power_model']} (quality={tiered['power_model_quality']})")
print(f"  Blended cost: ${tiered['blended_cost_per_task']}")
print(f"  Savings vs always-power: ${tiered['vs_always_expensive']}/task")
```

**핵심 인사이트:**
- 모델을 선택하기 전에 항상 벤치마크를 수행하라 — 비용/품질 트레이드오프에 대한 가정은 종종 틀린다
- 계층적 전략(쉬운 작업에는 저렴한 모델, 어려운 작업에는 고가 모델)이 일반적으로 단일 모델보다 낫다
- 월간 절감액 예측이 비즈니스 케이스를 구체화한다
</details>

---

### 연습문제 3: 루프 탐지와 서킷 브레이커 (Loop Detection and Circuit Breaker)

에이전트의 도구 호출 루프를 감싸는 `CircuitBreaker`를 구현하라. 세 가지 패턴을 탐지해야 한다: (a) 정확한 행동 반복(exact action repetition), (b) 관찰 반복(observation repetition, 같은 결과를 받음), (c) 비용 폭주(cost runaway). 트리거되면 에이전트를 우아하게 중단해야 한다.

<details>
<summary>정답 보기</summary>

```python
from collections import deque
import hashlib


class CircuitBreaker:
    """Detect and halt runaway agent behavior."""

    def __init__(self, max_repeated_actions: int = 3,
                 max_repeated_observations: int = 3,
                 max_cost_usd: float = 1.0,
                 window_size: int = 10):
        self.max_repeated_actions = max_repeated_actions
        self.max_repeated_observations = max_repeated_observations
        self.max_cost_usd = max_cost_usd
        self.window_size = window_size

        self.action_history: deque[str] = deque(maxlen=window_size)
        self.observation_hashes: deque[str] = deque(maxlen=window_size)
        self.total_cost: float = 0.0
        self.trip_reason: str | None = None
        self.is_tripped: bool = False

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def record_action(self, action: str):
        """Record an agent action and check for loops."""
        self.action_history.append(action)

    def record_observation(self, observation: str):
        """Record a tool observation and check for repeated results."""
        obs_hash = self._hash_text(observation)
        self.observation_hashes.append(obs_hash)

    def record_cost(self, cost_usd: float):
        """Accumulate cost."""
        self.total_cost += cost_usd

    def check(self) -> dict:
        """Check all circuit breaker conditions. Returns status."""
        checks = {
            "action_loop": self._check_action_loop(),
            "observation_loop": self._check_observation_loop(),
            "cost_exceeded": self._check_cost(),
        }

        tripped = any(c["triggered"] for c in checks.values())
        if tripped and not self.is_tripped:
            self.is_tripped = True
            reasons = [
                name for name, c in checks.items() if c["triggered"]
            ]
            self.trip_reason = ", ".join(reasons)

        return {
            "tripped": self.is_tripped,
            "reason": self.trip_reason,
            "checks": checks,
            "total_cost": round(self.total_cost, 4),
            "actions_recorded": len(self.action_history),
        }

    def _check_action_loop(self) -> dict:
        """Check for repeated actions."""
        if len(self.action_history) < self.max_repeated_actions:
            return {"triggered": False}

        recent = list(self.action_history)[-self.max_repeated_actions:]
        all_same = len(set(recent)) == 1

        # Also check for alternating patterns (A-B-A-B)
        if len(self.action_history) >= 4:
            last_4 = list(self.action_history)[-4:]
            alternating = last_4[0] == last_4[2] and last_4[1] == last_4[3]
        else:
            alternating = False

        triggered = all_same or alternating
        return {
            "triggered": triggered,
            "pattern": "exact_repeat" if all_same else "alternating" if alternating else "none",
            "repeated_action": recent[0] if all_same else None,
        }

    def _check_observation_loop(self) -> dict:
        """Check for repeated tool observations."""
        if len(self.observation_hashes) < self.max_repeated_observations:
            return {"triggered": False}

        recent = list(self.observation_hashes)[-self.max_repeated_observations:]
        all_same = len(set(recent)) == 1

        return {
            "triggered": all_same,
            "consecutive_same": all_same,
        }

    def _check_cost(self) -> dict:
        """Check if cost has exceeded the budget."""
        exceeded = self.total_cost >= self.max_cost_usd
        return {
            "triggered": exceeded,
            "total_cost": round(self.total_cost, 4),
            "budget": self.max_cost_usd,
            "utilization": round(self.total_cost / self.max_cost_usd * 100, 1),
        }

    def reset(self):
        """Reset the circuit breaker."""
        self.action_history.clear()
        self.observation_hashes.clear()
        self.total_cost = 0.0
        self.trip_reason = None
        self.is_tripped = False


def run_agent_with_circuit_breaker(agent_fn, query: str,
                                   max_steps: int = 20) -> dict:
    """Run an agent with circuit breaker protection."""
    cb = CircuitBreaker(
        max_repeated_actions=3,
        max_repeated_observations=3,
        max_cost_usd=0.50,
    )

    steps = []
    for step in range(max_steps):
        # Check circuit breaker before each step
        status = cb.check()
        if status["tripped"]:
            return {
                "status": "circuit_breaker_tripped",
                "reason": status["reason"],
                "steps_completed": len(steps),
                "total_cost": status["total_cost"],
                "steps": steps,
            }

        # Simulate agent step
        result = agent_fn(query, step)
        action = result.get("action", "")
        observation = result.get("observation", "")
        cost = result.get("cost", 0.01)

        cb.record_action(action)
        cb.record_observation(observation)
        cb.record_cost(cost)

        steps.append({"step": step, "action": action, "cost": cost})

        if result.get("done", False):
            return {
                "status": "completed",
                "steps_completed": len(steps),
                "total_cost": cb.total_cost,
                "steps": steps,
            }

    return {
        "status": "max_steps_reached",
        "steps_completed": len(steps),
        "total_cost": cb.total_cost,
    }


# Test: Agent that gets stuck in a loop
def loopy_agent(query: str, step: int) -> dict:
    if step < 2:
        return {"action": f"search_{step}", "observation": f"result_{step}", "cost": 0.01}
    # Gets stuck repeating the same action
    return {"action": "search_same", "observation": "same_result", "cost": 0.01}

result = run_agent_with_circuit_breaker(loopy_agent, "test query")
print(f"Status: {result['status']}")
print(f"Reason: {result.get('reason', 'N/A')}")
print(f"Steps: {result['steps_completed']}")
print(f"Cost: ${result['total_cost']:.4f}")
```

**서킷 브레이커가 중요한 이유:**
- 서킷 브레이커가 없으면 멈춘 에이전트가 몇 분 만에 API 예산을 소진할 수 있다
- 행동 루프 탐지(action loop detection)는 가장 흔한 실패 모드를 포착한다
- 관찰 루프 탐지(observation loop detection)는 에이전트가 도움이 되지 않는 결과를 반환하는 도구를 계속 호출하는 경우를 포착한다
- 비용 제한(cost limits)은 행동과 관계없이 강제 상한을 제공한다
</details>

---

### 연습문제 4: 벤치마크 보고서 생성기 (Benchmark Report Generator)

여러 벤치마크의 평가 결과를 받아 구조화된 비교 보고서를 생성하는 `BenchmarkReporter`를 만들어라. 모델 순위를 매기고, 신뢰 구간(confidence interval)을 계산하고, 통계적으로 유의미한 차이를 식별해야 한다.

<details>
<summary>정답 보기</summary>

```python
import math
from collections import defaultdict


class BenchmarkReporter:
    """Generate structured benchmark comparison reports."""

    def __init__(self):
        self.results: dict[str, list[dict]] = defaultdict(list)

    def add_result(self, benchmark: str, model: str,
                   score: float, metadata: dict | None = None):
        """Add a single benchmark result."""
        self.results[benchmark].append({
            "model": model,
            "score": score,
            "metadata": metadata or {},
        })

    def add_batch(self, benchmark: str, model: str,
                  scores: list[float]):
        """Add multiple runs of the same benchmark/model."""
        for score in scores:
            self.add_result(benchmark, model, score)

    def _confidence_interval(self, scores: list[float],
                             confidence: float = 0.95) -> tuple[float, float]:
        """Compute confidence interval using t-distribution approximation."""
        n = len(scores)
        if n < 2:
            return (scores[0], scores[0]) if scores else (0.0, 0.0)

        mean = sum(scores) / n
        variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
        std_err = math.sqrt(variance / n)

        # t-value approximation for 95% CI
        t_values = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                    10: 2.228, 20: 2.086, 30: 2.042}
        t_val = t_values.get(n, 1.96)  # Fall back to z-value

        margin = t_val * std_err
        return (round(mean - margin, 4), round(mean + margin, 4))

    def _is_significant(self, scores_a: list[float],
                        scores_b: list[float]) -> bool:
        """Simple significance test: do confidence intervals overlap?"""
        ci_a = self._confidence_interval(scores_a)
        ci_b = self._confidence_interval(scores_b)
        # No overlap = significant difference
        return ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0]

    def generate_report(self) -> dict:
        """Generate a comprehensive benchmark report."""
        report = {
            "benchmarks": {},
            "model_rankings": {},
            "pairwise_comparisons": [],
        }

        all_model_scores = defaultdict(list)

        for benchmark, results in self.results.items():
            # Group by model
            by_model = defaultdict(list)
            for r in results:
                by_model[r["model"]].append(r["score"])

            benchmark_report = {}
            for model, scores in by_model.items():
                mean = sum(scores) / len(scores)
                ci = self._confidence_interval(scores)
                benchmark_report[model] = {
                    "mean": round(mean, 4),
                    "std": round(
                        math.sqrt(sum((x - mean)**2 for x in scores) / max(len(scores) - 1, 1)),
                        4
                    ),
                    "ci_95": ci,
                    "n_runs": len(scores),
                    "min": round(min(scores), 4),
                    "max": round(max(scores), 4),
                }
                all_model_scores[model].append(mean)

            # Rank models for this benchmark
            ranked = sorted(
                benchmark_report.items(),
                key=lambda x: x[1]["mean"],
                reverse=True,
            )
            report["benchmarks"][benchmark] = {
                "models": benchmark_report,
                "ranking": [m for m, _ in ranked],
            }

            # Pairwise significance tests
            models = list(by_model.keys())
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    sig = self._is_significant(
                        by_model[models[i]], by_model[models[j]]
                    )
                    if sig:
                        winner = models[i] if (
                            sum(by_model[models[i]]) / len(by_model[models[i]])
                            > sum(by_model[models[j]]) / len(by_model[models[j]])
                        ) else models[j]
                        report["pairwise_comparisons"].append({
                            "benchmark": benchmark,
                            "model_a": models[i],
                            "model_b": models[j],
                            "significant": True,
                            "winner": winner,
                        })

        # Overall model rankings (average across benchmarks)
        for model, avg_scores in all_model_scores.items():
            report["model_rankings"][model] = {
                "avg_across_benchmarks": round(
                    sum(avg_scores) / len(avg_scores), 4
                ),
                "benchmarks_participated": len(avg_scores),
            }

        return report


# Test
reporter = BenchmarkReporter()

# Add results from multiple benchmarks and models
import random
random.seed(42)

for _ in range(10):
    reporter.add_result("SWE-bench", "claude-sonnet", 0.45 + random.uniform(-0.05, 0.05))
    reporter.add_result("SWE-bench", "gpt-4o", 0.42 + random.uniform(-0.05, 0.05))
    reporter.add_result("SWE-bench", "claude-haiku", 0.28 + random.uniform(-0.05, 0.05))

    reporter.add_result("GAIA-L1", "claude-sonnet", 0.75 + random.uniform(-0.05, 0.05))
    reporter.add_result("GAIA-L1", "gpt-4o", 0.72 + random.uniform(-0.05, 0.05))
    reporter.add_result("GAIA-L1", "claude-haiku", 0.55 + random.uniform(-0.05, 0.05))

report = reporter.generate_report()

for benchmark, data in report["benchmarks"].items():
    print(f"\n{benchmark}:")
    print(f"  Ranking: {data['ranking']}")
    for model, stats in data["models"].items():
        print(f"  {model}: {stats['mean']:.3f} ± CI {stats['ci_95']}")

print(f"\nSignificant differences: {len(report['pairwise_comparisons'])}")
for comp in report["pairwise_comparisons"]:
    print(f"  {comp['benchmark']}: {comp['winner']} > "
          f"{comp['model_a'] if comp['winner'] != comp['model_a'] else comp['model_b']}")
```

**핵심 특징:**
- 신뢰 구간(confidence intervals)은 제한된 평가 실행에서의 불확실성을 정량화한다
- 쌍별 유의성 검정(pairwise significance tests)은 의미 있는 차이와 노이즈를 구별한다
- 교차 벤치마크 순위(cross-benchmark rankings)는 전체적인 그림을 제공한다
- 에이전트의 비결정성 때문에 항상 여러 번의 평가 실행(n >= 5)을 수행하라
</details>

---

### 연습문제 5: 종단간 에이전트 평가기 (End-to-End Agent Evaluator)

궤적 분석(trajectory analysis), 실패 탐지(failure detection), 비용 추적(cost tracking), 품질 점수(quality scoring)를 하나의 평가 파이프라인으로 결합하는 완전한 `AgentEvaluator` 클래스를 구축하라. 테스트 케이스 목록과 에이전트 함수를 받아 종합 보고서를 생성해야 한다.

<details>
<summary>정답 보기</summary>

```python
from dataclasses import dataclass, field
from typing import Callable
from collections import defaultdict
import time
import json


@dataclass
class TestCase:
    case_id: str
    input_query: str
    expected_answer: str
    expected_tools: list[str] = field(default_factory=list)
    max_steps: int = 20
    max_cost_usd: float = 0.50
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    answer: str
    steps: list[dict]
    tools_used: list[str]
    total_tokens: int
    total_cost: float
    trajectory: list[str]


class AgentEvaluator:
    """Complete end-to-end evaluation pipeline for agent systems."""

    def __init__(self):
        self.test_cases: list[TestCase] = []
        self.results: list[dict] = []
        self.failure_detector = AgentFailureDetector()

    def add_test_case(self, case: TestCase):
        self.test_cases.append(case)

    def evaluate(self, agent_fn: Callable[[str], AgentResult],
                 verbose: bool = True) -> dict:
        """Run the complete evaluation pipeline."""
        self.results = []

        for case in self.test_cases:
            start = time.time()
            try:
                agent_result = agent_fn(case.input_query)
                latency = time.time() - start

                # 1. Correctness check
                correctness = self._check_correctness(
                    agent_result.answer, case.expected_answer
                )

                # 2. Efficiency analysis
                efficiency = self._analyze_efficiency(
                    agent_result, case
                )

                # 3. Tool accuracy
                tool_accuracy = self._check_tools(
                    agent_result.tools_used, case.expected_tools
                )

                # 4. Failure detection
                trajectory_data = [
                    {"action": a, "step_number": i}
                    for i, a in enumerate(agent_result.trajectory)
                ]
                failures = self.failure_detector.analyze(trajectory_data)

                # 5. Cost analysis
                cost = {
                    "total_cost_usd": agent_result.total_cost,
                    "within_budget": agent_result.total_cost <= case.max_cost_usd,
                    "tokens_used": agent_result.total_tokens,
                }

                # Overall score
                scores = [
                    1.0 if correctness["correct"] else 0.0,
                    efficiency["efficiency_score"],
                    tool_accuracy["f1_score"],
                    1.0 if failures["severity"] == "clean" else 0.5 if failures["severity"] == "warning" else 0.0,
                    1.0 if cost["within_budget"] else 0.5,
                ]
                overall = sum(scores) / len(scores)

                result = {
                    "case_id": case.case_id,
                    "difficulty": case.difficulty,
                    "tags": case.tags,
                    "correctness": correctness,
                    "efficiency": efficiency,
                    "tool_accuracy": tool_accuracy,
                    "failures": {"severity": failures["severity"], "issues": failures["total_issues"]},
                    "cost": cost,
                    "latency_seconds": round(latency, 3),
                    "overall_score": round(overall, 4),
                    "passed": overall >= 0.7,
                }

            except Exception as e:
                result = {
                    "case_id": case.case_id,
                    "difficulty": case.difficulty,
                    "tags": case.tags,
                    "error": str(e),
                    "overall_score": 0.0,
                    "passed": False,
                }

            self.results.append(result)
            if verbose:
                status = "PASS" if result["passed"] else "FAIL"
                print(f"  [{status}] {case.case_id}: score={result['overall_score']:.2f}")

        return self._generate_report()

    def _check_correctness(self, actual: str, expected: str) -> dict:
        actual_lower = actual.strip().lower()
        expected_lower = expected.strip().lower()
        exact = actual_lower == expected_lower
        contains = expected_lower in actual_lower
        return {
            "correct": exact or contains,
            "exact_match": exact,
            "contains_match": contains,
        }

    def _analyze_efficiency(self, result: AgentResult, case: TestCase) -> dict:
        num_steps = len(result.steps)
        within_limit = num_steps <= case.max_steps
        # Score: 1.0 if <= 5 steps, decreasing linearly to 0 at max_steps
        if num_steps <= 5:
            score = 1.0
        elif num_steps <= case.max_steps:
            score = 1.0 - (num_steps - 5) / max(case.max_steps - 5, 1)
        else:
            score = 0.0

        return {
            "num_steps": num_steps,
            "max_steps": case.max_steps,
            "within_limit": within_limit,
            "efficiency_score": round(max(0, score), 4),
        }

    def _check_tools(self, actual: list[str], expected: list[str]) -> dict:
        if not expected:
            return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0}

        actual_set = set(actual)
        expected_set = set(expected)
        tp = len(actual_set & expected_set)

        precision = tp / len(actual_set) if actual_set else 0
        recall = tp / len(expected_set) if expected_set else 0
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "extra_tools": list(actual_set - expected_set),
            "missing_tools": list(expected_set - actual_set),
        }

    def _generate_report(self) -> dict:
        """Generate the final evaluation report."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get("passed", False))
        errors = sum(1 for r in self.results if "error" in r)

        # By difficulty
        by_difficulty = defaultdict(list)
        for r in self.results:
            by_difficulty[r["difficulty"]].append(r.get("overall_score", 0))

        difficulty_report = {
            diff: {
                "count": len(scores),
                "avg_score": round(sum(scores) / len(scores), 4),
                "pass_rate": round(sum(1 for s in scores if s >= 0.7) / len(scores), 4),
            }
            for diff, scores in by_difficulty.items()
        }

        # By tag
        by_tag = defaultdict(list)
        for r in self.results:
            for tag in r.get("tags", []):
                by_tag[tag].append(r.get("overall_score", 0))

        tag_report = {
            tag: round(sum(scores) / len(scores), 4)
            for tag, scores in by_tag.items()
        }

        # Score distribution
        all_scores = [r.get("overall_score", 0) for r in self.results]

        return {
            "summary": {
                "total_cases": total,
                "passed": passed,
                "failed": total - passed - errors,
                "errors": errors,
                "pass_rate": round(passed / max(total, 1), 4),
                "avg_score": round(sum(all_scores) / max(total, 1), 4),
            },
            "by_difficulty": difficulty_report,
            "by_tag": tag_report,
            "worst_cases": sorted(
                [r for r in self.results if not r.get("passed")],
                key=lambda r: r.get("overall_score", 0),
            )[:5],
        }


# Test
evaluator = AgentEvaluator()

evaluator.add_test_case(TestCase(
    "tc-001", "What is 2+2?", "4",
    expected_tools=["calculator"], max_steps=5,
    difficulty="easy", tags=["math"],
))
evaluator.add_test_case(TestCase(
    "tc-002", "Find the capital of Japan", "Tokyo",
    expected_tools=["search"], max_steps=10,
    difficulty="easy", tags=["geography"],
))
evaluator.add_test_case(TestCase(
    "tc-003", "Compare GDP of US and China", "US has higher GDP",
    expected_tools=["search", "calculator"], max_steps=15,
    difficulty="medium", tags=["economics", "comparison"],
))

# Mock agent
def mock_agent(query: str) -> AgentResult:
    if "2+2" in query:
        return AgentResult("4", [{"tool": "calculator"}], ["calculator"], 100, 0.001, ["calculator"])
    elif "capital" in query:
        return AgentResult("The capital of Japan is Tokyo", [{"tool": "search"}], ["search"], 200, 0.005, ["search"])
    elif "GDP" in query:
        return AgentResult("The US has higher GDP than China", [{"tool": "search"}, {"tool": "calculator"}],
                          ["search", "calculator", "search"], 500, 0.02,
                          ["search", "calculator", "search", "format"])
    return AgentResult("I don't know", [], [], 50, 0.001, [])

report = evaluator.evaluate(mock_agent, verbose=True)
print(f"\n{'='*50}")
print(f"Pass rate: {report['summary']['pass_rate']:.0%}")
print(f"Avg score: {report['summary']['avg_score']:.2f}")
print(f"By difficulty: {json.dumps(report['by_difficulty'], indent=2)}")
print(f"By tag: {report['by_tag']}")
```

**이 평가기는 이 레슨의 모든 개념을 결합한다:**
- 정확도 검사 (정확 일치 + 포함 일치)
- 효율성 점수화 (단계 수 vs 제한)
- 도구 사용 정밀도/재현율/F1
- 실패 모드 탐지 (루프, 연쇄 등)
- 비용 예산 관리
- 다차원 보고 (난이도별, 태그별)
- 목표 개선을 위한 최악의 사례 식별
</details>

---

## 다음 단계

[에이전트 설계 패턴](./27_Agent_Design_Patterns.md)에서는 프로덕션 수준의 에이전트를 구축하기 위한 검증된 아키텍처 패턴인 오케스트레이터-워커(orchestrator-worker), 라우터(router), 슈퍼바이저(supervisor), 휴먼 인 더 루프(human-in-the-loop) 등을 살펴본다.
