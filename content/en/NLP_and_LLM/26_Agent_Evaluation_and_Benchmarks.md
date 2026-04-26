# 26. Agent Evaluation and Benchmarks

Previous: [Agent Memory and Planning](./25_Agent_Memory_and_Planning.md) | Next: [Agent Design Patterns](./27_Agent_Design_Patterns.md)

## Learning Objectives

- Understand the unique challenges of evaluating agentic systems vs static LLMs
- Compare established agent benchmarks: AgentBench, SWE-bench, WebArena, GAIA
- Design evaluation methodologies covering task completion, efficiency, and safety
- Analyze common agent failure modes and build detection mechanisms
- Implement agent observability, tracing, and cost-quality analysis
- Build custom evaluation frameworks for domain-specific agents

---

## Table of Contents

Before the benchmark and methodology reference, read [**Theory & Principles**](#theory--principles) — what makes agent evaluation different from LLM evaluation, the trajectory-vs-outcome distinction, and the cost-quality frontier that all agent benchmarks navigate.

1. [Agent Evaluation Challenges](#1-agent-evaluation-challenges)
2. [AgentBench](#2-agentbench)
3. [SWE-bench](#3-swe-bench)
4. [WebArena](#4-webarena)
5. [GAIA Benchmark](#5-gaia-benchmark)
6. [Evaluation Methodology](#6-evaluation-methodology)
7. [Failure Mode Analysis](#7-failure-mode-analysis)
8. [Agent Observability and Tracing](#8-agent-observability-and-tracing)
9. [Cost-Quality Analysis](#9-cost-quality-analysis)
10. [Building Custom Agent Evals](#10-building-custom-agent-evals)
11. [Exercises](#exercises)

---

## Theory & Principles

Evaluating an agent is harder than evaluating an LLM. An LLM has one input (a prompt) and one output (a response); evaluation compares output against reference. An agent has a *trajectory* — a sequence of (thought, action, observation) tuples — that may take many LLM calls and tool invocations to reach a goal. Two agents can both succeed at a task but with wildly different cost, latency, safety, and trajectory quality. Agent evaluation must measure not just whether the goal was achieved, but how, and at what cost.

This section covers:

- **(A) The trajectory-vs-outcome distinction** — why "did it succeed?" is not enough.
- **(B) Task success and partial credit** — designing a task is designing a measurement protocol.
- **(C) Major benchmarks** — AgentBench, SWE-bench, WebArena, GAIA, and what each really tests.
- **(D) Failure mode taxonomy** — common ways agents fail (loops, hallucination, tool misuse, premature stopping).
- **(E) Trajectory analysis** — observability, tracing, the value of structured logs.
- **(F) Cost-quality frontier** — Pareto trade-offs and how to communicate them.
- **(G) Building custom evals** — task design, ground truth, automatic scoring.

### A. Trajectory vs Outcome

Two perspectives on agent quality:

- **Outcome-only**: did the agent achieve the goal? Pass/fail or score on a final answer. Easy to measure, sometimes deceptive (an agent can stumble into the right answer through a tortured trajectory).
- **Trajectory**: examine each step. Did each tool call make sense? Did the agent waste cycles? Did it follow safety constraints? Hard to measure automatically, but essential for understanding agent behavior.

Production agent eval needs both: outcome metrics for top-level reporting, trajectory metrics for understanding when (and why) the agent fails.

### B. Task Success and Partial Credit

Defining "success" for an open-ended task is itself a research problem. Three patterns:

- **Binary**: pass or fail. Suitable when there's a clear unambiguous correct outcome (a code patch, a SQL query, a math answer).
- **Graded**: 0-1 score from a rubric (LLM-as-judge or automated checks). Suitable for nuanced tasks (writing, summarization).
- **Multi-objective**: separate scores for correctness, efficiency, safety, etc. Combine into a single score only with care — averaging hides trade-offs.

For benchmarks: **automatic verifiability** is what makes a benchmark scalable. SWE-bench works because patches can be tested. WebArena works because final web state is checkable. Tasks where success requires human judgment scale poorly.

### C. Major Benchmarks

**C.1 AgentBench** (Liu et al., 2023). Eight diverse environments (OS, DB, knowledge graph, etc.). Agent must complete tasks in each. Stress-tests generality — an agent that wins on one environment can fail badly on others.

**C.2 SWE-bench** (Jimenez et al., 2023). Real GitHub issues from popular Python repos. Agent must produce a patch that resolves the issue and passes the project's existing tests. Hard, realistic, automatic verification. The reference benchmark for code-agent capability.

**C.3 WebArena** (Zhou et al., 2023). Realistic web tasks (shopping, GitLab, content management) on actual websites running locally. Tests web browsing, form-filling, multi-page reasoning. Reproducible because the websites are containerized.

**C.4 GAIA** (Mialon et al., 2023). General assistant benchmark — questions that require multi-step reasoning, tool use, web search, file processing. Easy for humans (most score >90%); current LLMs (GPT-4 with tools) score 30-50%. Designed to expose the capability gap.

Each benchmark probes a different aspect: AgentBench for breadth, SWE-bench for code, WebArena for web, GAIA for general assistance. No single benchmark captures everything.

### D. Failure Mode Taxonomy

Common ways agents fail:

**D.1 Infinite loops.** Agent calls the same tool with the same args repeatedly. Detection: hash-and-compare past calls, alert on repeats.

**D.2 Hallucinated tools or arguments.** Agent invents a tool name, calls a real tool with imaginary parameters. Detection: schema validation (function calling makes this nearly impossible at the API level).

**D.3 Premature stopping.** Agent declares "task complete" before actually completing the task. Detection: outcome verification, force the agent to justify completion.

**D.4 Tool misuse.** Agent calls the right tool but with wrong arguments — searches for "Python" when the user asked about "Python the language." Detection: harder; usually requires LLM-as-judge on the trajectory.

**D.5 Cost runaway.** Agent gets stuck in unproductive exploration, racking up tokens and tool calls. Detection: hard caps on steps and budget.

**D.6 Safety violation.** Agent calls a destructive tool (delete, send, pay) without proper authorization. Detection: human-in-the-loop for high-stakes tools, output filtering for the rest.

A robust agent eval categorizes each failure into these (or domain-specific) classes, so you can prioritize fixes.

### E. Trajectory Analysis

Logging structure matters. A useful agent log includes, per step:

- Step number and timestamp.
- Thought (LLM reasoning).
- Action (tool name + args).
- Observation (tool result, possibly truncated).
- Tokens consumed, cost, latency.
- Optionally: a per-step LLM-as-judge score for "was this step productive?"

Tools like LangSmith, Phoenix, Langfuse, and Helicone (lesson 24) instrument agents with this structure automatically. Without trajectory logs, debugging a failed agent run is essentially impossible.

### F. Cost-Quality Frontier

Two agents with the same success rate can have wildly different cost:

- Agent A: 80% success, $0.10 per task.
- Agent B: 80% success, $1.50 per task.

Same outcome metric, 15× cost difference. Agent A is strictly better unless B has other advantages (latency, safety, etc.). Agent evaluation should report cost alongside quality.

For each agent, plot success vs cost and find the Pareto frontier. Decisions about which agent to deploy depend on the application's cost sensitivity. A back-office data extraction tool can afford expensive agents; a real-time chat companion cannot.

### G. Building Custom Evals

For a specific application, public benchmarks are necessary but insufficient. Custom evals:

1. **Define tasks** representative of real production traffic. Sample real user queries (with privacy controls) or write synthetic ones based on use cases.
2. **Establish ground truth.** For each task, what is the correct outcome? Manual labeling (expensive, gold standard) or LLM-as-judge (cheap, good enough for most).
3. **Run candidate agents** against the eval set.
4. **Compute metrics**: success rate, cost, latency, failure-mode distribution.
5. **Track over time.** Re-run on every release to catch regressions.

Standard pattern: a small "smoke test" eval (10-50 tasks, runs in minutes) for every commit; a larger "comprehensive" eval (200-1000 tasks, runs in hours) for releases.

### From Theory to the Functions Below

- §1 (challenges) — frames §A trajectory-vs-outcome and §B success definitions.
- §2-§5 (AgentBench, SWE-bench, WebArena, GAIA) — implements §C's main benchmarks.
- §6 (evaluation methodology) — §B graded scoring and §F multi-objective metrics.
- §7 (failure mode analysis) — implements §D's taxonomy with detectors.
- §8 (observability and tracing) — §E structured logging with LangSmith/Phoenix.
- §9 (cost-quality analysis) — §F Pareto frontier visualization.
- §10 (custom evals) — §G's task-definition + scoring + tracking pipeline.

---

## 1. Agent Evaluation Challenges

### Why Agent Evaluation Is Hard

> **Static LLM vs Agent Evaluation**
>
> - **Static LLM**: Input -> Output (single step, deterministic comparison)
> - **Agent**: Input -> [Plan, Act, Observe, Reflect]* -> Output (multi-step, non-deterministic)
>
> Agents introduce new dimensions: trajectory quality, tool usage correctness,
> efficiency, safety, and recovery from errors — none of which apply to
> traditional LLM benchmarks like MMLU or HellaSwag.

### Evaluation Dimensions

| Dimension | What It Measures | Example Metric |
|-----------|-----------------|----------------|
| **Task Completion** | Did the agent achieve the goal? | Success rate (%) |
| **Correctness** | Is the output factually correct? | F1 score, exact match |
| **Efficiency** | How many steps/tokens/cost to complete? | Steps-to-completion, total tokens |
| **Trajectory Quality** | Did the agent take a reasonable path? | Optimal step ratio |
| **Tool Use Accuracy** | Were tools called correctly? | Tool precision/recall |
| **Safety** | Did the agent avoid harmful actions? | Safety violation rate |
| **Robustness** | Does it handle edge cases and errors? | Error recovery rate |
| **Latency** | How long does the full task take? | End-to-end seconds |

### The Non-Determinism Problem

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

### Overview

> **AgentBench**
>
> A comprehensive benchmark for evaluating LLM agents across 8 environments:
>
> | Environment | Task Type | Example |
> |-------------|-----------|---------|
> | Operating System | Shell commands | "Find all .py files larger than 1MB" |
> | Database | SQL queries | "Find the top 5 customers by revenue" |
> | Knowledge Graph | SPARQL/Cypher | "Find all movies directed by Nolan" |
> | Digital Card Game | Strategy | "Play a turn in a card game" |
> | Lateral Thinking | Puzzles | "20 Questions"-style reasoning |
> | House-Holding | Sim tasks | "Put the apple in the fridge" |
> | Web Shopping | E-commerce | "Find a red shirt under $30" |
> | Web Browsing | Navigation | "Book a flight from NYC to LA" |

### AgentBench Evaluation Structure

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

### Software Engineering Benchmark

> **SWE-bench**
>
> Evaluates agents on real-world GitHub issues. The agent must:
> 1. Read the issue description
> 2. Understand the codebase
> 3. Generate a patch that fixes the issue
> 4. Pass the existing test suite
>
> **Key stats:**
> - 2,294 task instances from 12 Python repositories
> - SWE-bench Lite: 300 curated instances for faster evaluation
> - SWE-bench Verified: Human-verified subset with unambiguous solutions
> - Top agents solve ~50% of SWE-bench Verified (as of early 2025)

### SWE-bench Evaluation Framework

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

### Web-Based Agent Evaluation

> **WebArena**
>
> A realistic web environment benchmark with self-hosted websites:
>
> | Site Type | Platform | Example Tasks |
> |-----------|----------|---------------|
> | E-commerce | One-Stop Shop | "Find the cheapest laptop with 16GB RAM" |
> | Forum | Reddit-like | "Post a reply to the top thread in r/cooking" |
> | CMS | GitLab-like | "Create an issue in the frontend repo" |
> | Maps | OpenStreetMap | "Find driving directions from A to B" |
> | Wiki | Wikipedia-like | "Edit the article about machine learning" |
>
> **Key features:**
> - 812 human-verified tasks across 5 websites
> - Functional correctness evaluation (not just action matching)
> - Multi-tab, multi-step tasks that require planning

### Web Agent Evaluation

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

## 5. GAIA Benchmark

### General AI Assistants Benchmark

> **GAIA**
>
> Tests agents on real-world tasks requiring multiple capabilities:
>
> | Level | Complexity | Example |
> |-------|-----------|---------|
> | Level 1 | Single tool, 1-3 steps | "What is the population of France?" |
> | Level 2 | Multi-tool, 3-8 steps | "Compare GDP growth of US and China over the last 5 years" |
> | Level 3 | Complex reasoning + tools | "Plan a 3-day trip optimizing for budget and reviews" |
>
> **Why GAIA matters:**
> - 466 human-crafted questions with unambiguous answers
> - Tests real-world capabilities, not synthetic benchmarks
> - Humans score ~92%; best agents score ~75% (Level 1) and ~40% (Level 3)
> - Measures the gap between "can generate text" and "can actually help"

### GAIA-Style Evaluation

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

## 6. Evaluation Methodology

### Task Completion Evaluation

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

## 7. Failure Mode Analysis

### Common Agent Failure Modes

> **Agent Failure Taxonomy**
>
> | Failure Mode | Description | Frequency |
> |-------------|-------------|-----------|
> | **Infinite Loop** | Agent repeats the same action | High |
> | **Hallucinated Action** | Agent invents non-existent tools/APIs | High |
> | **Wrong Tool Selection** | Uses a tool inappropriate for the task | Medium |
> | **Premature Termination** | Stops before task is complete | Medium |
> | **Context Loss** | Forgets earlier information mid-task | Medium |
> | **Error Cascade** | One failure causes subsequent failures | Medium |
> | **Goal Drift** | Agent pursues a different goal than intended | Low |
> | **Safety Violation** | Agent takes harmful or unauthorized actions | Low |

### Failure Detector

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

## 8. Agent Observability and Tracing

### Structured Trace Collection

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

### OpenTelemetry Integration

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

## 9. Cost-Quality Analysis

### Cost Tracking

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

## 10. Building Custom Agent Evals

### Evaluation Framework

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

## Exercises

### Exercise 1: Trajectory Quality Scorer

Implement a `TrajectoryScorer` that evaluates an agent's action sequence against an optimal reference trajectory. It should compute: (a) action overlap score, (b) ordering score (are actions in the right order?), and (c) redundancy penalty (repeated or unnecessary actions).

<details>
<summary>Show Answer</summary>

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

**Key design points:**
- Action overlap (Jaccard) measures whether the agent used the right tools
- Ordering score (LCS) measures whether actions were in a logical sequence
- Redundancy penalty catches looping and unnecessary repetition
- The combined score balances all three dimensions
</details>

---

### Exercise 2: Multi-Model Cost Optimizer

Build a `CostOptimizer` that, given evaluation results from multiple models, recommends the most cost-effective model configuration. It should balance quality against cost using a configurable quality threshold.

<details>
<summary>Show Answer</summary>

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

**Key insights:**
- Always benchmark before choosing a model — assumptions about cost/quality trade-offs are often wrong
- A tiered strategy (cheap for easy tasks, expensive for hard) usually beats a single model
- Monthly savings projections make the business case concrete
</details>

---

### Exercise 3: Loop Detection and Circuit Breaker

Implement a `CircuitBreaker` that wraps an agent's tool-calling loop. It should detect three patterns: (a) exact action repetition, (b) observation repetition (getting the same result), and (c) cost runaway. When triggered, it should halt the agent gracefully.

<details>
<summary>Show Answer</summary>

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

**Why circuit breakers matter:**
- Without them, a stuck agent can burn through API budget in minutes
- Action loop detection catches the most common failure mode
- Observation loop detection catches cases where the agent keeps calling a tool that returns unhelpful results
- Cost limits provide a hard ceiling regardless of behavior
</details>

---

### Exercise 4: Benchmark Report Generator

Create a `BenchmarkReporter` that takes evaluation results from multiple benchmarks and generates a structured comparison report. It should rank models, compute confidence intervals, and identify statistically significant differences.

<details>
<summary>Show Answer</summary>

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

**Key features:**
- Confidence intervals quantify uncertainty from limited evaluation runs
- Pairwise significance tests identify meaningful differences vs noise
- Cross-benchmark rankings give an overall picture
- Always run multiple evaluation runs (n >= 5) due to agent non-determinism
</details>

---

### Exercise 5: End-to-End Agent Evaluator

Build a complete `AgentEvaluator` class that combines trajectory analysis, failure detection, cost tracking, and quality scoring into a single evaluation pipeline. It should accept a list of test cases and an agent function, then produce a comprehensive report.

<details>
<summary>Show Answer</summary>

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

**This evaluator combines all the concepts from this lesson:**
- Correctness checking (exact + contains match)
- Efficiency scoring (step count vs limit)
- Tool usage precision/recall/F1
- Failure mode detection (loops, cascades, etc.)
- Cost budgeting
- Multi-dimensional reporting (by difficulty, by tag)
- Worst-case identification for targeted improvement
</details>

---

## Next Steps

In [Agent Design Patterns](./27_Agent_Design_Patterns.md), we explore proven architectural patterns for building production-grade agents: orchestrator-worker, router, supervisor, human-in-the-loop, and more.
