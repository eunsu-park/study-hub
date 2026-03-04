# Software Engineering Examples

This directory contains example files demonstrating key software engineering concepts, from foundational principles to CI/CD pipelines. Python examples use only the standard library (no external dependencies).

## Files Overview

### 1. `01_se_principles.py` - Core SE Principles
**Concepts:**
- Brooks's Four Essential Properties (complexity, conformity, changeability, invisibility)
- Seven core principles (rigor, separation of concerns, modularity, abstraction, etc.)
- Programming vs. Software Engineering comparison

**Run:** `python 01_se_principles.py`

---

### 2. `02_sdlc_models.py` - SDLC Models and Selection
**Concepts:**
- SDLC phases and their artifacts
- V-Model verification/validation mapping
- Spiral model risk-driven iteration simulation
- Model selection decision framework (Waterfall, Agile, Spiral, RAD)

**Run:** `python 02_sdlc_models.py`

---

### 3. `03_agile_simulator.py` - Agile Development Simulator
**Concepts:**
- User stories with INVEST criteria and estimation
- Sprint planning with capacity and backlog management
- Kanban board with WIP limits
- Velocity tracking, burndown charts, and sprint forecasting

**Run:** `python 03_agile_simulator.py`

---

### 4. `04_user_story_template.md` - User Stories and Acceptance Criteria
**Concepts:**
- User story format: "As a [role], I want [feature], so that [benefit]"
- Acceptance criteria in Given/When/Then (Gherkin) format
- INVEST criteria checklist
- Definition of Done

---

### 5. `05_uml_class_diagram.py` - UML Class Diagram Generator
**Concepts:**
- ASCII-based UML class diagram rendering
- Class attributes and methods with visibility (+/-/#)
- Relationships: inheritance, composition, aggregation, association, dependency
- E-commerce domain model example

**Run:** `python 05_uml_class_diagram.py`

---

### 6. `06_estimation_calculator.py` - Software Estimation Calculator
**Concepts:**
- COCOMO II basic model (effort, duration, team size)
- Three-point PERT estimation with confidence intervals
- Story point velocity projection
- Function Point Analysis (IFPUG)

**Run:** `python 06_estimation_calculator.py`

---

### 7. `07_code_metrics.py` - Code Quality Metrics
**Concepts:**
- Cyclomatic complexity calculation using Python `ast` module
- Lines of code analysis (total, blank, comment, logical)
- Halstead metrics (vocabulary, volume, difficulty, effort)
- Risk rating classification

**Run:** `python 07_code_metrics.py`

---

### 8. `08_test_plan_template.md` - Test Plan Template
**Concepts:**
- Test plan structure and test cases
- Coverage targets and test priorities
- Entry/exit criteria for testing phases

---

### 9. `09_branching_strategy.py` - Git Branching Strategies
**Concepts:**
- Git Flow branching model
- GitHub Flow (trunk-based)
- Branch lifecycle simulation
- Merge conflict scenarios

**Run:** `python 09_branching_strategy.py`

---

### 10. `10_gantt_chart.py` - Gantt Chart and Critical Path
**Concepts:**
- Critical Path Method (CPM): forward/backward pass
- Task dependency resolution
- Slack calculation and critical path identification
- ASCII Gantt chart rendering

**Run:** `python 10_gantt_chart.py`

---

### 11. `11_tech_debt_tracker.py` - Technical Debt Tracker
**Concepts:**
- Technical debt modeling (type, severity, interest rate)
- ROI-based prioritization for debt payoff
- Sprint simulation with greedy payoff strategy
- Debt report generation

**Run:** `python 11_tech_debt_tracker.py`

---

### 12. `12_process_improvement.py` - Process Improvement
**Concepts:**
- CMMI maturity level assessment
- GQM (Goal-Question-Metric) framework
- Root cause analysis (5 Whys, Fishbone/Ishikawa diagram)
- DORA metrics benchmarking (Elite/High/Medium/Low)
- PDCA (Plan-Do-Check-Act) improvement cycle

**Run:** `python 12_process_improvement.py`

---

### 13. `13_ci_cd_pipeline.yml` - GitHub Actions CI/CD Pipeline
**Concepts:**
- Multi-stage pipeline: lint, test, build, deploy
- Matrix strategy for multiple Python versions
- Dependency caching and artifact upload
- Environment protection rules for production
- Rolling deployment strategy

---

### 14. `14_adr_template.md` - Architecture Decision Records
**Concepts:**
- ADR template (Nygard format)
- Status, Context, Decision, Consequences structure
- Two complete example ADRs (database selection, microservices extraction)
- Guidelines for writing effective ADRs

---

### 15. `15_team_dynamics.py` - Team Dynamics and Communication
**Concepts:**
- Team Topologies (stream-aligned, platform, enabling, complicated-subsystem)
- Conway's Law analysis and misalignment detection
- Code review quality assessment
- Meeting effectiveness and cost analysis
- Psychological safety assessment (Google Project Aristotle)

**Run:** `python 15_team_dynamics.py`

---

### 16. `16_capstone_project.md` - Capstone Project
**Concepts:**
- Comprehensive project combining all 16 lessons
- End-to-end software engineering practice
