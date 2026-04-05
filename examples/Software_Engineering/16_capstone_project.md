# Software Engineering Capstone Project

A comprehensive project that applies all software engineering
concepts from Lessons 01-16.

## Project: Task Management System (TaskFlow)

Build a full-featured task management system applying SE principles
at every stage.

---

## Phase 1: Requirements Engineering (L03-L04)

### Stakeholders

| Stakeholder | Role | Key Concerns |
|-------------|------|-------------|
| Team Member | End user | Easy task creation, status tracking |
| Team Lead | Manager | Sprint planning, workload visibility |
| Admin | System admin | User management, system health |

### User Stories

```
US-001: As a team member, I want to create tasks with title,
        description, and priority so that I can track my work.
        Acceptance: Task appears in "To Do" column after creation.

US-002: As a team member, I want to move tasks between columns
        (To Do → In Progress → Done) so that I can update status.
        Acceptance: Drag-and-drop updates persist across refresh.

US-003: As a team lead, I want to assign tasks to team members
        so that work is distributed clearly.
        Acceptance: Assignee receives notification, task shows
        assignee avatar.

US-004: As a team lead, I want to view a sprint board with all
        team tasks so that I can track progress.
        Acceptance: Board shows tasks grouped by status with
        burndown chart.
```

### Non-Functional Requirements

- **NFR-001**: Page load time < 2 seconds (P95)
- **NFR-002**: Support 50 concurrent users
- **NFR-003**: Data encrypted at rest and in transit
- **NFR-004**: 99.5% uptime (monthly)

---

## Phase 2: Architecture & Design (L05)

### UML Class Diagram

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│    User      │     │    Task       │     │   Sprint     │
├─────────────┤     ├──────────────┤     ├──────────────┤
│ id           │     │ id            │     │ id           │
│ email        │     │ title         │     │ name         │
│ name         │     │ description   │     │ start_date   │
│ role         │     │ status        │     │ end_date     │
├─────────────┤     │ priority      │     │ goal         │
│ create_task()│     │ assignee_id   │     ├──────────────┤
│ assign_task()│     │ sprint_id     │     │ add_task()   │
└─────────────┘     │ created_at    │     │ burndown()   │
       │            ├──────────────┤     └──────────────┘
       │            │ move_to()     │            │
       └──────1:N──▶│ add_comment() │◀──N:1──────┘
                    └──────────────┘
                           │
                      1:N  │
                           ▼
                    ┌──────────────┐
                    │   Comment    │
                    ├──────────────┤
                    │ id           │
                    │ body         │
                    │ author_id    │
                    │ created_at   │
                    └──────────────┘
```

### API Design

```
POST   /api/tasks              Create task
GET    /api/tasks              List tasks (with filters)
GET    /api/tasks/:id          Get task details
PATCH  /api/tasks/:id          Update task
DELETE /api/tasks/:id          Delete task
POST   /api/tasks/:id/comments Add comment
GET    /api/sprints            List sprints
POST   /api/sprints            Create sprint
GET    /api/sprints/:id/board  Get sprint board
```

---

## Phase 3: Estimation (L06)

### Story Point Estimation

| Story | Estimate | Rationale |
|-------|----------|-----------|
| US-001 | 3 SP | Standard CRUD, form + validation |
| US-002 | 5 SP | Drag-and-drop, state management |
| US-003 | 3 SP | Assignment + notification |
| US-004 | 8 SP | Board view + burndown chart |

### Sprint Plan

- **Sprint 1** (2 weeks): US-001 + US-002 = 8 SP
- **Sprint 2** (2 weeks): US-003 + US-004 = 11 SP
- **Sprint 3** (2 weeks): Polish, testing, deployment

---

## Phase 4: Quality Assurance (L07-L08)

### Test Plan Summary

| Level | Scope | Tools | Coverage Target |
|-------|-------|-------|----------------|
| Unit | Models, services | pytest | 80% |
| Integration | API endpoints | pytest + httpx | All endpoints |
| E2E | User workflows | Playwright | Critical paths |
| Performance | Load testing | k6 | NFR-001, NFR-002 |

### Sample Test Cases

```python
# Unit test: Task creation
def test_create_task():
    task = Task(title="Fix bug", priority="high")
    assert task.status == "todo"
    assert task.priority == "high"

# Integration test: API endpoint
def test_create_task_api(client):
    resp = client.post("/api/tasks", json={
        "title": "Fix bug",
        "priority": "high"
    })
    assert resp.status_code == 201
    assert resp.json()["id"] is not None

# E2E test: Drag and drop
def test_move_task(page):
    page.goto("/board")
    task = page.locator("[data-task-id='1']")
    target = page.locator("[data-column='in_progress']")
    task.drag_to(target)
    expect(task).to_be_attached()
```

---

## Phase 5: Configuration Management (L09)

### Git Branching Strategy

Using GitHub Flow:

```
main ─────────●───────●───────●──── (production)
               \     /         \   /
feature/login   ●──●    feature/board ●──●
```

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest --cov=app tests/
      - run: ruff check app/
  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - run: docker build -t taskflow .
      - run: docker push registry/taskflow
```

---

## Phase 6: Project Management (L10)

### Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Scope creep | High | High | Strict sprint backlog |
| Key person dependency | Medium | High | Pair programming |
| Integration issues | Medium | Medium | CI + integration tests |

### Definition of Done

- [ ] Code reviewed (1 approval)
- [ ] Unit tests pass (≥80% coverage)
- [ ] Integration tests pass
- [ ] No lint warnings
- [ ] Documentation updated
- [ ] Deployed to staging
- [ ] Product owner accepted

---

## Deliverables Checklist

- [ ] Requirements document (user stories + NFRs)
- [ ] Architecture diagram (UML class + component)
- [ ] API specification (OpenAPI/Swagger)
- [ ] Sprint plan with estimates
- [ ] Test plan + test cases
- [ ] CI/CD pipeline configuration
- [ ] Working prototype (Sprint 1 scope)
- [ ] Deployment documentation
- [ ] Retrospective notes
