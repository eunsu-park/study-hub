# Test Plan Template

A reusable test plan template aligned with IEEE 829 and adapted
for modern agile/DevOps workflows.

## 1. Test Plan Identifier

- **Project**: [Project Name]
- **Version**: 1.0
- **Date**: YYYY-MM-DD
- **Author**: [Name]
- **Status**: Draft | Review | Approved

## 2. Introduction

### 2.1 Purpose

Describe the scope and objectives of this test plan.

### 2.2 Scope

| In Scope | Out of Scope |
|----------|-------------|
| Feature A: user registration | Third-party payment gateway |
| Feature B: search functionality | Mobile app (separate plan) |
| API endpoints v2 | Legacy API v1 |

### 2.3 References

- Requirements: [link to SRS or user stories]
- Design: [link to architecture docs]
- Previous test results: [link]

## 3. Test Items

| Item | Version | Source |
|------|---------|--------|
| User Service API | 2.1.0 | `services/user/` |
| Search Engine | 1.3.0 | `services/search/` |
| Web Frontend | 3.0.0 | `frontend/` |

## 4. Features to Be Tested

### 4.1 Functional

- [ ] User registration (email, OAuth)
- [ ] User login / logout
- [ ] Search by keyword, filter, sort
- [ ] Search result pagination

### 4.2 Non-Functional

- [ ] Response time < 200ms (P95) for search
- [ ] Support 1000 concurrent users
- [ ] Accessibility (WCAG 2.1 AA)

## 5. Features Not to Be Tested

- Admin dashboard (covered by separate plan)
- Email delivery (external service, use mock)

## 6. Approach

### 6.1 Test Levels

| Level | Description | Tools |
|-------|-------------|-------|
| Unit | Individual functions/methods | pytest, Jest |
| Integration | Service interactions, DB queries | pytest + testcontainers |
| System | End-to-end user flows | Playwright, Cypress |
| Performance | Load, stress, soak testing | k6, Locust |

### 6.2 Test Types

- **Smoke**: Critical path (login → search → view). Run on every deploy.
- **Regression**: Full suite. Run nightly and before release.
- **Exploratory**: Manual sessions (2h/week). Focus on new features.

### 6.3 Entry / Exit Criteria

**Entry Criteria:**
- Code merged to `develop` branch
- Build passes CI pipeline
- Test environment provisioned

**Exit Criteria:**
- All critical/high severity bugs resolved
- Code coverage ≥ 80%
- Performance SLAs met
- Zero open P1/P2 defects

## 7. Test Environment

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Browser  │───▶│  Nginx   │───▶│  App     │
│ (Cypress) │    │  (proxy) │    │ (Docker) │
└──────────┘    └──────────┘    └────┬─────┘
                                     │
                               ┌─────▼─────┐
                               │ PostgreSQL │
                               │  (Docker)  │
                               └───────────┘
```

| Component | Specification |
|-----------|--------------|
| OS | Ubuntu 22.04 (Docker) |
| Database | PostgreSQL 16 |
| Browser | Chrome (latest), Firefox (latest) |
| CI | GitHub Actions |

## 8. Test Data

| Data Set | Description | Source |
|----------|-------------|--------|
| Seed users | 100 test accounts | `fixtures/users.json` |
| Search corpus | 10,000 documents | `fixtures/search_data.sql` |
| Edge cases | Unicode, long strings, SQL injection | `fixtures/edge_cases.json` |

## 9. Schedule

| Phase | Start | End | Deliverable |
|-------|-------|-----|-------------|
| Test design | Week 1 | Week 2 | Test cases |
| Environment setup | Week 2 | Week 2 | Test env ready |
| Execution (Round 1) | Week 3 | Week 4 | Bug report |
| Bug fix + retest | Week 5 | Week 5 | Updated results |
| Final report | Week 6 | Week 6 | Sign-off |

## 10. Roles and Responsibilities

| Role | Person | Responsibilities |
|------|--------|-----------------|
| Test Lead | TBD | Plan, coordinate, report |
| QA Engineer | TBD | Write and execute test cases |
| Developer | TBD | Fix defects, support env setup |
| Product Owner | TBD | Accept/reject, priority decisions |

## 11. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Test env instability | Medium | High | Docker-compose reproducible env |
| Insufficient test data | Low | Medium | Generate synthetic data |
| Third-party API downtime | Medium | Medium | Mock external services |
| Scope creep | High | Medium | Strict change control |

## 12. Defect Management

### Severity Levels

| Level | Description | SLA |
|-------|-------------|-----|
| P1 - Critical | System crash, data loss | Fix within 4h |
| P2 - Major | Feature broken, no workaround | Fix within 24h |
| P3 - Minor | Feature degraded, workaround exists | Fix in next sprint |
| P4 - Trivial | Cosmetic, typo | Backlog |

### Bug Report Template

```markdown
**Title**: [Short description]
**Severity**: P1/P2/P3/P4
**Steps to Reproduce**:
1. ...
2. ...
3. ...
**Expected Result**: ...
**Actual Result**: ...
**Environment**: [browser, OS, API version]
**Screenshots/Logs**: [attach]
```

## 13. Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Test Lead | | | |
| Dev Lead | | | |
| Product Owner | | | |
