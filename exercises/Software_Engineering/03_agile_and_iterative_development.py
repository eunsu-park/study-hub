"""
Exercises for Lesson 03: Agile and Iterative Development
Topic: Software_Engineering

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Sprint Planning Agenda ===
# Problem: Write a detailed agenda for the first Sprint Planning meeting for
# a 5-person team building a mobile banking app (2-week sprints).

def exercise_1():
    """Sprint Planning agenda for the first sprint."""

    agenda = """
SPRINT PLANNING MEETING — Sprint 1
===================================
Team: 5-person mobile banking app team
Sprint Length: 2 weeks (10 working days)
Total Meeting Time: 4 hours (Scrum Guide recommends max 2 hrs per sprint week)

--- PART 1: WHAT CAN WE DO THIS SPRINT? (2 hours) ---

[0:00 - 0:20] OPENING AND CONTEXT (20 min)
  - Scrum Master: Welcome, introductions, explain Sprint Planning purpose
  - Product Owner: Present the Product Goal and the current state of the
    Product Backlog (top 15-20 items, ordered by priority)
  - Product Owner: Explain the Sprint Goal — the overarching objective for
    this Sprint (e.g., "Users can register, log in, and view account balance")

[0:20 - 0:50] PRODUCT BACKLOG WALKTHROUGH (30 min)
  - Product Owner walks through the top 10 items in detail
  - For each: what it is, why it matters, acceptance criteria
  - Team asks clarifying questions — this is NOT estimation, just understanding
  - Product Owner answers or captures unknowns as spike stories

[0:50 - 1:30] STORY SELECTION AND ESTIMATION (40 min)
  - Team reviews their capacity: 5 people × 10 days = 50 person-days
    (minus meetings, onboarding ramp-up for Sprint 1 → ~35 effective days)
  - Since Sprint 1 has no historical velocity, use gut feel + comparison
  - Team selects stories from top of backlog that they believe they can
    complete within the Sprint, working down until capacity feels full
  - Use Planning Poker or T-shirt sizing for quick relative estimates
  - Note: First sprint velocity will be used to calibrate future sprints

[1:30 - 2:00] SPRINT GOAL CONFIRMATION (30 min)
  - Confirm the Sprint Goal with the Product Owner
  - Read back the selected Product Backlog Items (PBIs)
  - Product Owner confirms: "Yes, if you deliver these, the Sprint Goal is met"
  - If too much/too little was selected, adjust now

--- BREAK (15 min) ---

--- PART 2: HOW WILL WE DO IT? (1 hour 45 min) ---

[2:15 - 3:30] TASK DECOMPOSITION (75 min)
  - For each selected PBI, the team identifies technical tasks:
    Example for "User Registration":
      - Design registration API endpoint (2 hrs)
      - Implement email validation service (4 hrs)
      - Create registration UI screen (6 hrs)
      - Write unit tests for registration logic (3 hrs)
      - Integration test: API + database (2 hrs)
  - Tasks should be small enough to complete in < 1 day
  - Identify dependencies between tasks
  - Assign initial owners (not binding — Scrum encourages swarming)

[3:30 - 3:50] DEFINITION OF DONE REVIEW (20 min)
  - Agree on the team's Definition of Done for Sprint 1:
    1. Code reviewed by at least one other team member
    2. Unit tests pass with >= 80% coverage
    3. Acceptance criteria verified
    4. No known critical bugs
    5. Deployed to staging environment
  - This is especially important for Sprint 1 — future sprints will inherit it

[3:50 - 4:00] CLOSING (10 min)
  - Scrum Master summarizes: Sprint Goal, selected PBIs, task board status
  - Confirm Daily Scrum time and location
  - Team commits to the Sprint Backlog
"""
    print(agenda)


# === Exercise 2: Kanban Board Design ===
# Problem: Design a Kanban board for a 3-person team handling both feature
# development and production support. Specify WIP limits and incident handling.

def exercise_2():
    """Kanban board for mixed feature/support team."""

    print("KANBAN BOARD: Feature Development + Production Support (3-person team)")
    print("=" * 75)

    columns = [
        {
            "name": "Backlog",
            "wip_limit": "No limit",
            "wip_rationale": "Incoming queue; limiting it would reject valid work items",
            "description": "Prioritized list of features and non-urgent support tasks",
        },
        {
            "name": "Ready",
            "wip_limit": 5,
            "wip_rationale": (
                "Enough refined items for 1-2 days of work. Too many means "
                "over-refinement; too few means idle developers."
            ),
            "description": "Items that are fully specified and ready to be pulled",
        },
        {
            "name": "In Progress",
            "wip_limit": 3,
            "wip_rationale": (
                "One item per person. Forces focus and prevents multitasking. "
                "With 3 people, WIP = 3 means each person works on exactly one "
                "item at a time."
            ),
            "description": "Actively being worked on (coding, designing)",
        },
        {
            "name": "Code Review",
            "wip_limit": 3,
            "wip_rationale": (
                "Matches 'In Progress' WIP. If this column fills up, the team "
                "should stop starting new work and review instead — 'stop starting, "
                "start finishing.'"
            ),
            "description": "Waiting for or undergoing peer review",
        },
        {
            "name": "Testing / QA",
            "wip_limit": 2,
            "wip_rationale": (
                "Slightly lower than In Progress because testing is faster per item "
                "and the team does their own testing. Prevents a testing bottleneck."
            ),
            "description": "Functional testing, regression checks",
        },
        {
            "name": "Done",
            "wip_limit": "No limit",
            "wip_rationale": "Completed items accumulate here until deployed or released",
            "description": "Deployed to production or resolved",
        },
    ]

    # Print the board
    for col in columns:
        wip = col["wip_limit"]
        wip_str = f"WIP: {wip}" if isinstance(wip, int) else f"WIP: {wip}"
        print(f"\n  [{col['name']}] ({wip_str})")
        print(f"    Purpose: {col['description']}")
        print(f"    WIP Rationale: {col['wip_rationale']}")

    # Swim lanes
    print("\n\nSWIM LANES:")
    print("-" * 40)
    print("  The board has TWO horizontal swim lanes:")
    print("  1. FEATURES    — New development work (standard priority)")
    print("  2. SUPPORT     — Production bugs and support tickets")
    print("  Both lanes share the same WIP limits (combined).")

    # Urgent incident handling
    incident_policy = """
URGENT PRODUCTION INCIDENT POLICY:
====================================
When a P1/P2 incident arrives mid-flow:

1. SIGNAL: The incident is placed in a special 'URGENT' tag/color (red card)
   at the top of the 'Ready' column.

2. SWARM: One developer immediately pulls the incident into 'In Progress',
   displacing their current work item if necessary. The displaced item moves
   back to 'Ready' (it does NOT stay in 'In Progress' — respect WIP limits).

3. WIP EXCEPTION: If all 3 slots in 'In Progress' are occupied, the team may
   temporarily exceed the WIP limit by 1 for the incident. This must be
   explicitly called out on the board (e.g., a red flag icon).

4. TIME-BOX: The incident has a 4-hour time-box for initial resolution. If
   unresolved, the team swarms: a second developer joins. This is NOT hidden
   work — both devs' cards are visible on the board.

5. POSTMORTEM: After resolution, the team adds a brief retrospective note to
   identify systemic improvements that could prevent recurrence.
"""
    print(incident_policy)


# === Exercise 3: Velocity Forecasting ===
# Problem: Velocities of 22, 19, 25, 21. Backlog has 180 points. Forecast
# completion. Then add a 45-point epic and re-forecast.

def exercise_3():
    """Velocity forecasting and backlog change analysis."""

    velocities = [22, 19, 25, 21]
    remaining_points = 180
    new_epic_points = 45

    avg_velocity = sum(velocities) / len(velocities)
    min_velocity = min(velocities)
    max_velocity = max(velocities)

    print("VELOCITY ANALYSIS AND FORECASTING")
    print("=" * 55)

    print(f"\nHistorical velocities: {velocities}")
    print(f"Average velocity:      {avg_velocity:.1f} points/sprint")
    print(f"Min velocity:          {min_velocity} points/sprint")
    print(f"Max velocity:          {max_velocity} points/sprint")
    print(f"Remaining backlog:     {remaining_points} points")

    # Forecasts (three-point: optimistic, most likely, pessimistic)
    sprints_optimistic = remaining_points / max_velocity
    sprints_likely = remaining_points / avg_velocity
    sprints_pessimistic = remaining_points / min_velocity

    print(f"\n--- ORIGINAL FORECAST (180 points) ---")
    print(f"  Optimistic  (@ {max_velocity} pts/sprint): {sprints_optimistic:.1f} sprints")
    print(f"  Most Likely (@ {avg_velocity:.1f} pts/sprint): {sprints_likely:.1f} sprints")
    print(f"  Pessimistic (@ {min_velocity} pts/sprint): {sprints_pessimistic:.1f} sprints")

    import math
    weeks_likely = math.ceil(sprints_likely) * 2  # 2-week sprints
    print(f"\n  Most likely completion: ~{math.ceil(sprints_likely)} sprints "
          f"({weeks_likely} weeks)")

    # After adding the new epic
    new_remaining = remaining_points + new_epic_points
    new_sprints_optimistic = new_remaining / max_velocity
    new_sprints_likely = new_remaining / avg_velocity
    new_sprints_pessimistic = new_remaining / min_velocity

    print(f"\n--- REVISED FORECAST (+ {new_epic_points}-point epic = {new_remaining} points) ---")
    print(f"  Optimistic  (@ {max_velocity} pts/sprint): {new_sprints_optimistic:.1f} sprints")
    print(f"  Most Likely (@ {avg_velocity:.1f} pts/sprint): {new_sprints_likely:.1f} sprints")
    print(f"  Pessimistic (@ {min_velocity} pts/sprint): {new_sprints_pessimistic:.1f} sprints")

    delta_sprints = new_sprints_likely - sprints_likely
    delta_weeks = math.ceil(delta_sprints) * 2
    print(f"\n  Impact: +{delta_sprints:.1f} sprints (~{delta_weeks} weeks delay)")
    print(f"  New most likely completion: ~{math.ceil(new_sprints_likely)} sprints "
          f"({math.ceil(new_sprints_likely) * 2} weeks)")

    info_needed = """
INFORMATION NEEDED FOR RELIABLE COMMITMENT:
  1. Has the new epic been refined? Story-point estimates for epics are
     notoriously inaccurate. The 45 points need to be broken into stories
     and estimated by the team.
  2. Does the new epic require new technical skills or dependencies that
     could slow the team down (reducing velocity)?
  3. Are any team members leaving or joining? Velocity is team-specific.
  4. Does the Product Owner want to de-scope existing backlog items to
     accommodate the new epic (trade-off), or is it purely additive?
  5. What is the confidence level needed? A range (8-12 sprints) is more
     honest than a single number.

RECOMMENDATION TO PRODUCT OWNER:
  "Based on our average velocity of 21.8 points/sprint, adding 45 points
  extends the project by approximately 2 sprints (4 weeks). Our best case
  is 9 sprints, worst case is 12 sprints. I recommend we refine the new
  epic next sprint to get more accurate estimates before committing to a
  date."
"""
    print(info_needed)


# === Exercise 4: Agile Manifesto Principle Violations ===
# Problem: Identify which of the 12 principles are violated by given anti-patterns.

def exercise_4():
    """Identify Agile Manifesto principle violations."""

    anti_patterns = [
        {
            "label": "(a) Management asks developers to report Sprint progress "
                     "in a weekly status meeting",
            "violated_principles": [
                (5, "Build projects around motivated individuals. Give them the "
                    "environment and support they need, and trust them to get the "
                    "job done.",
                 "A mandated weekly status report to management signals DISTRUST. "
                 "Agile trusts the team to self-manage. The Daily Scrum is the "
                 "team's own coordination mechanism -- adding a management "
                 "reporting layer implies the team cannot be trusted."),
                (11, "The best architectures, requirements, and designs emerge "
                     "from self-organizing teams.",
                 "Requiring external reporting disrupts self-organization. The "
                 "team should decide how to communicate progress (e.g., Sprint "
                 "Review is the appropriate ceremony for stakeholder visibility)."),
            ],
            "better_alternative": (
                "Invite management to attend the Sprint Review (every 2 weeks) "
                "where the team demonstrates working software. Use an information "
                "radiator (visible Kanban board, burndown chart) so management "
                "can check progress without interrupting the team."
            ),
        },
        {
            "label": "(b) The team skips the Sprint Retrospective because "
                     "they are behind schedule",
            "violated_principles": [
                (12, "At regular intervals, the team reflects on how to become "
                     "more effective, then tunes and adjusts its behavior "
                     "accordingly.",
                 "This principle DIRECTLY mandates the retrospective. Skipping "
                 "it eliminates the team's primary mechanism for continuous "
                 "improvement. Ironically, they are behind schedule and the "
                 "retrospective is exactly where they would identify WHY and "
                 "how to fix it."),
                (8, "Agile processes promote sustainable development. The "
                    "sponsors, developers, and users should be able to maintain "
                    "a constant pace indefinitely.",
                 "Skipping process improvements to 'go faster' is a sign of "
                 "unsustainable pace. The team is sacrificing long-term "
                 "effectiveness for short-term velocity."),
            ],
            "better_alternative": (
                "Time-box the retrospective to 30 minutes instead of skipping "
                "it entirely. Focus on one actionable improvement. Being behind "
                "schedule makes reflection MORE important, not less."
            ),
        },
        {
            "label": "(c) The Product Owner is unavailable and delegates to "
                     "a BA who cannot make decisions",
            "violated_principles": [
                (4, "Business people and developers must work together daily "
                    "throughout the project.",
                 "The Product Owner IS the business representative. Delegating "
                 "to a BA who cannot make decisions means business people are "
                 "NOT working with the team -- there is a proxy who creates "
                 "delays and potential miscommunication."),
                (6, "The most efficient and effective method of conveying "
                    "information to and within a development team is face-to-face "
                    "conversation.",
                 "An absent PO forces the team to communicate through a proxy, "
                 "reducing information bandwidth. The BA becomes a bottleneck "
                 "and a filter that distorts requirements."),
            ],
            "better_alternative": (
                "The PO must be available for at least Sprint Planning, Sprint "
                "Review, and backlog refinement. If the PO is too busy, either "
                "(a) empower the BA to make binding decisions, or (b) split the "
                "PO role between two people who share authority. An absent, "
                "indecisive PO is worse than no PO at all."
            ),
        },
    ]

    print("AGILE MANIFESTO PRINCIPLE VIOLATIONS")
    print("=" * 65)

    for ap in anti_patterns:
        print(f"\nAnti-Pattern: {ap['label']}")
        print("-" * 65)
        for principle_num, principle_text, explanation in ap["violated_principles"]:
            print(f"\n  Violated Principle #{principle_num}:")
            print(f"    \"{principle_text}\"")
            print(f"    Analysis: {explanation}")
        print(f"\n  Better Alternative: {ap['better_alternative']}")


# === Exercise 5: SAFe vs LeSS Comparison ===
# Problem: Compare SAFe and LeSS for 50 teams at a large insurance company.

def exercise_5():
    """Compare SAFe and LeSS for scaling agile at a large insurance company."""

    comparison = """
SAFe vs LeSS: SCALING 50 TEAMS AT AN INSURANCE COMPANY
========================================================

CONTEXT: A large insurance company has 50 development teams working on a
single enterprise policy management platform. This is a significant scaling
challenge.

--- SAFe (Scaled Agile Framework) ---

Structure:
  - SAFe organizes teams into Agile Release Trains (ARTs) of 5-12 teams each.
  - For 50 teams: ~5 ARTs, coordinated through a Solution Train.
  - This is SAFe's "Full SAFe" or "Large Solution" configuration.

Roles Added:
  - Release Train Engineer (RTE) per ART
  - Solution Train Engineer
  - Solution Architect
  - Epic Owners
  - Business Owners

Ceremonies:
  - PI (Program Increment) Planning: All teams in an ART plan together
    for 8-12 weeks. A 2-day event, in-person.
  - System Demos every 2 weeks, Inspect & Adapt every PI.

Strengths for This Context:
  1. Provides a clear organizational structure for 50 teams — roles and
     responsibilities are well-defined.
  2. Insurance companies have heavy compliance requirements; SAFe's
     documentation and governance practices align well.
  3. Easier executive buy-in because SAFe is a defined framework with
     training and certification (SAFe badges give management confidence).
  4. Handles portfolio-level prioritization of large investment themes.

Weaknesses:
  1. Heavyweight: Adds many roles, ceremonies, and artifacts — teams may
     feel constrained rather than agile.
  2. Can become "Waterfall in Agile clothing" if implemented mechanically.
  3. High cost: training, tooling (Jira Align), and consultant fees.


--- LeSS (Large-Scale Scrum) ---

Structure:
  - LeSS Huge: For >8 teams, organizes around Requirement Areas.
  - Each Requirement Area has ~4-8 teams and one Area Product Owner.
  - For 50 teams: ~7-8 Requirement Areas (e.g., Underwriting, Claims,
    Billing, Compliance, Customer Portal, Reporting, Integrations).
  - One overall Product Owner + Area Product Owners.

Roles Added:
  - Area Product Owners (one per Requirement Area)
  - No new management roles — existing Scrum roles are reused.

Ceremonies:
  - Same Scrum events, with multi-team coordination through:
    - Overall Sprint Planning 1 (cross-area alignment)
    - Sprint Planning 2 (team-level)
    - Overall Retrospective + team retrospectives

Strengths for This Context:
  1. Simpler framework: Fewer roles and ceremonies mean less overhead.
  2. Truly empowers teams: No additional management layers.
  3. Aligns with Scrum values more closely — less process accumulation.
  4. Cheaper to adopt: No expensive certifications or tooling required.

Weaknesses:
  1. Requires radical organizational change: LeSS demands that management
     give up control. An insurance company's traditional hierarchy will resist.
  2. Less guidance for portfolio management and large-scale coordination.
  3. Finding one Product Owner who understands the entire policy platform
     (across underwriting, claims, billing) is extremely difficult.
  4. Fewer guardrails: Teams must be highly mature to self-organize at scale.


--- KEY TRADE-OFFS ---

| Dimension               | SAFe                        | LeSS                       |
|--------------------------|-----------------------------|-----------------------------|
| Organizational change    | Moderate (adds layers)      | Radical (removes layers)    |
| Framework complexity     | High                        | Low                         |
| Management acceptance    | Easier (defined roles)      | Harder (demands trust)      |
| Team autonomy            | Moderate                    | High                        |
| Compliance fit           | Strong (built-in governance)| Weaker (must be added)      |
| Cost of adoption         | High                        | Low                         |
| Risk of "fake agile"     | Higher (process theater)    | Lower (but failure = chaos) |


--- RECOMMENDATION ---

For THIS context (50 teams, insurance company, enterprise platform):

RECOMMENDED: SAFe (Full SAFe / Large Solution) — with caveats.

RATIONALE:
1. Insurance companies are heavily regulated. SAFe's governance structures
   (compliance, auditing, documentation) align with regulatory requirements.
2. The existing organizational culture is almost certainly hierarchical.
   LeSS requires flattening the hierarchy, which is politically infeasible
   in a large insurer.
3. 50 teams need explicit coordination mechanisms. SAFe's PI Planning and
   ART structure provide these out of the box.

ORGANIZATIONAL CHANGES REQUIRED:
1. Create 5 Agile Release Trains organized around business domains
   (Underwriting ART, Claims ART, Billing ART, Digital ART, Platform ART).
2. Appoint RTEs and a Solution Train Engineer.
3. Train ALL 50 teams in SAFe (this is a 6-12 month effort).
4. Establish a Lean Portfolio Management function.
5. Convert project managers to RTEs or Scrum Masters (role transition plan).
6. Build a DevOps pipeline that supports continuous integration across ARTs.

CAVEAT: Adopt SAFe's structure but fight "process theater" — regularly
assess whether ceremonies are delivering value, not just checking boxes.
"""
    print(comparison)


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Sprint Planning Agenda ===")
    print("=" * 60)
    exercise_1()
    print("\n" + "=" * 60)
    print("=== Exercise 2: Kanban Board Design ===")
    print("=" * 60)
    exercise_2()
    print("\n" + "=" * 60)
    print("=== Exercise 3: Velocity Forecasting ===")
    print("=" * 60)
    exercise_3()
    print("\n" + "=" * 60)
    print("=== Exercise 4: Agile Principle Violations ===")
    print("=" * 60)
    exercise_4()
    print("\n" + "=" * 60)
    print("=== Exercise 5: SAFe vs LeSS Comparison ===")
    print("=" * 60)
    exercise_5()
    print("\nAll exercises completed!")
