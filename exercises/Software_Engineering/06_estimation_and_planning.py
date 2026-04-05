"""
Exercises for Lesson 06: Estimation and Planning
Topic: Software_Engineering

Solutions to practice problems from the lesson.
This lesson has two exercise sections: Practice Exercises (5) and Exercises (5).
Heavy on calculations: PERT, Function Points, Critical Path, COCOMO, Story Points.
"""

import math
from typing import Dict, List, Tuple


# =====================================================================
# PRACTICE EXERCISES (Section 15)
# =====================================================================

# === Practice Exercise 1: PERT Estimation ===
# Problem: Three sequential tasks with O/M/P estimates. Calculate PERT
# expected duration, std dev, 90% confidence interval, and Z-score.

def practice_exercise_1():
    """PERT estimation with confidence intervals."""

    print("PRACTICE EXERCISE 1: PERT Estimation")
    print("=" * 60)

    tasks = [
        ("Database migration", 1, 2, 6),
        ("API implementation", 3, 5, 10),
        ("Frontend integration", 2, 4, 9),
    ]

    print("\n  (a) PERT Expected Duration and Standard Deviation per Task:")
    print(f"  {'Task':<25} {'O':>3} {'M':>3} {'P':>3} {'E(t)':>7} {'SD':>6}")
    print(f"  {'-' * 50}")

    total_expected = 0
    total_variance = 0

    for name, o, m, p in tasks:
        # PERT formula: E(t) = (O + 4M + P) / 6
        expected = (o + 4 * m + p) / 6
        # Standard deviation: SD = (P - O) / 6
        sd = (p - o) / 6
        variance = sd ** 2
        total_expected += expected
        total_variance += variance
        print(f"  {name:<25} {o:>3} {m:>3} {p:>3} {expected:>7.2f} {sd:>6.2f}")

    total_sd = math.sqrt(total_variance)

    print(f"\n  (b) Total Project Duration (sequential tasks):")
    print(f"      Total Expected Duration = {total_expected:.2f} days")
    print(f"      Total Variance = {total_variance:.4f}")
    print(f"      Total Standard Deviation = sqrt({total_variance:.4f}) = {total_sd:.2f} days")

    # 90% confidence: Z = 1.28
    z_90 = 1.28
    ci_90_lower = total_expected - z_90 * total_sd
    ci_90_upper = total_expected + z_90 * total_sd

    print(f"\n  (c) 90% Confidence Interval:")
    print(f"      Z-score for 90% = {z_90}")
    print(f"      CI = E(t) +/- Z * SD = {total_expected:.2f} +/- {z_90} * {total_sd:.2f}")
    print(f"      CI = [{ci_90_lower:.2f}, {ci_90_upper:.2f}] days")
    print(f"      We are 90% confident the project completes between "
          f"{ci_90_lower:.1f} and {ci_90_upper:.1f} days.")

    # (d) Z-score for 13 days
    committed_days = 13
    z_score = (committed_days - total_expected) / total_sd

    print(f"\n  (d) Probability of completing in {committed_days} days:")
    print(f"      Z = (X - E) / SD = ({committed_days} - {total_expected:.2f}) / {total_sd:.2f} "
          f"= {z_score:.2f}")

    # Approximate probability from Z-score (using normal distribution)
    # Z = 1.10 -> ~86.4%
    # Use a simple lookup for common Z values
    z_table = {
        0.0: 0.5000, 0.5: 0.6915, 1.0: 0.8413, 1.1: 0.8643,
        1.2: 0.8849, 1.28: 0.8997, 1.5: 0.9332, 2.0: 0.9772,
    }
    # Find closest Z
    closest_z = min(z_table.keys(), key=lambda x: abs(x - z_score))
    prob = z_table[closest_z]

    print(f"      From Z-table: Z={z_score:.2f} -> approximately {prob*100:.1f}% probability")
    print(f"      The manager's commitment of {committed_days} days has roughly "
          f"a {prob*100:.0f}% chance of success.")


# === Practice Exercise 2: Function Point Counting ===
# Problem: Online poll application — count UFP and estimate effort.

def practice_exercise_2():
    """Function Point counting for an online poll application."""

    print("PRACTICE EXERCISE 2: Function Point Counting")
    print("=" * 60)

    # FP weights (Low, Average, High) for each type
    weights = {
        "EI":  {"Low": 3, "Average": 4, "High": 6},
        "EO":  {"Low": 4, "Average": 5, "High": 7},
        "EQ":  {"Low": 3, "Average": 4, "High": 6},
        "ILF": {"Low": 7, "Average": 10, "High": 15},
        "EIF": {"Low": 5, "Average": 7, "High": 10},
    }

    components = [
        ("Create poll form", "EI", "Average", "Input: question + up to 10 options"),
        ("Cast vote form", "EI", "Low", "Input: poll ID + option choice"),
        ("View results page", "EO", "Average", "Output: bar chart of votes"),
        ("Search polls by keyword", "EQ", "Low", "Input/output, no data change"),
        ("Polls table", "ILF", "Average", "Maintained by system"),
        ("Users table", "ILF", "Low", "Maintained by system"),
        ("Authentication service", "EIF", "Low", "External, referenced only"),
    ]

    print(f"\n  {'Component':<30} {'Type':<5} {'Complexity':<10} {'Weight':>6} {'FP':>4}")
    print(f"  {'-' * 60}")

    total_ufp = 0
    for name, fp_type, complexity, desc in components:
        weight = weights[fp_type][complexity]
        total_ufp += weight
        print(f"  {name:<30} {fp_type:<5} {complexity:<10} {weight:>6} {weight:>4}")

    print(f"  {'-' * 60}")
    print(f"  {'Total Unadjusted Function Points (UFP)':<50} {total_ufp:>6}")

    productivity = 8  # FP/person-month
    effort = total_ufp / productivity

    print(f"\n  Historical productivity: {productivity} FP/person-month")
    print(f"  Estimated effort: {total_ufp} / {productivity} = {effort:.1f} person-months")
    print(f"  With 2 developers: ~{effort / 2:.1f} calendar months")


# === Practice Exercise 3: Critical Path Method ===
# Problem: Find critical path for a 7-task project network.

def practice_exercise_3():
    """Critical Path Method calculation."""

    print("PRACTICE EXERCISE 3: Critical Path Method")
    print("=" * 60)

    # Tasks: (name, duration, predecessors)
    tasks = {
        "A": {"duration": 4, "predecessors": []},
        "B": {"duration": 6, "predecessors": []},
        "C": {"duration": 3, "predecessors": ["A"]},
        "D": {"duration": 5, "predecessors": ["A", "B"]},
        "E": {"duration": 4, "predecessors": ["C", "D"]},
        "F": {"duration": 2, "predecessors": ["D"]},
        "G": {"duration": 3, "predecessors": ["E", "F"]},
    }

    # (a) Network diagram (textual)
    print("\n  (a) Network Diagram:")
    print("      Start --> A --> C ---------> E --> G --> End")
    print("        |                          ^       ^")
    print("        +-----> B --> D ---------->+       |")
    print("                     |                     |")
    print("                     +-------> F ----------+")

    # (b) Forward pass: compute ES and EF
    es = {}
    ef = {}
    for task_name in ["A", "B", "C", "D", "E", "F", "G"]:
        t = tasks[task_name]
        if not t["predecessors"]:
            es[task_name] = 0
        else:
            es[task_name] = max(ef[p] for p in t["predecessors"])
        ef[task_name] = es[task_name] + t["duration"]

    project_duration = ef["G"]

    # Backward pass: compute LF and LS
    lf = {}
    ls = {}
    successors = {t: [] for t in tasks}
    for task_name, t in tasks.items():
        for p in t["predecessors"]:
            successors[p].append(task_name)

    for task_name in reversed(["A", "B", "C", "D", "E", "F", "G"]):
        if not successors[task_name]:
            lf[task_name] = project_duration
        else:
            lf[task_name] = min(ls[s] for s in successors[task_name])
        ls[task_name] = lf[task_name] - tasks[task_name]["duration"]

    # Float
    float_vals = {t: ls[t] - es[t] for t in tasks}

    print(f"\n  (b) Forward and Backward Pass Results:")
    print(f"  {'Task':<6} {'Dur':>4} {'ES':>4} {'EF':>4} {'LS':>4} {'LF':>4} {'Float':>6} {'Critical?'}")
    print(f"  {'-' * 50}")
    for t in ["A", "B", "C", "D", "E", "F", "G"]:
        is_critical = "YES" if float_vals[t] == 0 else ""
        print(f"  {t:<6} {tasks[t]['duration']:>4} {es[t]:>4} {ef[t]:>4} "
              f"{ls[t]:>4} {lf[t]:>4} {float_vals[t]:>6} {is_critical}")

    # (c) Critical path
    critical_path = [t for t in tasks if float_vals[t] == 0]
    print(f"\n  (c) Critical Path: {' -> '.join(critical_path)}")
    print(f"      Project Duration: {project_duration} days")

    # (d) If task D is delayed by 2 days
    print(f"\n  (d) If task D is delayed by 2 days:")
    d_new_ef = ef["D"] + 2  # D finishes at day 13 instead of 11
    e_new_es = max(ef["C"], d_new_ef)  # E depends on C and D
    e_new_ef = e_new_es + tasks["E"]["duration"]
    f_new_es = d_new_ef
    f_new_ef = f_new_es + tasks["F"]["duration"]
    g_new_es = max(e_new_ef, f_new_ef)
    g_new_ef = g_new_es + tasks["G"]["duration"]

    print(f"      D now finishes at day {d_new_ef} (was {ef['D']})")
    print(f"      E now starts at day {e_new_es}, finishes at day {e_new_ef}")
    print(f"      G now finishes at day {g_new_ef}")
    print(f"      New project duration: {g_new_ef} days (was {project_duration})")
    print(f"      Impact: +{g_new_ef - project_duration} days delay")
    print(f"      D was on the critical path, so the delay passes through fully.")


# === Practice Exercise 4: Release Planning ===
# Problem: 240 story points, velocity [28, 32, 30, 26], release planning.

def practice_exercise_4():
    """Velocity-based release planning."""

    print("PRACTICE EXERCISE 4: Release Planning with Velocity")
    print("=" * 60)

    velocities = [28, 32, 30, 26]
    total_backlog = 240
    v1_scope = 100
    sprint_weeks = 2

    avg_velocity = sum(velocities) / len(velocities)

    print(f"\n  (a) Average Velocity:")
    print(f"      Velocities: {velocities}")
    print(f"      Average = ({' + '.join(str(v) for v in velocities)}) / {len(velocities)} "
          f"= {avg_velocity:.0f} points/sprint")

    sprints_total = total_backlog / avg_velocity
    months_total = math.ceil(sprints_total) * sprint_weeks / 4.33

    print(f"\n  (b) Estimated Sprints to Complete Full Backlog:")
    print(f"      {total_backlog} / {avg_velocity:.0f} = {sprints_total:.1f} sprints")
    print(f"      Rounded up: {math.ceil(sprints_total)} sprints "
          f"= {math.ceil(sprints_total) * sprint_weeks} weeks "
          f"= ~{months_total:.1f} months")

    sprints_v1 = v1_scope / avg_velocity
    print(f"\n  (c) Sprints for v1.0 ({v1_scope} story points):")
    print(f"      {v1_scope} / {avg_velocity:.0f} = {sprints_v1:.2f} sprints")
    print(f"      Rounded up: {math.ceil(sprints_v1)} sprints "
          f"= {math.ceil(sprints_v1) * sprint_weeks} weeks")

    # (d) One sprint with 20% reduced velocity
    reduced_velocity = avg_velocity * 0.80
    # Sprint 1: normal, Sprint 2: reduced, Sprint 3: normal, Sprint 4: normal
    # Best case: 4 sprints. 3 at normal + 1 at reduced
    remaining_after_3_normal = v1_scope - (3 * avg_velocity)
    if remaining_after_3_normal <= 0:
        scenario = f"All {v1_scope} points completed in 3 normal sprints"
    else:
        scenario = f"After 3 normal sprints: {3 * avg_velocity:.0f} points done"

    # More realistic: one slow sprint within the first ceil(sprints_v1) sprints
    normal_sprints = math.ceil(sprints_v1) - 1
    total_with_reduction = normal_sprints * avg_velocity + reduced_velocity
    print(f"\n  (d) Impact of one sprint at 20% reduced velocity:")
    print(f"      Normal velocity: {avg_velocity:.0f} pts/sprint")
    print(f"      Reduced velocity: {reduced_velocity:.0f} pts/sprint")
    print(f"      Without reduction: {math.ceil(sprints_v1)} sprints × {avg_velocity:.0f} = "
          f"{math.ceil(sprints_v1) * avg_velocity:.0f} points capacity")
    print(f"      With one reduced sprint: "
          f"{normal_sprints} × {avg_velocity:.0f} + 1 × {reduced_velocity:.0f} = "
          f"{total_with_reduction:.0f} points capacity")

    if total_with_reduction >= v1_scope:
        print(f"      {total_with_reduction:.0f} >= {v1_scope}: "
              f"v1.0 still fits in {math.ceil(sprints_v1)} sprints")
        print(f"      No change to release date — the buffer absorbed the hit.")
    else:
        shortfall = v1_scope - total_with_reduction
        print(f"      Shortfall: {shortfall:.0f} points")
        print(f"      Need 1 additional sprint: "
              f"v1.0 now takes {math.ceil(sprints_v1) + 1} sprints "
              f"(+{sprint_weeks} weeks)")


# === Practice Exercise 5: Estimation Pitfalls ===
# Problem: Identify pitfalls and suggest mitigations for 5 scenarios.

def practice_exercise_5():
    """Identify estimation pitfalls and mitigations."""

    print("PRACTICE EXERCISE 5: Estimation Pitfalls")
    print("=" * 60)

    scenarios = [
        {
            "description": (
                "(a) PM announces 6-month deadline in kickoff before any estimation. "
                "All subsequent estimates cluster around 6 months."
            ),
            "pitfall": "ANCHORING BIAS",
            "explanation": (
                "The 6-month number acts as a cognitive anchor. Even when teams "
                "try to estimate independently, the announced number biases their "
                "judgment toward that value."
            ),
            "mitigation": (
                "Have the team estimate BEFORE revealing any target dates. Use "
                "blind estimation techniques (Planning Poker where estimates are "
                "revealed simultaneously). If a target date exists, frame it as "
                "'this is the business constraint' separately from 'this is the "
                "engineering estimate.'"
            ),
        },
        {
            "description": (
                "(b) Team reports database migration as '95% complete' for three "
                "consecutive weeks."
            ),
            "pitfall": "90% SYNDROME (Ninety-Ninety Rule)",
            "explanation": (
                "The first 90% of the code accounts for the first 90% of development "
                "time. The remaining 10% accounts for the other 90% of time. The team "
                "underestimated the difficulty of the remaining edge cases, data "
                "validation, and integration work."
            ),
            "mitigation": (
                "Track progress by working, tested features — not perceived percentage. "
                "Decompose remaining work into specific tasks with hour estimates. "
                "Use burn-down charts that show actual work remaining, not subjective "
                "completion percentages."
            ),
        },
        {
            "description": (
                "(c) Developer working alone waits until day 9 of a 10-day sprint "
                "to start coding."
            ),
            "pitfall": "PARKINSON'S LAW + STUDENT SYNDROME",
            "explanation": (
                "Parkinson's Law: work expands to fill time available. "
                "Student Syndrome: delaying start until just before the deadline. "
                "The developer procrastinates knowing there is a full sprint."
            ),
            "mitigation": (
                "Break work into daily deliverables tracked in Daily Scrum. "
                "Pair programming eliminates solo procrastination. Set intermediate "
                "milestones (e.g., 'PR ready for review by day 7')."
            ),
        },
        {
            "description": (
                "(d) Manager adds 3 new developers to a slipping project with 2 weeks left."
            ),
            "pitfall": "BROOKS'S LAW",
            "explanation": (
                "'Adding manpower to a late software project makes it later.' "
                "New developers need onboarding, existing developers must spend time "
                "mentoring, and communication overhead increases quadratically."
            ),
            "mitigation": (
                "Do NOT add people this late. Instead: (1) reduce scope to fit the "
                "deadline, (2) extend the deadline, or (3) accept lower quality "
                "(consciously, not by accident). If people must be added, give them "
                "independent, well-defined tasks with minimal cross-dependencies."
            ),
        },
        {
            "description": (
                "(e) Team estimates based on best-case, ignoring that integration "
                "testing has repeatedly caused blocking bugs."
            ),
            "pitfall": "PLANNING FALLACY (Optimism Bias)",
            "explanation": (
                "Teams overweight the ideal scenario and underweight historical evidence "
                "of delays. Despite KNOWING that integration testing causes problems, "
                "they estimate as if 'this time it will go smoothly.'"
            ),
            "mitigation": (
                "Use REFERENCE CLASS FORECASTING: base estimates on what ACTUALLY "
                "happened in past sprints, not what you hope will happen. Add explicit "
                "buffer for integration testing based on historical data (e.g., 'our "
                "last 5 releases needed 3-5 days of integration debugging')."
            ),
        },
    ]

    for sc in scenarios:
        print(f"\n  {sc['description']}")
        print(f"    Pitfall: {sc['pitfall']}")
        print(f"    Why: {sc['explanation']}")
        print(f"    Mitigation: {sc['mitigation']}")


# =====================================================================
# EXERCISES (End of Lesson)
# =====================================================================

# === Exercise 1: Cone of Uncertainty ===
# Problem: PM asked to commit to launch date at project start.

def exercise_1():
    """Apply the Cone of Uncertainty."""

    print("EXERCISE 1: Cone of Uncertainty")
    print("=" * 60)

    print("""
  (a) Accuracy Range at Project Start:
      At the initial concept stage (rough user stories, no architecture),
      the Cone of Uncertainty indicates estimates can be off by a factor
      of 0.25x to 4x. If the team's gut estimate is 6 months:
        - Best case: 6 × 0.25 = 1.5 months
        - Worst case: 6 × 4 = 24 months
      This 16:1 range is too wide for any meaningful commitment.

  (b) Milestones to Narrow to +/- 2x:
      The cone narrows to approximately 2x after:
      1. Requirements are defined and baselined (SRS or refined backlog)
      2. High-level architecture is complete (technology choices made,
         major components identified, interfaces defined)
      3. At least one iteration/spike has been completed (team has built
         something real and measured actual velocity)
      Typically this corresponds to completing the "Approved Product
      Definition" milestone — roughly 20-30% into the project.

  (c) Response to Stakeholder Demanding a Firm Date:
      "I understand the need for a date to plan around. At this stage,
      our estimate has an accuracy range of 0.25x to 4x — meaning a
      6-month gut estimate could actually take anywhere from 2 to 24
      months. Committing to a specific date now creates a false sense
      of certainty that will lead to disappointment. What I CAN offer:
      (1) A rough range today: 4-12 months. (2) A commitment to
      deliver a refined estimate with +/-25% accuracy within 6 weeks,
      after we complete requirements analysis and one development
      spike. (3) Monthly re-forecasts as the cone narrows. This gives
      you increasingly precise planning data without the risk of an
      unrealistic commitment."
""")


# === Exercise 2: COCOMO Estimation ===
# Problem: Organic project, 24 KLOC -> effort, schedule, team size.

def exercise_2():
    """COCOMO basic model calculations."""

    print("EXERCISE 2: COCOMO Basic Model")
    print("=" * 60)

    # COCOMO Basic Model coefficients for Organic projects:
    # Effort = a * (KLOC)^b  where a=2.4, b=1.05
    # Schedule = c * (Effort)^d  where c=2.5, d=0.38
    a, b = 2.4, 1.05
    c, d = 2.5, 0.38

    kloc_1 = 24
    effort_1 = a * (kloc_1 ** b)
    schedule_1 = c * (effort_1 ** d)
    team_1 = effort_1 / schedule_1

    print(f"\n  COCOMO Basic Model — Organic Project")
    print(f"  Coefficients: a={a}, b={b}, c={c}, d={d}")

    print(f"\n  (a) Effort for {kloc_1} KLOC:")
    print(f"      Effort = {a} × ({kloc_1})^{b}")
    print(f"      Effort = {a} × {kloc_1 ** b:.2f}")
    print(f"      Effort = {effort_1:.1f} person-months")

    print(f"\n  (b) Schedule Duration:")
    print(f"      Schedule = {c} × ({effort_1:.1f})^{d}")
    print(f"      Schedule = {c} × {effort_1 ** d:.2f}")
    print(f"      Schedule = {schedule_1:.1f} months")

    print(f"\n  (c) Average Team Size:")
    print(f"      Team = Effort / Schedule = {effort_1:.1f} / {schedule_1:.1f}")
    print(f"      Team = {team_1:.1f} people")

    # (d) Scope grows to 40 KLOC
    kloc_2 = 40
    effort_2 = a * (kloc_2 ** b)
    schedule_2 = c * (effort_2 ** d)
    team_2 = effort_2 / schedule_2

    size_increase_pct = ((kloc_2 - kloc_1) / kloc_1) * 100
    effort_increase_pct = ((effort_2 - effort_1) / effort_1) * 100

    print(f"\n  (d) Scope grows to {kloc_2} KLOC:")
    print(f"      New Effort = {a} × ({kloc_2})^{b} = {effort_2:.1f} person-months")
    print(f"      New Schedule = {schedule_2:.1f} months")
    print(f"      New Team Size = {team_2:.1f} people")
    print(f"\n      Size increase: {size_increase_pct:.0f}%")
    print(f"      Effort increase: {effort_increase_pct:.1f}%")
    print(f"      Effort increased by {effort_increase_pct:.0f}% for a {size_increase_pct:.0f}% "
          f"size increase.")
    print(f"      This demonstrates the SUPERLINEAR relationship between size and effort:")
    print(f"      because b={b} > 1.0, effort grows faster than size.")


# === Exercise 3: Sprint Planning with Story Points ===
# Problem: 8 stories, velocity 34, plan sprints.

def exercise_3():
    """Sprint planning with story points."""

    print("EXERCISE 3: Sprint Planning with Story Points")
    print("=" * 60)

    velocity = 34
    stories = [
        ("User registration", 5),
        ("Email verification", 3),
        ("Profile editing", 8),
        ("Password reset", 5),
        ("Two-factor authentication", 13),
        ("OAuth login (Google)", 8),
        ("Account deletion", 3),
        ("Admin user management", 13),
    ]

    total_points = sum(pts for _, pts in stories)
    print(f"\n  Team velocity: {velocity} points/sprint")
    print(f"  Total backlog: {total_points} points")
    print(f"\n  Stories (priority order):")
    for name, pts in stories:
        print(f"    {name:<30} {pts:>3} points")

    # (a) Sprint 1 selection (top-down by priority)
    sprint_1 = []
    sprint_1_total = 0
    remaining = list(stories)

    for name, pts in stories:
        if sprint_1_total + pts <= velocity:
            sprint_1.append((name, pts))
            sprint_1_total += pts

    print(f"\n  (a) Sprint 1 Selection (capacity = {velocity} points):")
    for name, pts in sprint_1:
        print(f"    [x] {name:<30} {pts:>3}")
    print(f"    {'Total:':<30} {sprint_1_total:>3} / {velocity}")

    # What didn't fit
    sprint_1_names = {n for n, _ in sprint_1}
    not_selected = [(n, p) for n, p in stories if n not in sprint_1_names]
    print(f"    Remaining: {', '.join(f'{n}({p})' for n, p in not_selected)}")

    # (b) How many sprints for all stories
    sprints_needed = math.ceil(total_points / velocity)
    print(f"\n  (b) Sprints for all stories:")
    print(f"      {total_points} / {velocity} = {total_points / velocity:.2f} -> "
          f"{sprints_needed} sprints")

    # Simulate sprint-by-sprint
    remaining_pts = total_points
    for s in range(1, sprints_needed + 1):
        done_this_sprint = min(velocity, remaining_pts)
        remaining_pts -= done_this_sprint
        print(f"      Sprint {s}: {done_this_sprint} points done, "
              f"{remaining_pts} remaining")

    # (c) Reduced velocity scenario
    reduced_velocity = 22
    sprints_reduced = math.ceil(total_points / reduced_velocity)
    print(f"\n  (c) If velocity drops to {reduced_velocity} points/sprint:")
    print(f"      {total_points} / {reduced_velocity} = "
          f"{total_points / reduced_velocity:.2f} -> {sprints_reduced} sprints")
    print(f"      Change: {sprints_needed} -> {sprints_reduced} sprints "
          f"(+{sprints_reduced - sprints_needed} sprints, "
          f"+{(sprints_reduced - sprints_needed) * 2} weeks)")


# === Exercise 4: Estimation Pitfalls ===
# Problem: Identify pitfalls for 4 scenarios.

def exercise_4():
    """Identify estimation biases and pitfalls."""

    print("EXERCISE 4: Estimation Pitfalls")
    print("=" * 60)

    scenarios = [
        {
            "description": (
                "(a) Developer estimates 'about a week'; manager says "
                "'Great, so five days.'"
            ),
            "pitfall": "PRECISION BIAS + ANCHORING",
            "analysis": (
                "The developer's 'about a week' is a rough estimate with "
                "uncertainty (could mean 4-8 days). The manager reinterprets it "
                "as a precise commitment of exactly 5 days, losing the "
                "uncertainty range. The manager also anchors on the low end."
            ),
            "mitigation": (
                "Always give estimates as ranges: 'I think 5-8 days.' Use "
                "three-point (PERT) estimates. Document that '1 week' means "
                "'5 working days +/- 2 days' and get the manager to acknowledge "
                "the range."
            ),
        },
        {
            "description": (
                "(b) Team spends first 8 days on design, rushes implementation "
                "in last 2 days of a 10-day sprint."
            ),
            "pitfall": "STUDENT SYNDROME + PARKINSON'S LAW",
            "analysis": (
                "Design expanded to fill available time (Parkinson). Implementation "
                "was deferred until deadline pressure forced action (Student Syndrome). "
                "The result is rushed, low-quality code."
            ),
            "mitigation": (
                "Time-box design to a fixed percentage of the sprint (e.g., 20-30%). "
                "Define intermediate milestones: 'Design complete by day 3, first PR "
                "by day 6, testing by day 8.' Track progress daily in standup."
            ),
        },
        {
            "description": (
                "(c) A 5-point story consistently takes 3x longer than "
                "comparable 5-point stories from other teams."
            ),
            "pitfall": "RELATIVE ESTIMATION CALIBRATION FAILURE",
            "analysis": (
                "Story points are team-specific. This team's '5 points' represents "
                "more actual work than other teams' '5 points.' The team may have "
                "hidden complexity (environment issues, tech debt, process overhead) "
                "not reflected in their estimates."
            ),
            "mitigation": (
                "Do NOT compare story points across teams — they are a relative "
                "measure within a team. Instead, investigate why this team's "
                "throughput is lower: is it tech debt, unclear requirements, "
                "environment problems, or lack of skills? Focus on removing "
                "impediments rather than recalibrating estimates."
            ),
        },
        {
            "description": (
                "(d) Project 'almost done' for six weeks; same 20% remains."
            ),
            "pitfall": "90% SYNDROME",
            "analysis": (
                "The last 20% is the hardest: integration, edge cases, "
                "performance tuning, bug fixing. Progress feels fast at the "
                "beginning (easy parts) and slows dramatically at the end "
                "(hard parts). The team conflates code written with work done."
            ),
            "mitigation": (
                "Measure 'done' by running, tested features — not code completion. "
                "Use Definition of Done that includes testing and integration. "
                "Decompose the remaining 20% into specific tasks with hour "
                "estimates and track them on a burn-down chart."
            ),
        },
    ]

    for sc in scenarios:
        print(f"\n  {sc['description']}")
        print(f"    Pitfall: {sc['pitfall']}")
        print(f"    Analysis: {sc['analysis']}")
        print(f"    Mitigation: {sc['mitigation']}")


# === Exercise 5: WBS and Critical Path ===
# Problem: Data migration project — WBS, forward/backward pass, delay analysis.

def exercise_5():
    """WBS and Critical Path for a data migration project."""

    print("EXERCISE 5: WBS and Critical Path — Data Migration Project")
    print("=" * 60)

    # WBS
    print("""
  WORK BREAKDOWN STRUCTURE (3 levels):

  1. Data Migration Project
     1.1 Analysis & Design
         1.1.1 Requirements analysis (3 days)
         1.1.2 Source schema mapping (4 days)
         1.1.3 Target schema design (5 days)
     1.2 Development
         1.2.1 ETL script development (8 days)
         1.2.2 Data validation rules (3 days)
     1.3 Testing & Migration
         1.3.1 Testing environment setup (2 days)
         1.3.2 Migration dry run (4 days)
         1.3.3 Validation testing (3 days)
         1.3.4 Production migration (1 day)
""")

    # Define tasks
    tasks = {
        "ReqAnalysis":      {"duration": 3, "predecessors": [],                              "name": "Requirements analysis"},
        "SrcSchemaMap":      {"duration": 4, "predecessors": ["ReqAnalysis"],                 "name": "Source schema mapping"},
        "TgtSchemaDesign":   {"duration": 5, "predecessors": ["ReqAnalysis"],                 "name": "Target schema design"},
        "ETLDev":            {"duration": 8, "predecessors": ["SrcSchemaMap", "TgtSchemaDesign"], "name": "ETL script development"},
        "DataValRules":      {"duration": 3, "predecessors": ["TgtSchemaDesign"],             "name": "Data validation rules"},
        "TestEnvSetup":      {"duration": 2, "predecessors": [],                              "name": "Testing env setup"},
        "DryRun":            {"duration": 4, "predecessors": ["ETLDev", "TestEnvSetup"],      "name": "Migration dry run"},
        "ValTesting":        {"duration": 3, "predecessors": ["DryRun", "DataValRules"],      "name": "Validation testing"},
        "ProdMigration":     {"duration": 1, "predecessors": ["ValTesting"],                  "name": "Production migration"},
    }

    # Topological order
    order = ["ReqAnalysis", "SrcSchemaMap", "TgtSchemaDesign", "ETLDev",
             "DataValRules", "TestEnvSetup", "DryRun", "ValTesting", "ProdMigration"]

    # Forward pass
    es = {}
    ef = {}
    for task_id in order:
        t = tasks[task_id]
        if not t["predecessors"]:
            es[task_id] = 0
        else:
            es[task_id] = max(ef[p] for p in t["predecessors"])
        ef[task_id] = es[task_id] + t["duration"]

    project_duration = ef["ProdMigration"]

    # Backward pass
    successors = {t: [] for t in tasks}
    for task_id, t in tasks.items():
        for p in t["predecessors"]:
            successors[p].append(task_id)

    lf = {}
    ls = {}
    for task_id in reversed(order):
        if not successors[task_id]:
            lf[task_id] = project_duration
        else:
            lf[task_id] = min(ls[s] for s in successors[task_id])
        ls[task_id] = lf[task_id] - tasks[task_id]["duration"]

    float_vals = {t: ls[t] - es[t] for t in tasks}

    # (a) Print results
    print("  (a) Forward and Backward Pass Results:")
    print(f"  {'Task':<25} {'Dur':>4} {'ES':>4} {'EF':>4} {'LS':>4} {'LF':>4} {'Float':>6} {'Crit?'}")
    print(f"  {'-' * 70}")
    for t in order:
        is_crit = " *" if float_vals[t] == 0 else ""
        print(f"  {tasks[t]['name']:<25} {tasks[t]['duration']:>4} "
              f"{es[t]:>4} {ef[t]:>4} {ls[t]:>4} {lf[t]:>4} {float_vals[t]:>6}{is_crit}")

    # (b) Critical path
    critical = [tasks[t]["name"] for t in order if float_vals[t] == 0]
    print(f"\n  (b) Critical Path: {' -> '.join(critical)}")
    print(f"      Project Duration: {project_duration} days")

    # (c) ETL delayed by 3 days
    etl_delay = 3
    new_etl_ef = ef["ETLDev"] + etl_delay
    new_dry_run_es = max(new_etl_ef, ef["TestEnvSetup"])
    new_dry_run_ef = new_dry_run_es + tasks["DryRun"]["duration"]
    new_val_es = max(new_dry_run_ef, ef["DataValRules"])
    new_val_ef = new_val_es + tasks["ValTesting"]["duration"]
    new_prod_es = new_val_ef
    new_prod_ef = new_prod_es + tasks["ProdMigration"]["duration"]

    print(f"\n  (c) If ETL development is delayed by {etl_delay} days:")
    print(f"      ETL now finishes at day {new_etl_ef} (was {ef['ETLDev']})")
    print(f"      Dry Run starts day {new_dry_run_es}, ends day {new_dry_run_ef}")
    print(f"      Validation testing starts day {new_val_es}, ends day {new_val_ef}")
    print(f"      Production migration ends day {new_prod_ef}")
    print(f"      New project duration: {new_prod_ef} days (was {project_duration})")
    print(f"      Impact: +{new_prod_ef - project_duration} days")
    print(f"      ETL is on the critical path, so the full delay propagates.")


if __name__ == "__main__":
    print("=" * 65)
    print("=== PRACTICE EXERCISES (Section 15) ===")
    print("=" * 65)

    for i, func in enumerate([practice_exercise_1, practice_exercise_2,
                               practice_exercise_3, practice_exercise_4,
                               practice_exercise_5], 1):
        print(f"\n{'=' * 65}")
        print(f"=== Practice Exercise {i} ===")
        print("=" * 65)
        func()

    print("\n\n" + "=" * 65)
    print("=== EXERCISES (End of Lesson) ===")
    print("=" * 65)

    for i, func in enumerate([exercise_1, exercise_2, exercise_3,
                               exercise_4, exercise_5], 1):
        print(f"\n{'=' * 65}")
        print(f"=== Exercise {i} ===")
        print("=" * 65)
        func()

    print("\nAll exercises completed!")
