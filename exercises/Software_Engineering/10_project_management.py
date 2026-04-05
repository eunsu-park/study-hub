"""
Exercises for Lesson 10: Project Management
Topic: Software_Engineering

Solutions to practice problems from the lesson.
Covers iron triangle, risk management, EVM calculations, WBS, stakeholder analysis.
"""

import math


# === Exercise 1: Iron Triangle Trade-offs ===

def exercise_1():
    """Iron Triangle memo to a client with scope/schedule/budget conflict."""

    print("EXERCISE 1: Iron Triangle Trade-offs")
    print("=" * 65)

    print("""
  MEMO TO CLIENT
  ==============

  Subject: Project Feasibility Analysis — Scope, Schedule, and Budget Options
  From: Project Manager
  To: Client Stakeholder
  Date: 2025-02-20

  (a) THE TRIPLE CONSTRAINT:

  Every software project is constrained by three interdependent variables:
  Scope (what we build), Schedule (when we deliver), and Cost (what we spend).
  These form the "Iron Triangle" — changing one side forces changes in the
  others. It is impossible to simultaneously maximize scope, minimize schedule,
  AND minimize cost.

  Your current request:
    - Full scope delivery
    - 6-month deadline
    - $200,000 fixed budget

  Our estimate:
    - Full scope requires 9 months at $280,000.

  The gap: 3 months and $80,000. We must adjust at least one constraint.

  (b) THREE OPTIONS:

  OPTION A: Reduce Scope (Fixed Schedule + Budget)
    Deliver in 6 months at $200,000 by cutting ~30% of features.
    We prioritize the MVP: core user journeys that deliver the most
    business value. Deferred features are built in a Phase 2.
    Risk: Phase 2 may never be funded; users get a partial product.
    Benefit: On time, on budget. Early user feedback improves Phase 2.

  OPTION B: Extend Schedule (Fixed Scope + Budget)
    Deliver full scope in 9 months at $280,000.
    The team works at sustainable pace, delivering all features.
    Risk: Market window may close; stakeholders wait 3 months longer.
    Benefit: Complete product, no technical debt from rushing.

  OPTION C: Increase Budget (Fixed Scope + Schedule)
    Deliver full scope in 6 months by adding 2 developers ($280,000).
    Accelerate delivery through parallel workstreams.
    Risk: Brooks's Law — adding people to a project increases
    communication overhead. Actual delivery may be 7 months, not 6.
    Benefit: Full scope delivered close to the original timeline.

  (c) RECOMMENDATION:

  I recommend OPTION A: Reduce Scope.

  Justification:
  1. The 6-month deadline is driven by a market event (trade show in July).
     Missing it eliminates the primary business reason for the project.
  2. $200,000 is a hard budget constraint (board-approved).
  3. A focused MVP in 6 months, followed by iterative enhancements, is
     lower risk than rushing full scope with more people.
  4. We will use MoSCoW prioritization to ensure the most valuable 70%
     of features are in the MVP. I will present the prioritized feature
     list for your approval next week.
""")


# === Exercise 2: Risk Register ===

def exercise_2():
    """Risk register for a ride-sharing app with 4-month deadline."""

    print("EXERCISE 2: Risk Register — Ride-Sharing App")
    print("=" * 65)

    risks = [
        # (Category, Risk, Probability 1-3, Impact 1-5, Score, Strategy, Action)
        ("Technical", "Payment gateway integration fails or has latency issues",
         3, 5, 15, "Mitigate",
         "Build payment abstraction layer. Test with sandbox early in Sprint 1. "
         "Have a backup gateway (Stripe + Braintree)."),
        ("Technical", "GPS accuracy is poor in urban canyons, causing wrong pickups",
         2, 4, 8, "Mitigate",
         "Use fused location providers (GPS + WiFi + cell). Allow manual pin adjustment. "
         "Test in 5 real-world urban locations."),
        ("People", "Lead mobile developer leaves during the project",
         1, 5, 5, "Mitigate",
         "Cross-train a second developer on the mobile codebase. Ensure all "
         "architecture decisions are documented in ADRs."),
        ("People", "Team burnout from aggressive 4-month deadline",
         2, 4, 8, "Mitigate",
         "Cap overtime at 5 hours/week. Protect weekends. If behind schedule, "
         "reduce scope rather than increase hours."),
        ("External", "Regulatory approval for ride-sharing delayed in target city",
         2, 5, 10, "Avoid",
         "Start regulatory application in month 1 (parallel to development). "
         "Have legal counsel on retainer. Choose a backup launch city."),
        ("External", "Map API pricing increases mid-project (e.g., Google Maps)",
         1, 3, 3, "Accept",
         "Lock in current pricing tier. Budget a 20% contingency for API costs. "
         "Evaluate open-source alternatives (OpenStreetMap) as fallback."),
        ("Requirements", "Stakeholders add 'just one more feature' mid-sprint",
         3, 3, 9, "Mitigate",
         "Enforce change control: all new features go to the backlog, not the "
         "current sprint. Product Owner is the sole gatekeeper."),
        ("Requirements", "Driver onboarding flow requirements are unclear",
         2, 4, 8, "Mitigate",
         "Schedule a dedicated requirements workshop in Week 1. Prototype "
         "the driver signup flow and validate with 5 actual drivers."),
    ]

    print(f"\n  {'#':>2} {'Category':<14} {'Risk':<45} {'P':>2} {'I':>2} {'Score':>5} "
          f"{'Strategy':<10}")
    print(f"  {'-' * 90}")

    sorted_risks = sorted(risks, key=lambda r: r[4], reverse=True)
    for i, (cat, risk, prob, impact, score, strategy, action) in enumerate(sorted_risks, 1):
        print(f"\n  {i:>2} {cat:<14} {risk[:45]:<45} {prob:>2} {impact:>2} {score:>5} "
              f"{strategy:<10}")
        print(f"     Action: {action}")

    print(f"\n  RISK HEAT MAP:")
    print(f"  Score >= 10: RED (immediate action required)")
    print(f"  Score 5-9:   YELLOW (monitor closely, mitigate)")
    print(f"  Score < 5:   GREEN (accept and monitor)")


# === Exercise 3: Earned Value Management ===

def exercise_3():
    """EVM calculations and interpretation."""

    print("EXERCISE 3: Earned Value Management")
    print("=" * 65)

    bac = 200_000  # Budget at Completion
    planned_months = 8
    current_month = 4
    planned_pct = 50  # Should be 50% done at month 4
    actual_pct = 42  # Actually 42% done
    actual_cost = 95_000  # Spent $95,000

    pv = bac * (planned_pct / 100)  # Planned Value
    ev = bac * (actual_pct / 100)   # Earned Value
    ac = actual_cost                 # Actual Cost

    sv = ev - pv  # Schedule Variance
    cv = ev - ac  # Cost Variance
    spi = ev / pv  # Schedule Performance Index
    cpi = ev / ac  # Cost Performance Index
    eac = bac / cpi  # Estimate at Completion

    print(f"\n  Given:")
    print(f"    BAC (Budget at Completion) = ${bac:,}")
    print(f"    Planned duration = {planned_months} months")
    print(f"    Current month = {current_month}")
    print(f"    Planned completion = {planned_pct}%")
    print(f"    Actual completion = {actual_pct}%")
    print(f"    Actual Cost (AC) = ${ac:,}")

    print(f"\n  Calculations:")
    print(f"    PV (Planned Value)  = BAC × {planned_pct}% = ${pv:,.0f}")
    print(f"    EV (Earned Value)   = BAC × {actual_pct}% = ${ev:,.0f}")
    print(f"    AC (Actual Cost)    = ${ac:,}")

    print(f"\n    SV (Schedule Variance) = EV - PV = ${ev:,.0f} - ${pv:,.0f} = ${sv:,.0f}")
    print(f"    CV (Cost Variance)     = EV - AC = ${ev:,.0f} - ${ac:,} = ${cv:,.0f}")
    print(f"    SPI (Schedule Perf)    = EV / PV = {ev:,.0f} / {pv:,.0f} = {spi:.2f}")
    print(f"    CPI (Cost Perf)        = EV / AC = {ev:,.0f} / {ac:,} = {cpi:.2f}")
    print(f"    EAC (Est. at Compl.)   = BAC / CPI = {bac:,} / {cpi:.2f} = ${eac:,.0f}")

    print(f"\n  Interpretation:")
    print(f"    SV = ${sv:,.0f} -> BEHIND SCHEDULE by ${abs(sv):,.0f} worth of work")
    print(f"    CV = ${cv:,.0f} -> OVER BUDGET by ${abs(cv):,.0f}")
    print(f"    SPI = {spi:.2f} -> Getting only {spi*100:.0f}% of planned work done per period")
    print(f"    CPI = {cpi:.2f} -> Getting only ${cpi*100:.0f} of value for every $100 spent")
    print(f"    EAC = ${eac:,.0f} -> At current efficiency, project will cost ${eac:,.0f}")
    print(f"         This is ${eac - bac:,.0f} OVER the original budget.")

    print(f"\n  VERDICT: This project is NOT in good shape.")
    print(f"    - Behind schedule (SPI < 1.0) AND over budget (CPI < 1.0)")
    print(f"    - If the trend continues, it will cost ${eac:,.0f} instead of ${bac:,}")
    print(f"    - Corrective actions needed: reduce scope, add resources, or extend schedule")


# === Exercise 4: Work Breakdown Structure ===

def exercise_4():
    """WBS for a personal finance tracking web application."""

    print("EXERCISE 4: WBS — Personal Finance Tracking App")
    print("=" * 65)

    wbs = {
        "1. Project Management": {
            "1.1 Planning": [
                ("1.1.1 Define project scope and objectives", 8),
                ("1.1.2 Create project schedule", 4),
                ("1.1.3 Risk assessment", 4),
            ],
            "1.2 Monitoring": [
                ("1.2.1 Weekly status meetings (8 weeks)", 16),
                ("1.2.2 Sprint planning and retrospectives", 16),
            ],
        },
        "2. Requirements & Design": {
            "2.1 Requirements": [
                ("2.1.1 User story creation and prioritization", 12),
                ("2.1.2 UI/UX wireframing", 16),
                ("2.1.3 Data model design", 8),
            ],
            "2.2 Architecture": [
                ("2.2.1 Technology stack selection", 4),
                ("2.2.2 API design (OpenAPI spec)", 8),
                ("2.2.3 Database schema design", 8),
            ],
        },
        "3. Backend Development": {
            "3.1 Authentication": [
                ("3.1.1 User registration and login", 12),
                ("3.1.2 Password reset flow", 6),
            ],
            "3.2 Core Features": [
                ("3.2.1 Expense logging API", 12),
                ("3.2.2 Category management API", 8),
                ("3.2.3 Monthly summary calculation", 10),
                ("3.2.4 Budget alerts engine", 8),
            ],
            "3.3 Data Layer": [
                ("3.3.1 Database setup and migrations", 6),
                ("3.3.2 Data validation layer", 6),
            ],
        },
        "4. Frontend Development": {
            "4.1 Core UI": [
                ("4.1.1 Dashboard page", 16),
                ("4.1.2 Expense entry form", 10),
                ("4.1.3 Category management screen", 8),
            ],
            "4.2 Reporting UI": [
                ("4.2.1 Monthly summary chart", 12),
                ("4.2.2 Spending breakdown by category", 10),
                ("4.2.3 Export to CSV feature", 6),
            ],
        },
        "5. Testing & Deployment": {
            "5.1 Testing": [
                ("5.1.1 Unit test suite", 16),
                ("5.1.2 Integration testing", 12),
                ("5.1.3 UAT with test users", 8),
            ],
            "5.2 Deployment": [
                ("5.2.1 CI/CD pipeline setup", 8),
                ("5.2.2 Production deployment", 4),
                ("5.2.3 Monitoring and alerting setup", 6),
            ],
        },
    }

    total_hours = 0
    for phase, categories in wbs.items():
        print(f"\n  {phase}")
        for category, packages in categories.items():
            print(f"    {category}")
            for pkg, hours in packages:
                print(f"      {pkg} ({hours}h)")
                total_hours += hours

    print(f"\n  {'=' * 50}")
    print(f"  Total estimated effort: {total_hours} hours")
    print(f"  At 40 hrs/week with 2 developers: ~{total_hours / 80:.0f} weeks")


# === Exercise 5: Stakeholder Analysis ===

def exercise_5():
    """Stakeholder analysis for hospital appointment scheduling system."""

    print("EXERCISE 5: Stakeholder Analysis — Hospital Scheduling System")
    print("=" * 65)

    stakeholders = [
        ("Hospital CIO", "H", "H", "Manage Closely",
         "Executive sponsor. Controls budget and strategic direction."),
        ("Chief Medical Officer", "H", "M", "Keep Satisfied",
         "Clinical authority. Must approve workflow changes affecting patient care."),
        ("IT Department Manager", "M", "H", "Keep Informed",
         "Responsible for deployment, integration, and ongoing support."),
        ("Receptionist (Front Desk)", "L", "H", "Keep Informed",
         "Primary daily user. Their efficiency depends on the system."),
        ("Doctors", "H", "M", "Keep Satisfied",
         "Schedule consumers. High power to reject system if it disrupts workflow."),
        ("Nurses", "L", "M", "Monitor",
         "Secondary users. Need visibility into appointment schedules."),
        ("Patients", "L", "H", "Keep Informed",
         "End users of the patient portal. Cannot be ignored despite low power."),
        ("Insurance Companies", "M", "L", "Monitor",
         "Require specific appointment data formats for claims processing."),
        ("Compliance Officer", "H", "H", "Manage Closely",
         "Must ensure HIPAA compliance. Can block deployment."),
        ("Third-party EMR Vendor", "M", "M", "Keep Informed",
         "Provides the electronic medical records system we must integrate with."),
    ]

    print(f"\n  {'Stakeholder':<25} {'Power':>5} {'Interest':>8} {'Strategy':<18} ")
    print(f"  {'-' * 75}")
    for name, power, interest, strategy, notes in stakeholders:
        print(f"  {name:<25} {power:>5} {interest:>8} {strategy:<18}")
        print(f"    Note: {notes}")

    print("""
  HIGHEST COMMUNICATION INVESTMENT:

  1. Hospital CIO (High Power, High Interest):
     Rationale: As executive sponsor, the CIO controls budget approval,
     resource allocation, and can escalate or cancel the project. Their high
     interest means they want frequent updates and involvement in key
     decisions. Strategy: Weekly 1-on-1 status meetings, dashboard access,
     early involvement in milestone reviews. Surprises are unacceptable.

  2. Compliance Officer (High Power, High Interest):
     Rationale: HIPAA violations carry fines up to $1.5M per incident. The
     Compliance Officer can block go-live if they find violations. Their high
     interest means they are actively monitoring. Strategy: Involve them from
     Day 1 in requirements, include a HIPAA compliance checklist in every
     sprint review, and get written sign-off before each deployment.
""")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Iron Triangle Trade-offs", exercise_1),
        ("Exercise 2: Risk Register", exercise_2),
        ("Exercise 3: Earned Value Management", exercise_3),
        ("Exercise 4: Work Breakdown Structure", exercise_4),
        ("Exercise 5: Stakeholder Analysis", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 65}")
        print(f"=== {title} ===")
        print("=" * 65)
        func()

    print("\nAll exercises completed!")
