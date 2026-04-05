"""
Exercises for Lesson 02: Software Development Life Cycle
Topic: Software_Engineering

Solutions to practice problems from the lesson.
"""


# === Exercise 1: SDLC Model Recommendation ===
# Problem: A city government is building a traffic management system (800 intersections,
# fixed budget, safety certification required, 3 years). Recommend an SDLC model.

def exercise_1():
    """Recommend an SDLC model for a traffic management system."""

    recommendation = """
RECOMMENDED MODEL: V-Model

JUSTIFICATION (addressing three decision criteria from Section 9):

1. REGULATORY REQUIREMENTS (Safety Certification):
   Traffic management systems are safety-critical -- malfunctions can cause
   accidents and loss of life. The V-Model explicitly maps each development
   phase to a corresponding testing phase, producing the traceability matrix
   required for safety certification. Each requirement is linked to a specific
   test case, satisfying regulatory audit requirements.

2. REQUIREMENTS STABILITY:
   Traffic signal control rules are defined by traffic engineering standards
   and municipal regulations. These requirements are well-understood and
   unlikely to change significantly during the 3-year project. The V-Model's
   upfront requirements definition is appropriate when requirements are stable.

3. DOCUMENTATION NEEDS:
   A city government project with a fixed budget requires comprehensive
   documentation for accountability, regulatory compliance, and long-term
   maintenance by potentially different contractors. The V-Model produces
   extensive documentation at each phase, which is a strength in this context.

WHY NOT WATERFALL?
   Waterfall is also viable, but the V-Model's explicit testing traceability
   provides additional safety assurance that pure Waterfall lacks.

WHY NOT AGILE?
   While iterative delivery has benefits, the safety-critical nature and
   regulatory requirements demand formal verification documentation that
   Agile alone does not naturally produce. A hybrid (V-Model structure with
   iterative development within phases) could work but adds complexity.
"""
    print(recommendation)


# === Exercise 2: Waterfall Timeline for Expense App ===
# Problem: Draw a Waterfall timeline for a mobile expense report app
# integrated with SAP ERP. Estimate phase durations and key artifacts.

def exercise_2():
    """Waterfall timeline for a mobile expense report app."""

    timeline = {
        "Requirements": {
            "duration_pct": 15,
            "artifacts": [
                "Software Requirements Specification (SRS)",
                "Use case diagrams for expense submission workflow",
            ],
        },
        "Design": {
            "duration_pct": 20,
            "artifacts": [
                "Architecture document (mobile app + SAP integration layer)",
                "Database schema for offline expense storage",
            ],
        },
        "Implementation": {
            "duration_pct": 30,
            "artifacts": [
                "Source code (mobile app + backend API)",
                "SAP RFC/BAPI integration module",
            ],
        },
        "Testing": {
            "duration_pct": 25,
            "artifacts": [
                "Test plan and test case repository",
                "SAP integration test results and defect reports",
            ],
        },
        "Deployment": {
            "duration_pct": 10,
            "artifacts": [
                "Deployment guide and rollback procedure",
                "User training materials and app store listing",
            ],
        },
    }

    print("WATERFALL TIMELINE: Mobile Expense Report App")
    print("=" * 65)
    total_months = 8  # Assume 8-month project
    print(f"Total project duration: ~{total_months} months\n")

    bar_width = 40
    for phase, info in timeline.items():
        pct = info["duration_pct"]
        months = round(total_months * pct / 100, 1)
        bar = "#" * int(bar_width * pct / 100)
        print(f"  {phase:<16} [{bar:<{bar_width}}] {pct}% ({months} months)")
        for artifact in info["artifacts"]:
            print(f"    - {artifact}")
        print()


# === Exercise 3: Why Waterfall Fails for a Social Network Startup ===
# Problem: Explain why Waterfall is a poor choice, then design a 3-sprint
# incremental plan for the first 6 weeks.

def exercise_3():
    """Social network startup: why not Waterfall + 3-sprint plan."""

    print("WHY WATERFALL IS A POOR CHOICE FOR A SOCIAL NETWORK STARTUP:")
    print("-" * 60)
    reasons = [
        "1. Requirements are unclear: The startup has 'only a rough idea' of features. "
        "   Waterfall demands stable, upfront requirements.",
        "2. Market feedback is critical: A social network must iterate based on user "
        "   behavior. Waterfall delays all feedback until deployment.",
        "3. Risk of building the wrong product: By the time Waterfall delivers (12-18 "
        "   months), the market may have shifted. Early feedback is essential.",
        "4. Competitive pressure: Faster competitors using iterative approaches will "
        "   capture the market while Waterfall is still in the design phase.",
    ]
    for r in reasons:
        print(f"  {r}")

    print("\n\nTHREE-SPRINT INCREMENTAL PLAN (6 weeks, 2-week sprints):")
    print("=" * 60)

    sprints = [
        {
            "name": "Sprint 1: Core Identity (Weeks 1-2)",
            "features": [
                "User registration and login (email + password)",
                "Basic user profile (name, photo, bio)",
                "Simple news feed showing recent posts from all users",
            ],
            "rationale": "Establishes the core user identity loop. Without accounts "
                         "and a feed, nothing else makes sense.",
        },
        {
            "name": "Sprint 2: Social Connections (Weeks 3-4)",
            "features": [
                "Follow/unfollow other users",
                "Feed filtered to show only followed users' posts",
                "Create text posts with optional image",
            ],
            "rationale": "Introduces the social graph -- the core value proposition "
                         "of any social network. Now users have a reason to return.",
        },
        {
            "name": "Sprint 3: Engagement (Weeks 5-6)",
            "features": [
                "Like and comment on posts",
                "Notification system (new follower, likes, comments)",
                "User search/discovery feature",
            ],
            "rationale": "Adds engagement loops that increase retention. Notifications "
                         "bring users back; search helps grow the network.",
        },
    ]

    for sprint in sprints:
        print(f"\n  {sprint['name']}")
        for feature in sprint["features"]:
            print(f"    - {feature}")
        print(f"    Rationale: {sprint['rationale']}")


# === Exercise 4: V-Model vs Waterfall Comparison ===
# Problem: Create a table listing five specific differences.

def exercise_4():
    """Compare V-Model and Waterfall with five differences."""

    differences = [
        ("Testing approach",
         "Testing occurs once after implementation",
         "Testing is planned at each development phase"),
        ("Traceability",
         "Implicit: requirements may not link to specific tests",
         "Explicit: each requirement maps to a test level"),
        ("Defect detection timing",
         "Defects found late in integration/system testing",
         "Defects caught earlier due to parallel test planning"),
        ("Documentation volume",
         "Heavy but focused on development phases",
         "Heavier: includes test plans for each corresponding phase"),
        ("Suitability for safety-critical",
         "Acceptable but lacks explicit verification mapping",
         "Preferred: explicit V&V mapping satisfies regulatory requirements"),
    ]

    print("V-MODEL vs WATERFALL: Five Differences")
    print("=" * 80)
    print(f"{'Dimension':<28} {'Waterfall':<25} {'V-Model':<25}")
    print("-" * 80)
    for dim, wf, vm in differences:
        # Wrap long strings
        print(f"\n{dim}")
        print(f"  Waterfall: {wf}")
        print(f"  V-Model:   {vm}")

    print("\n\nWHEN TO CHOOSE V-MODEL OVER WATERFALL:")
    print("  Choose V-Model when the cost of post-deployment defects is very high")
    print("  (safety-critical, medical, aerospace) and regulatory compliance requires")
    print("  formal traceability from requirements to tests. The additional documentation")
    print("  overhead is justified by the assurance it provides.")


# === Exercise 5: Spiral Model Risk Resolution ===
# Problem: Real-time fraud detection for a bank. Identify 3 risks and prototypes.

def exercise_5():
    """Spiral model risks and prototypes for fraud detection system."""

    risks = [
        {
            "risk": "Latency requirement may be unachievable with chosen ML model",
            "description": (
                "Real-time fraud detection must score transactions within 50ms. "
                "Complex ML models (gradient boosting, neural networks) may exceed "
                "this threshold at production transaction volumes."
            ),
            "prototype": (
                "PERFORMANCE SPIKE: Build a minimal inference pipeline with synthetic "
                "transaction data. Test three model architectures (logistic regression, "
                "gradient boosting, small neural net) at 10,000 TPS. Measure p99 latency. "
                "If none meet 50ms, evaluate model optimization (ONNX runtime, quantization) "
                "or architectural changes (async scoring with fallback rules)."
            ),
        },
        {
            "risk": "Integration with legacy core banking system may be fragile",
            "description": (
                "The bank's mainframe transaction processing system uses a 30-year-old "
                "message format. Real-time interception of transactions for scoring "
                "requires hooking into this pipeline without affecting throughput."
            ),
            "prototype": (
                "INTEGRATION POC: Build a lightweight message interceptor that taps into "
                "the mainframe transaction queue (e.g., IBM MQ). Process a shadow copy of "
                "live transactions without affecting the production path. Verify message "
                "format parsing, error handling for malformed messages, and throughput "
                "under load. Run for 2 weeks to catch edge cases."
            ),
        },
        {
            "risk": "False positive rate too high, causing customer friction",
            "description": (
                "A fraud model that blocks too many legitimate transactions will anger "
                "customers and cost the bank more in customer churn than fraud losses."
            ),
            "prototype": (
                "DATA ANALYSIS + A/B FRAMEWORK: Using 6 months of historical transaction "
                "data with known fraud labels, train and evaluate models at different "
                "thresholds. Build a decision framework that visualizes the trade-off "
                "between false positive rate and fraud detection rate. Present to "
                "stakeholders to agree on acceptable thresholds before full development."
            ),
        },
    ]

    print("SPIRAL MODEL: Fraud Detection System — Risk Resolution")
    print("=" * 65)
    for i, r in enumerate(risks, 1):
        print(f"\n--- RISK {i}: {r['risk']} ---")
        print(f"  Description: {r['description']}")
        print(f"\n  Early Spiral Prototype:")
        print(f"  {r['prototype']}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: SDLC Model Recommendation ===")
    print("=" * 60)
    exercise_1()
    print("\n" + "=" * 60)
    print("=== Exercise 2: Waterfall Timeline ===")
    print("=" * 60)
    exercise_2()
    print("\n" + "=" * 60)
    print("=== Exercise 3: Incremental Plan ===")
    print("=" * 60)
    exercise_3()
    print("\n" + "=" * 60)
    print("=== Exercise 4: V-Model vs Waterfall ===")
    print("=" * 60)
    exercise_4()
    print("\n" + "=" * 60)
    print("=== Exercise 5: Spiral Risk Resolution ===")
    print("=" * 60)
    exercise_5()
    print("\nAll exercises completed!")
