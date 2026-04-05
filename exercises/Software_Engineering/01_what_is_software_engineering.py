"""
Exercises for Lesson 01: What Is Software Engineering
Topic: Software_Engineering

Solutions to practice problems from the lesson.
These exercises are primarily analytical/essay-style. The solutions below
provide structured frameworks and key points for each exercise.
"""


# === Exercise 1: Compare Software Engineering with Civil Engineering ===
# Problem: Compare and contrast software engineering with civil engineering.
# Identify three similarities and three fundamental differences.
# Which of Brooks's four properties has no direct analog in civil engineering?

def exercise_1():
    """Compare software engineering and civil engineering."""

    similarities = [
        "1. Both require systematic planning, design, and project management before construction begins.",
        "2. Both must satisfy stakeholder requirements within constraints of time, budget, and quality.",
        "3. Both rely on testing and verification to ensure the final product meets specifications.",
    ]

    differences = [
        "1. Invisibility: Software has no physical form; civil structures are visible and inspectable.",
        "2. Changeability: Software is expected to change constantly after delivery; a bridge is not "
        "   routinely redesigned after construction.",
        "3. Replication cost: Copying software costs nearly nothing; replicating a bridge costs as "
        "   much as building the first one.",
    ]

    no_analog = (
        "INVISIBILITY has no direct analog in civil engineering. A civil engineer can see "
        "the structure of a bridge, measure its tolerances, and physically inspect stress points. "
        "Software is invisible -- there is no physical artifact to observe. Diagrams and "
        "documentation are imperfect proxies. This makes it fundamentally harder to visualize "
        "and communicate software architecture."
    )

    print("=== Similarities ===")
    for s in similarities:
        print(f"  {s}")
    print("\n=== Differences ===")
    for d in differences:
        print(f"  {d}")
    print(f"\n=== Brooks's property with no civil engineering analog ===")
    print(f"  {no_analog}")


# === Exercise 2: Principles for a Personal Script ===
# Problem: You are a sole developer maintaining a personal script that automates
# your email sorting. Identify which software engineering principles you should
# still apply, and which are less critical at this scale.

def exercise_2():
    """Evaluate SE principles for a personal automation script."""

    still_apply = {
        "Rigor (minimal)": (
            "Even personal scripts benefit from basic correctness checks. A regex that "
            "silently misclassifies emails could lose important messages."
        ),
        "Modularity": (
            "Separating email fetching, classification logic, and action execution makes "
            "debugging easier when the script inevitably breaks after a server change."
        ),
        "Anticipation of change": (
            "Email providers change APIs, label schemas evolve, and your sorting rules will "
            "change. Keeping configuration separate from code is worthwhile even for personal tools."
        ),
        "Version control": (
            "Even a one-person script benefits from Git: you can revert bad changes, understand "
            "what you changed and when, and maintain a history of your sorting rules."
        ),
    }

    less_critical = {
        "Separation of concerns (at scale)": (
            "A small script does not need multi-layer architecture with separate presentation, "
            "business logic, and data layers."
        ),
        "Formal documentation": (
            "A README with setup instructions is enough. A formal SRS is overkill."
        ),
        "Generality": (
            "Building a general-purpose email classification framework is over-engineering. "
            "Solve YOUR problem."
        ),
        "Incrementality": (
            "With one user and simple requirements, there is no need for staged releases "
            "or user feedback loops."
        ),
    }

    print("=== Principles that STILL APPLY ===")
    for principle, justification in still_apply.items():
        print(f"\n  [{principle}]")
        print(f"    {justification}")

    print("\n=== Principles that are LESS CRITICAL ===")
    for principle, justification in less_critical.items():
        print(f"\n  [{principle}]")
        print(f"    {justification}")


# === Exercise 3: Software Failure Analysis ===
# Problem: Analyze the Knight Capital incident (2012). What went wrong?
# Which SE principles were violated? What could have been done differently?

def exercise_3():
    """One-page analysis of the Knight Capital incident (2012)."""

    analysis = """
KNIGHT CAPITAL GROUP INCIDENT — August 1, 2012
================================================

WHAT HAPPENED:
- Knight Capital deployed new trading software (SMARS) to 8 production servers.
- An engineer reused an old feature flag that had been repurposed, activating
  dormant test code from a decade-old module (Power Peg functionality).
- 7 of 8 servers were updated correctly; 1 server still had old code.
- When markets opened, the old code executed millions of unintended trades
  in 45 minutes, accumulating $7 billion in unwanted positions.
- Knight Capital lost $440 million and nearly went bankrupt.

WHICH PRINCIPLES WERE VIOLATED:
1. Configuration Management: Reusing a flag name for a different purpose violated
   the principle of unique, traceable configuration items.
2. Rigor and Formality: The deployment lacked a checklist to verify all servers
   were in the same state before going live.
3. Anticipation of Change: Dead code was left in the codebase for a decade.
   It should have been removed when it was no longer needed.
4. Testing/Verification: No smoke test or canary deployment verified that
   production behavior matched expectations before full traffic was enabled.
5. Incrementality: All 8 servers were cut over simultaneously with no
   rollback capability or phased deployment.

WHAT COULD HAVE BEEN DONE:
- Remove dead code aggressively (preventive maintenance).
- Use unique, descriptive feature flag names; never reuse identifiers.
- Deploy to a single server first (canary deployment) and verify behavior.
- Implement automated deployment verification that compares software versions
  across all servers before enabling traffic.
- Have a kill switch to halt trading within seconds if anomalous volume is detected.
"""
    print(analysis)


# === Exercise 4: Research a Software Engineering Role ===
# Problem: Research the Site Reliability Engineer (SRE) role.

def exercise_4():
    """Describe the SRE role and its intersection with developers."""

    report = """
ROLE: Site Reliability Engineer (SRE)
======================================

DAY-TO-DAY RESPONSIBILITIES:
- Monitor production systems and respond to alerts and incidents.
- Define and maintain Service Level Objectives (SLOs) and error budgets.
- Automate operational tasks (toil reduction): deployment, scaling, failover.
- Conduct capacity planning to ensure systems can handle future load.
- Write and maintain infrastructure code (Terraform, Kubernetes manifests).
- Participate in blameless postmortems after incidents.
- Improve observability: logging, metrics, distributed tracing.
- Maintain on-call rotation, typically covering nights/weekends in shifts.

INTERSECTION WITH SOFTWARE DEVELOPERS:
- SREs review production-readiness of new services (scalability, monitoring).
- They collaborate on defining SLOs that balance feature velocity with reliability.
- When the error budget is exhausted, SREs can halt feature deployments until
  reliability improves -- creating a feedback loop that incentivizes quality.
- SREs often contribute code to the application itself (e.g., circuit breakers,
  graceful degradation logic).
- They maintain shared tools used by all developers (CI/CD pipelines, deployment
  tooling, observability platforms).

KEY DISTINCTION:
- A traditional ops engineer primarily manages infrastructure manually.
- An SRE applies software engineering principles TO operations, automating
  everything possible and treating operations as a software problem.
"""
    print(report)


# === Exercise 5: No Silver Bullet Revisited ===
# Problem: Review Brooks's "No Silver Bullet" claim in light of modern developments.

def exercise_5():
    """Assess Brooks's No Silver Bullet claim with modern evidence."""

    analysis = """
BROOKS'S 'NO SILVER BULLET' (1986) -- MODERN ASSESSMENT
=========================================================

BROOKS'S CLAIM: No single development will produce a 10x improvement in
software productivity, reliability, or simplicity within a decade.

ARGUMENTS FOR (the claim still holds):
1. AI Code Assistants (e.g., Copilot, Claude): These dramatically speed up
   TYPING code but do not address essential complexity -- the hard part is
   deciding WHAT to build, not typing it. Productivity gains are estimated
   at 20-55% for certain tasks, far from 10x overall.

2. Cloud Infrastructure (AWS, GCP, Azure): Reduces accidental complexity of
   provisioning servers but introduces NEW essential complexity (distributed
   systems, cost management, cloud-specific failure modes).

3. Mature Frameworks (Django, Rails, React): Provide enormous leverage for
   common patterns but cannot eliminate the essential complexity of unique
   business logic.

ARGUMENTS AGAINST (partial challenges):
1. Open-source ecosystems: A modern developer can assemble functionality
   from thousands of libraries that would have required years of custom
   development in 1986. This is arguably a >10x improvement in SOME domains.

2. Version control + CI/CD: The ability to test and deploy continuously
   has fundamentally changed the cost structure of finding and fixing bugs.

CONCLUSION: Brooks was largely correct. No single silver bullet exists.
However, the COMBINATION of many incremental improvements (cloud + CI/CD +
open-source + better languages + AI assistance) has collectively produced
productivity gains that exceed 10x over the 40-year period since the essay.
The key insight is that these are MANY tools working together, not one
silver bullet -- which is exactly what Brooks predicted.
"""
    print(analysis)


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: SE vs Civil Engineering ===")
    print("=" * 60)
    exercise_1()
    print("\n" + "=" * 60)
    print("=== Exercise 2: Principles for Personal Script ===")
    print("=" * 60)
    exercise_2()
    print("\n" + "=" * 60)
    print("=== Exercise 3: Knight Capital Failure Analysis ===")
    print("=" * 60)
    exercise_3()
    print("\n" + "=" * 60)
    print("=== Exercise 4: SRE Role Research ===")
    print("=" * 60)
    exercise_4()
    print("\n" + "=" * 60)
    print("=== Exercise 5: No Silver Bullet Revisited ===")
    print("=" * 60)
    exercise_5()
    print("\nAll exercises completed!")
