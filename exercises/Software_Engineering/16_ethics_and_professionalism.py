"""
Exercises for Lesson 16: Ethics and Professionalism
Topic: Software_Engineering

Solutions to practice problems from the lesson.
Covers ethical analysis, bias auditing, open source licensing, case studies, career planning.
"""


# === Exercise 1: Ethical Analysis — Airline Scheduling ===

def exercise_1():
    """Ethical analysis of an airline crew scheduling system."""

    print("EXERCISE 1: Ethical Analysis — Airline Crew Scheduling")
    print("=" * 65)

    print("""
  SCENARIO: Automated crew scheduling system for irregular operations.
  Management wants cost minimization. Safety analyst warns about fatigue.

  ACM/IEEE CODE OF ETHICS — RELEVANT PRINCIPLES:

  Principle 1.03 — PUBLIC:
  "Approve software only if they have a well-founded belief that it is safe,
  meets specifications, passes appropriate tests, and does not diminish
  quality of life..."

  Application: A scheduling system that assigns fatigued crews to flights
  directly threatens public safety. Crew fatigue is a known factor in
  aviation incidents (Colgan Air 3407, 2009). Approving a cost-only
  optimization that ignores fatigue violates the "safe" requirement.

  Principle 3.10 — PRODUCT:
  "Ensure adequate testing, debugging, and review of software."

  Application: The system must be tested specifically for fatigue-related
  scenarios. If testing only validates cost optimization but not safety
  constraints, the testing is inadequate.

  Principle 6.06 — PROFESSION:
  "Be accurate in stating the characteristics of software on which they work,
  avoiding not only false claims but also claims that might reasonably be
  supposed to be speculative, vacuous, deceptive, misleading, or doubtful."

  Application: If the system is marketed as "optimized scheduling" without
  disclosing that it does not account for fatigue, the engineers are
  misleading the airline and the public.

  OBLIGATIONS OF THE ENGINEERS:

  1. Raise the safety concern formally and in writing.
     Document the fatigue correlation data the safety analyst provided.
     Present it to management with a clear recommendation.

  2. Propose a constraint-based approach: optimize for cost SUBJECT TO
     safety constraints (minimum rest hours, maximum consecutive duty hours,
     fatigue risk score thresholds). This satisfies both business and safety.

  3. Refuse to ship the system without fatigue safeguards if management
     insists on cost-only optimization. The ACM Code is clear: public
     safety takes precedence over employer directives.

  IF MANAGEMENT OVERRIDES SAFETY CONCERNS:

  Step 1: Put your objection in writing. Email to your manager and their
  manager. State specifically: "I believe this system poses a safety risk
  because [data]. I recommend [change]. I am documenting this objection."

  Step 2: Escalate to the company's safety officer or chief engineer.
  Aviation companies have mandatory safety reporting structures.

  Step 3: If the company still refuses, consider external reporting.
  The FAA has whistleblower protections for aviation safety concerns.
  The ACM Code (Principle 1.04) states: "Disclose to appropriate persons
  or authorities any actual or potential danger to the user, the public,
  or the environment, that they reasonably believe to be associated with
  software or related documents."

  Step 4: As a last resort, refuse to work on the project. Your
  professional obligation to public safety supersedes your employment.
  Document everything for legal protection.
""")


# === Exercise 2: Bias Audit Design ===

def exercise_2():
    """Design a bias audit plan for a job applicant scoring model."""

    print("EXERCISE 2: Bias Audit — Job Applicant Scoring Model")
    print("=" * 65)

    print("""
  AUDIT PLAN:

  1. DATA ANALYSIS:

  a) Examine training data:
     - What features are used? (education, experience, skills, location, etc.)
     - Are any features proxies for protected characteristics?
       e.g., ZIP code -> race/income, university name -> socioeconomic status
     - What is the demographic distribution of the training data?
       Broken down by: gender, race/ethnicity, age, disability status
     - Is the label (hired/not hired) itself biased? If historical hiring
       was biased, the model learns that bias.

  b) Examine model predictions:
     - Score distribution by demographic group
     - Selection rate by demographic group at various thresholds
     - Feature importance: which features drive the score?

  2. FAIRNESS METRICS TO COMPUTE:

  a) Demographic Parity (Statistical Parity):
     P(positive | group A) = P(positive | group B)
     "Each group should be selected at approximately equal rates."

  b) Equalized Odds:
     P(positive | group A, qualified) = P(positive | group B, qualified)
     "Among qualified candidates, the selection rate should be equal."

  c) Four-Fifths Rule (EEOC guideline):
     Selection rate of protected group >= 80% of the majority group's rate.
     e.g., If 50% of men are selected, at least 40% of women must be.

  d) Calibration:
     Among candidates scored at 0.8, the actual hire rate should be ~80%
     regardless of demographic group.

  3. IF DISPARATE IMPACT IS FOUND:

  a) Quantify the impact: What is the selection rate ratio? By how much
     does it violate the four-fifths rule?

  b) Investigate root cause: Is the bias in the training data (historical
     hiring patterns), the features (proxy variables), or the model
     architecture (e.g., overfitting to majority patterns)?

  c) Remediation options:
     - Remove or transform proxy features
     - Re-weight training data to balance demographic representation
     - Apply fairness constraints during model training (e.g., equalized
       odds constraint)
     - Use a different threshold per group to achieve demographic parity
       (legally complex — consult legal)
     - As a last resort: do not use the model for the affected group

  d) Stakeholder communication: Present findings to hiring managers and
     legal team BEFORE any deployment decision. The decision to deploy
     or not deploy is a business/legal decision, not an engineering one.

  4. MODEL CARD DOCUMENTATION:

  The model card should include:
  - Model purpose and intended use
  - Training data description (source, size, demographics)
  - Performance metrics overall and per demographic group
  - Fairness metrics (demographic parity, equalized odds, four-fifths)
  - Known limitations and failure modes
  - Ethical considerations and potential for harm
  - Recommended monitoring plan after deployment
  - Contact information for the responsible team
""")


# === Exercise 3: Open Source License Compatibility ===

def exercise_3():
    """Assess open source license compatibility for a commercial product."""

    print("EXERCISE 3: Open Source License Compatibility")
    print("=" * 65)

    licenses = [
        {
            "library": "Library A (MIT License)",
            "can_use": "YES — fully compatible with closed-source commercial products",
            "restrictions": (
                "MIT is the most permissive common license. Requirements:\n"
                "      - Include the MIT license text and copyright notice in your distribution\n"
                "      - That's it. No source disclosure required."
            ),
            "risk": "Minimal. MIT is the safest choice for commercial use.",
        },
        {
            "library": "Library B (GPL v2)",
            "can_use": "NO — creates a conflict with closed-source commercial product",
            "restrictions": (
                "GPL v2 is a 'copyleft' license. If you link GPL code into your\n"
                "      application, the ENTIRE application must be distributed under GPL.\n"
                "      This means you must open-source your own code.\n"
                "      - 'Linking' includes static linking, dynamic linking, and in many\n"
                "        interpretations, importing a Python module.\n"
                "      - You cannot distribute a closed-source product containing GPL code."
            ),
            "risk": (
                "HIGH. This is the library that creates a potential conflict.\n"
                "      Options:\n"
                "      1. Find a non-GPL alternative (e.g., MIT or Apache-licensed)\n"
                "      2. Use the library as a separate process communicating via IPC/API\n"
                "         (the 'GPL boundary' argument — legally debated)\n"
                "      3. Open-source your product under GPL (usually not acceptable)\n"
                "      4. Contact the library author to negotiate a commercial license"
            ),
        },
        {
            "library": "Library C (Apache 2.0)",
            "can_use": "YES — compatible with closed-source commercial products",
            "restrictions": (
                "Apache 2.0 is permissive with additional patent protections.\n"
                "      Requirements:\n"
                "      - Include the Apache license text and NOTICE file\n"
                "      - State any modifications you made to the library\n"
                "      - Include a patent grant (protects you from patent claims by\n"
                "        contributors)\n"
                "      You do NOT need to open-source your own code."
            ),
            "risk": "Low. Apache 2.0 is widely used in commercial products.",
        },
    ]

    for lib in licenses:
        print(f"\n  {lib['library']}")
        print(f"    Can use in closed-source product? {lib['can_use']}")
        print(f"    Restrictions: {lib['restrictions']}")
        print(f"    Risk: {lib['risk']}")

    print("""
  SUMMARY:
  - MIT: Use freely. Include license notice.
  - Apache 2.0: Use freely. Include license + NOTICE. Bonus: patent protection.
  - GPL v2: CANNOT use in closed-source product without open-sourcing your code.
    This is the CONFLICT. Find an alternative or isolate behind an API boundary.

  NOTE: LGPL (Lesser GPL) would allow linking without copyleft infection,
  but GPL v2 does not have this exception.
""")


# === Exercise 4: Volkswagen Defeat Device Case Study ===

def exercise_4():
    """Ethical decision-making for the VW defeat device scenario."""

    print("EXERCISE 4: Case Study — Volkswagen Defeat Device")
    print("=" * 65)

    print("""
  SCENARIO: You are a software engineer at VW in 2010. You are asked to
  implement emissions defeat device logic as a "routine feature."

  STAKEHOLDERS:
  1. VW Customers — Trust VW's environmental claims; harmed by deception
  2. The Public — Breathes polluted air; health impacts from excess NOx
  3. Environmental Regulators (EPA, EU) — Rely on test results for policy
  4. VW Shareholders — Financial interest in sales; harmed by scandal
  5. VW Engineers (you and colleagues) — Professional reputation, careers
  6. VW Management — Pressing for the feature; facing competitive pressure
  7. Competitors — Disadvantaged if VW cheats and they comply honestly
  8. Future Generations — Environmental damage is long-term

  ETHICAL PRINCIPLES THAT APPLY:

  1. ACM Principle 1.01 — Public Interest:
     "Accept full responsibility for your own work." The defeat device
     causes real harm: excess NOx emissions cause respiratory disease
     (estimated 59 premature deaths in the US alone, per MIT study).

  2. ACM Principle 1.02 — Avoid Harm:
     "Avoid harm to others." The software deliberately produces 40x the
     legal limit of NOx during normal driving. This is active harm.

  3. ACM Principle 1.06 — Privacy and Dignity:
     "Respect the privacy of others." Customers who chose VW for
     environmental reasons were deceived. Their trust was violated.

  4. IEEE Principle 5 — Quality of Life:
     "Maintain the quality of life." Air pollution from 11 million
     affected vehicles degrades public health.

  OPTIONS AVAILABLE TO THE ENGINEER:

  Option A: Implement the feature as requested.
    Consequence: You become complicit in fraud. If discovered, criminal
    liability (VW engineers were sentenced to prison in the US).
    Ethical assessment: WRONG. Violates multiple principles.

  Option B: Refuse and stay silent.
    Consequence: Someone else implements it. The harm still occurs.
    You avoid direct complicity but fail to prevent harm.
    Ethical assessment: INSUFFICIENT. You have knowledge of fraud.

  Option C: Raise the concern internally.
    Consequence: If management listens, the problem is solved. If not,
    you have documented your objection but the harm continues.
    Ethical assessment: NECESSARY FIRST STEP. But not sufficient alone.

  Option D: Refuse, raise internally, and if ignored, report externally.
    Consequence: Whistleblower protections apply (EU Directive 2019/1937,
    US Dodd-Frank Act). Career risk exists but legal protections help.
    Ethical assessment: The CORRECT action under the Code of Ethics.

  WHAT I WOULD DO:

  1. REFUSE to implement the defeat device. State clearly: "I believe this
     is illegal and I will not write this code."

  2. DOCUMENT my refusal in writing (email to manager, CC to legal).
     Include: what I was asked to do, why I believe it is wrong, and what
     I recommend instead (e.g., invest in actual emissions reduction
     technology, even if it costs more or delays the product).

  3. ESCALATE internally: Report to the ethics/compliance department
     (Volkswagen has one). Report to the board if compliance is captured.

  4. If the company proceeds despite my objection: REPORT EXTERNALLY
     to the relevant regulatory authority (EPA in the US, European
     Environment Agency in the EU). Use whistleblower channels.

  5. LEAVE the company if it becomes clear that fraud is systemic and
     the organization is unwilling to change.

  WHY THIS IS THE RIGHT APPROACH:
  - The ACM Code explicitly states that public safety overrides employer loyalty.
  - Engineers who implemented the device faced criminal prosecution.
  - The total cost to VW exceeded $35 billion in fines and settlements.
  - The engineers who could have prevented it did not — not because they
    lacked ethics, but because they lacked the courage to act on them.
""")


# === Exercise 5: Career Planning ===

def exercise_5():
    """10-year career plan for a distributed systems principal engineer."""

    print("EXERCISE 5: 10-Year Career Plan — Principal Engineer (Distributed Systems)")
    print("=" * 65)

    plan = {
        "Years 1-3: Senior Software Engineer": {
            "skills": [
                "Master one programming language deeply (Go or Rust for systems work)",
                "Learn distributed systems fundamentals: CAP theorem, consensus (Raft/Paxos), "
                "eventual consistency, distributed transactions",
                "Build proficiency in Kubernetes, gRPC, and observability (Prometheus, Grafana, Jaeger)",
                "Develop strong debugging skills for distributed systems (network partitions, clock skew)",
            ],
            "credentials": [
                "AWS Solutions Architect Professional certification",
                "Contribute to a distributed systems open-source project (etcd, CockroachDB, TiKV)",
            ],
            "community": [
                "Start a technical blog: write about distributed systems problems you solve at work",
                "Attend Strange Loop, KubeCon, or USENIX conferences",
                "Join the Papers We Love community — present a distributed systems paper",
            ],
            "reputation": [
                "Publish 6-12 blog posts covering real-world distributed systems challenges",
                "Give a lightning talk at a local meetup",
            ],
        },
        "Years 4-6: Staff Engineer": {
            "skills": [
                "Design large-scale systems: multi-region architectures, global databases",
                "Learn formal methods basics: TLA+ for modeling distributed protocols",
                "Develop leadership skills: mentor junior engineers, lead design reviews",
                "Study reliability engineering: chaos engineering, disaster recovery, SLOs",
                "Broaden beyond your stack: understand databases (CockroachDB, Spanner internals), "
                "message queues (Kafka internals), and storage systems",
            ],
            "credentials": [
                "Consider a part-time MS in Computer Science (distributed systems focus) "
                "or complete MIT 6.824 (Distributed Systems) online",
                "Google Cloud Professional Cloud Architect certification",
            ],
            "community": [
                "Speak at a major conference (KubeCon, Strange Loop, QCon)",
                "Contribute a significant feature or fix to a CNCF project",
                "Mentor 2-3 engineers through their career growth",
            ],
            "reputation": [
                "Publish a case study or experience report at a workshop (HotOS, SREcon)",
                "Be recognized as the go-to person for distributed systems at your company",
            ],
        },
        "Years 7-10: Principal Engineer": {
            "skills": [
                "Architect organization-wide technical strategy",
                "Navigate the business side: understand P&L impact of technical decisions",
                "Develop influence without authority: persuade across teams and orgs",
                "Stay current: follow research from NSDI, OSDI, SOSP, VLDB",
                "Specialize deeply: pick a niche (e.g., global consensus, conflict-free "
                "replicated data types, edge computing)",
            ],
            "credentials": [
                "No certifications needed at this level — your work IS your credential",
                "Consider writing a book chapter or co-authoring a paper",
            ],
            "community": [
                "Serve on a conference program committee (reviewer)",
                "Maintain an active presence: blog, conference talks, podcast appearances",
                "Build relationships with peers at other companies (architecture review exchanges)",
                "Mentor staff engineers who aspire to principal level",
            ],
            "reputation": [
                "Be known outside your company as a distributed systems expert",
                "Your blog posts are cited by others in the field",
                "Companies reach out to you for advice on distributed systems architecture",
            ],
        },
    }

    for phase, activities in plan.items():
        print(f"\n  {'=' * 60}")
        print(f"  {phase}")
        print(f"  {'=' * 60}")
        for category, items in activities.items():
            print(f"\n    {category.upper()}:")
            for item in items:
                print(f"      - {item}")

    print("""
  KEY PRINCIPLES THROUGHOUT:

  1. T-SHAPED EXPERTISE: Deep in distributed systems, broad enough to
     communicate with frontend, ML, data, and product teams.

  2. BUILD IN PUBLIC: Blog, speak, open-source. Your external reputation
     is what separates a principal engineer from a very experienced senior.

  3. COMPOUND INTEREST: Each year's learning builds on the last. A blog post
     in year 2 becomes a conference talk in year 4 becomes a book chapter
     in year 8. Start creating content early.

  4. RELATIONSHIPS OVER CREDENTIALS: After year 5, certifications matter
     less than who knows your work and trusts your judgment.
""")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Ethical Analysis — Airline Scheduling", exercise_1),
        ("Exercise 2: Bias Audit Design", exercise_2),
        ("Exercise 3: Open Source Licensing", exercise_3),
        ("Exercise 4: VW Defeat Device Case Study", exercise_4),
        ("Exercise 5: Career Planning", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 65}")
        print(f"=== {title} ===")
        print("=" * 65)
        func()

    print("\nAll exercises completed!")
