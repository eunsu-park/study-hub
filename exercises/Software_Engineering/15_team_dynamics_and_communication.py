"""
Exercises for Lesson 15: Team Dynamics and Communication
Topic: Software_Engineering

Solutions to practice problems from the lesson.
Covers Conway's Law, code review, Team Topologies, retrospectives, psychological safety.
"""


# === Exercise 1: Conway's Law Analysis ===

def exercise_1():
    """Analyze Conway's Law for a hypothetical organization."""

    print("EXERCISE 1: Conway's Law Analysis")
    print("=" * 65)

    print("""
  ORGANIZATION STRUCTURE (hypothetical e-commerce company):

  VP Engineering
    |
    +-- Frontend Team (6 engineers) — React SPA
    +-- Backend Team (8 engineers) — Python API
    +-- Data Team (4 engineers) — Analytics pipeline, data warehouse
    +-- Infrastructure Team (3 engineers) — AWS, Kubernetes, CI/CD
    +-- Mobile Team (4 engineers) — iOS + Android apps

  CONWAY'S LAW PREDICTION:
  "Organizations design systems which mirror their own communication structure."

  Predicted architecture:
  1. SEPARATE FRONTEND AND BACKEND: Because Frontend and Backend are
     separate teams, the system will have a clear API boundary between
     them. The React SPA communicates with the Python API via REST/GraphQL.
     The frontend team does NOT access the database directly.

  2. BFF PATTERN (Backend for Frontend): The Mobile team is separate from
     Frontend. Prediction: there will be a separate API layer (or at least
     different endpoints) for mobile vs web, because each team optimizes
     for its own client's needs.

  3. DATA SILO: The Data Team operates independently. Prediction: the
     analytics pipeline will be a separate system that reads from the
     production database via CDC or batch ETL, not integrated into the
     main API. The data warehouse is a separate system with its own schema.

  4. INFRASTRUCTURE AS A PLATFORM: The Infra team provides Kubernetes and
     CI/CD as a platform. Prediction: there is a clear "platform boundary"
     with Terraform modules, Helm charts, and deployment pipelines that
     other teams consume as a service.

  5. PAIN POINT — FRONTEND-BACKEND COUPLING: Because the teams are
     separate, the API contract becomes a negotiation bottleneck. Prediction:
     there will be frequent disagreements about API design, slow iterations
     when the frontend needs a new field, and possibly an over-fetching or
     under-fetching problem (which GraphQL would solve).

  COMPARISON TO REALITY (if known):
  This prediction closely matches typical companies structured this way.
  The key insight: if you want microservices organized by business domain
  (orders, payments, users), your TEAMS must be organized by business
  domain — not by technology layer (frontend, backend, data).

  CONWAY'S LAW REMEDY:
  If the company wants to move to domain-oriented microservices:
  - Reorganize into cross-functional teams: "Orders Team" has frontend +
    backend + data engineers who OWN the orders domain end-to-end.
  - Apply the Inverse Conway Maneuver: design the team structure to
    match the desired architecture, and the architecture will follow.
""")


# === Exercise 2: Code Review Rewrite ===

def exercise_2():
    """Rewrite a harsh code review comment to be constructive."""

    print("EXERCISE 2: Code Review Comment Rewrite")
    print("=" * 65)

    print("""
  ORIGINAL COMMENT:
  "This is wrong. Why would you use a list here when a dict is obviously
  faster? Rewrite this."

  PROBLEMS WITH THIS COMMENT:
  1. "This is wrong" — Judgmental, not specific about WHAT is wrong
  2. "Why would you" — Implies the author is incompetent
  3. "Obviously faster" — Assumes the reviewer's knowledge is universal
  4. "Rewrite this" — Demands action without explaining the benefit
  5. No code example — The author has to guess what the reviewer wants

  REWRITTEN COMMENT:

  ---
  Nit: Consider using a `dict` here instead of a `list` for the lookup.

  Currently this does a linear scan of the list on every call:
  ```python
  # Current: O(n) per lookup
  for item in items:
      if item.id == target_id:
          return item
  ```

  With a dict keyed by `id`, the lookup becomes O(1):
  ```python
  # Suggested: O(1) per lookup
  items_by_id = {item.id: item for item in items}
  return items_by_id.get(target_id)
  ```

  This matters because `find_item()` is called once per request in the
  hot path. With 10K items, the list scan adds ~2ms per request.

  If the list is small (< 50 items) or this code is rarely called, the
  current approach is fine and this is just a nit. What are your thoughts?
  ---

  WHY THIS VERSION IS BETTER:
  1. Specific: Shows the exact code and the proposed change
  2. Educational: Explains the O(n) vs O(1) performance difference
  3. Contextual: Explains WHY it matters (hot path, 10K items)
  4. Respectful: Acknowledges the current approach may be fine
  5. Collaborative: Ends with "What are your thoughts?" not "Rewrite this"
  6. Labeled: "Nit" signals this is a suggestion, not a blocking issue
""")


# === Exercise 3: Team Type Classification ===

def exercise_3():
    """Design a 60-person engineering org using Team Topologies."""

    print("EXERCISE 3: Team Topologies — 60-Person Engineering Org")
    print("=" * 65)

    print("""
  CONTEXT: 60 engineers, SaaS product, 3 product areas, K8s infrastructure,
  shared design system.

  TEAM STRUCTURE (10 teams):

  STREAM-ALIGNED TEAMS (6 teams, ~7-8 people each):
  These teams own end-to-end delivery for a business domain.

  1. User Management Team (8 people)
     Domain: Registration, authentication, profiles, permissions
     Includes: 2 frontend, 3 backend, 1 QA, 1 designer, 1 PM

  2. User Management - Growth Team (7 people)
     Domain: Onboarding, activation, retention features
     Includes: 2 frontend, 2 backend, 1 data engineer, 1 QA, 1 PM

  3. Billing Team (7 people)
     Domain: Subscriptions, invoicing, payments, pricing
     Includes: 2 frontend, 3 backend, 1 QA, 1 PM

  4. Analytics - Dashboard Team (7 people)
     Domain: Customer-facing analytics dashboards, reports
     Includes: 2 frontend, 2 backend, 1 data engineer, 1 QA, 1 PM

  5. Analytics - Pipeline Team (6 people)
     Domain: Data ingestion, transformation, data warehouse
     Includes: 4 data/backend engineers, 1 QA, 1 PM

  6. Analytics - Insights Team (6 people)
     Domain: ML-powered insights, recommendations, alerts
     Includes: 2 ML engineers, 2 backend, 1 QA, 1 PM

  PLATFORM TEAM (1 team, 7 people):
  Provides internal developer platform as a service.

  7. Platform Team (7 people)
     Domain: Kubernetes, CI/CD pipelines, observability, developer tooling
     Includes: 4 platform engineers, 2 SREs, 1 engineering manager
     Provides: Self-service deployment, monitoring dashboards, service mesh

  ENABLING TEAMS (2 teams):
  Help stream-aligned teams adopt new practices.

  8. Developer Experience Team (3 people)
     Domain: Developer onboarding, coding standards, shared libraries
     Includes: 2 senior engineers, 1 tech writer
     Mode: Facilitating — helps teams, does not own features

  9. Security Enabling Team (3 people)
     Domain: Security practices, SAST/DAST integration, threat modeling
     Includes: 2 security engineers, 1 compliance specialist
     Mode: Facilitating — runs security training, reviews critical code

  COMPLICATED-SUBSYSTEM TEAM (1 team):

  10. Design System Team (6 people)
      Domain: Shared component library, design tokens, accessibility
      Includes: 3 frontend engineers, 2 designers, 1 QA
      Mode: X-as-a-Service — provides components consumed by all stream teams

  INTERACTION MODES:

  Stream Teams <-> Platform Team: X-as-a-Service
    Platform provides self-service tools. Stream teams consume without
    needing to understand Kubernetes internals.

  Stream Teams <-> Design System Team: X-as-a-Service
    Design System provides React components. Stream teams import and use
    them without modifying the component library directly.

  Stream Teams <-> Enabling Teams: Facilitating
    Enabling teams help stream teams adopt practices (security reviews,
    developer onboarding) then step back. Not a permanent dependency.

  Stream Teams <-> Stream Teams: Collaboration (when needed)
    e.g., User Management and Billing collaborate on the payment
    authorization flow, then return to independent operation.

  TOTAL: 60 engineers across 10 teams
  Stream-aligned: 41 | Platform: 7 | Enabling: 6 | Complicated-subsystem: 6
""")


# === Exercise 4: Retrospective for Slow Code Reviews ===

def exercise_4():
    """Design a retrospective for slow PRs and unclear service ownership."""

    print("EXERCISE 4: Retrospective — Slow Code Reviews + Unclear Ownership")
    print("=" * 65)

    print("""
  FORMAT: Start-Stop-Continue (simple, action-oriented)

  AGENDA (45 minutes):

  [0:00 - 0:03] OPENING (3 min)
    - Read the prime directive
    - State the two focus areas: (1) slow code reviews, (2) legacy service ownership
    - "We are here to find improvements, not assign blame."

  [0:03 - 0:15] SILENT BRAINSTORMING (12 min)
    Each team member writes sticky notes in three columns:
    - START doing (things we should begin)
    - STOP doing (things that are not working)
    - CONTINUE doing (things that are working well)

    Prompt questions:
    - "What happens when you submit a PR? How long until first review?"
    - "When the legacy service breaks at 2 AM, who fixes it? How do you know?"
    - "What would make you WANT to review PRs faster?"

  [0:15 - 0:30] CLUSTER AND DISCUSS (15 min)
    Facilitator reads notes, groups by theme. Expected clusters:
    - "PRs sit for 2+ days because no one is assigned"
    - "Legacy service has no owner — whoever is on-call gets stuck with it"
    - "PR reviews take long because PRs are too large"

    For each cluster, ask:
    - "Can someone give a specific example?"
    - "What is the root cause here?"
    - "What ONE change would have the biggest impact?"

  [0:30 - 0:40] DOT-VOTE AND ACTION ITEMS (10 min)
    3 dots per person on the most impactful items.
    Top 3 become action items with SMART criteria:

    Example action items:
    1. "Assign a primary reviewer to every PR automatically using CODEOWNERS.
       Owner: Tech Lead. Due: End of this sprint."
    2. "Declare an explicit owner for the legacy payment service.
       Owner: Engineering Manager (to decide). Due: End of week."
    3. "Enforce a PR size limit of 400 lines. Larger PRs must be split.
       Owner: Team agreement. Starts: Next sprint."

  [0:40 - 0:45] CLOSING (5 min)
    - Read back the 3 action items with owners and dates
    - "These will be reviewed at the START of next sprint's retro."
    - Quick temperature check: thumbs up/middle/down on confidence
      that these actions will be completed.

  KEY FACILITATION TECHNIQUES:
  1. Force specificity: When someone says "reviews are slow," ask
     "How slow? Give me a number." Data drives action.
  2. One-conversation rule: Only one person speaks at a time. Use a
     talking token if people talk over each other.
  3. Action item quality gate: Every action must have an OWNER and a
     DATE. "We should review PRs faster" is not an action item.
     "Set up CODEOWNERS by Friday" is.
""")


# === Exercise 5: Psychological Safety Assessment ===

def exercise_5():
    """Five actions to improve psychological safety."""

    print("EXERCISE 5: Psychological Safety Improvement Plan")
    print("=" * 65)

    print("""
  OBSERVATIONS:
  - Same 2-3 engineers always speak in meetings
  - Junior engineers rarely contribute
  - Production mistakes are discussed behind closed doors, not openly

  FIVE ACTIONS TO IMPROVE PSYCHOLOGICAL SAFETY:

  1. MODEL VULNERABILITY AS A LEADER
     Action: In the next team meeting, share a mistake YOU made recently.
     Example: "Last week I pushed a config change that broke staging. Here
     is what I learned and what I changed in my process."
     Rationale: If the team lead admits mistakes openly, it signals that
     mistakes are safe to discuss. Leaders must go first — you cannot ask
     the team to be vulnerable if you are not.

  2. STRUCTURED TURN-TAKING IN MEETINGS
     Action: Use round-robin for technical discussions. Go around the table
     and ask each person for their input. "Alex, what do you think about
     this approach?" Then: "Sam, anything to add?"
     Rationale: Junior engineers often have valuable insights but lack the
     social capital to interrupt senior engineers. Round-robin ensures every
     voice is heard. Over time, juniors build confidence to speak without
     being explicitly called on.

  3. BLAMELESS INCIDENT REVIEWS (PUBLIC)
     Action: Hold blameless postmortems for ALL incidents, not just major
     ones. Use "the system" language, never individual names. Publish
     the postmortem summary to the team channel.
     Rationale: When mistakes are discussed privately, the implicit message
     is "mistakes are shameful." Public, blameless reviews normalize the
     idea that mistakes are learning opportunities, not career threats.

  4. CREATE LOW-STAKES FEEDBACK OPPORTUNITIES
     Action: Introduce anonymous retrospective input (Miro, Google Forms).
     Start each retro with 5 minutes of anonymous sticky note writing.
     Also introduce weekly "learning moments" — a 5-minute Slack thread
     where anyone shares something they learned that week.
     Rationale: Anonymous input removes the fear of judgment. "Learning
     moments" reframe mistakes as knowledge-sharing events. Both lower
     the barrier to contribution.

  5. RECOGNIZE AND REWARD QUESTION-ASKING
     Action: When a junior engineer asks a question in a meeting, explicitly
     validate it: "That is a great question. I think several of us had the
     same uncertainty." After meetings, privately thank people who spoke up.
     Rationale: Edmondson's research shows that the response to the first
     act of vulnerability determines whether others follow. If questions
     are met with impatience ("that's obvious"), no one will ask again.
     If they are met with appreciation, the behavior spreads.

  MEASUREMENT:
  After 3 months, measure improvement with:
  - Anonymous survey: "I feel safe to take risks on this team" (1-5 scale)
  - Meeting participation: count unique speakers per meeting (target: 80%+ of attendees)
  - Incident reporting time: time from detection to team notification (should decrease)
""")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Conway's Law Analysis", exercise_1),
        ("Exercise 2: Code Review Rewrite", exercise_2),
        ("Exercise 3: Team Type Classification", exercise_3),
        ("Exercise 4: Retrospective Facilitation", exercise_4),
        ("Exercise 5: Psychological Safety", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 65}")
        print(f"=== {title} ===")
        print("=" * 65)
        func()

    print("\nAll exercises completed!")
