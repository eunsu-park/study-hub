"""
Exercises for Lesson 11: Software Maintenance and Evolution
Topic: Software_Engineering

Solutions to practice problems from the lesson.
Covers maintenance classification, Lehman's Laws, Strangler Fig, refactoring, deprecation.
"""


# === Exercise 1: Maintenance Classification ===

def exercise_1():
    """Classify maintenance tasks by type."""

    print("EXERCISE 1: Maintenance Classification")
    print("=" * 65)

    tasks = [
        {
            "task": "(a) Fixing a crash when a user uploads a PNG with EXIF data",
            "type": "Corrective Maintenance",
            "justification": (
                "A defect (crash) exists in production. Fixing bugs discovered after "
                "deployment is the textbook definition of corrective maintenance."
            ),
        },
        {
            "task": "(b) Updating the SMS library after the vendor retires v1 API",
            "type": "Adaptive Maintenance",
            "justification": (
                "The software must adapt to a change in its external environment "
                "(vendor API retirement). The software's functionality is unchanged — "
                "it still sends SMS — but the implementation must change."
            ),
        },
        {
            "task": "(c) Adding dark mode to the UI",
            "type": "Perfective Maintenance",
            "justification": (
                "Dark mode is a new feature requested by users to improve the product. "
                "No bug is being fixed; no environmental change forces this. It is an "
                "enhancement — adding functionality to satisfy a user need."
            ),
        },
        {
            "task": "(d) Refactoring the 3,000-line OrderProcessor class into smaller components",
            "type": "Preventive Maintenance",
            "justification": (
                "No bug exists today. No user-visible behavior changes. The refactoring "
                "improves internal quality to prevent future defects and reduce maintenance "
                "cost. This is analogous to oil changes for a car."
            ),
        },
        {
            "task": "(e) Adding rate limiting to the API before a marketing campaign",
            "type": "Preventive Maintenance",
            "justification": (
                "The system works today, but the anticipated traffic spike could cause "
                "failures. Adding rate limiting is a proactive measure to prevent a future "
                "problem. Some argue this is adaptive (adapting to expected environment "
                "change), but since the campaign has not happened yet, preventive is more "
                "accurate."
            ),
        },
    ]

    for t in tasks:
        print(f"\n  {t['task']}")
        print(f"    Type: {t['type']}")
        print(f"    Justification: {t['justification']}")


# === Exercise 2: Lehman's Laws Analysis ===

def exercise_2():
    """Analyze Lehman's Laws using Linux kernel as an example."""

    print("EXERCISE 2: Lehman's Laws — Linux Kernel Analysis")
    print("=" * 65)

    print("""
  PROJECT: Linux Kernel (chosen for its extensive public history)

  LAW I — CONTINUING CHANGE:
  "An E-type system must be continually adapted or it becomes progressively
  less satisfactory."

  Example: Linux kernel's support for new hardware and syscalls.
  - The Linux kernel releases a new version approximately every 9-10 weeks.
  - Between kernel 5.0 (March 2019) and 6.7 (January 2024), over 300,000
    commits were made.
  - If the kernel stopped adapting, it would quickly lose compatibility with
    new CPUs (Apple M-series, AMD Zen 4), new peripherals (USB4, Thunderbolt 5),
    and new security requirements (speculative execution mitigations).
  - Proof: The kernel added Rust language support (6.1, Dec 2022) specifically
    to adapt to the growing demand for memory-safe systems programming.

  LAW II — INCREASING COMPLEXITY:
  "As an E-type system evolves, its complexity increases unless work is done
  to maintain or reduce it."

  Example: The kernel's growth in lines of code and subsystem interdependencies.
  - Kernel 1.0 (1994): ~176,000 lines of code
  - Kernel 5.0 (2019): ~26,000,000 lines of code
  - Kernel 6.7 (2024): ~36,000,000+ lines of code
  - The driver subsystem alone is larger than most entire operating systems.
  - Config options grew from hundreds to over 15,000 Kconfig symbols.

  HOW THE PROJECT COUNTERACTS COMPLEXITY:
  1. Subsystem maintainer model: Each subsystem (networking, filesystem,
     drivers) has designated maintainers who enforce local quality standards.
     This prevents a "tragedy of the commons" where everyone adds code but
     no one refactors.

  2. Aggressive deprecation: The kernel regularly removes deprecated APIs
     and unused drivers. `make allnoconfig` keeps the minimum viable kernel
     small.

  3. Static analysis and testing: Tools like sparse, coccinelle (automated
     code transformation), and the kernel's kselftest framework detect and
     fix complexity-induced bugs.

  4. Coding style enforcement: checkpatch.pl rejects commits that violate
     the kernel's coding standards, preventing style divergence.

  5. Periodic cleanups: Major releases often include "tree-wide" cleanups
     that refactor common patterns across the entire codebase.
""")


# === Exercise 3: Strangler Fig Design ===

def exercise_3():
    """Strangler Fig migration plan for an HR monolith."""

    print("EXERCISE 3: Strangler Fig Migration — HR Monolith")
    print("=" * 65)

    print("""
  MONOLITH: Java HR application (Payroll, Leave, Performance Reviews, Onboarding)

  FIRST SERVICE TO EXTRACT: Leave Management

  WHY LEAVE MANAGEMENT FIRST:
  1. Lowest coupling: Leave management has the fewest dependencies on other
     modules. It primarily reads employee data (one-way dependency) and does
     not modify payroll or performance records.
  2. Moderate complexity: Simpler than payroll (which has tax calculations,
     government reporting) but complex enough to validate the migration approach.
  3. High business value: Leave is used daily by all employees. Improvement
     is immediately visible to stakeholders, building momentum for further
     migration.
  4. Clear boundary: Leave has a well-defined API surface: request leave,
     approve leave, check balance, view calendar. Easy to define an interface.

  ROUTING PROXY STRATEGY:

  Step 1: Deploy a reverse proxy (nginx or API gateway) in front of the monolith.
  All traffic flows through the proxy to the monolith — no behavior change.

    Client -> [Proxy] -> [Monolith (Payroll + Leave + Perf + Onboarding)]

  Step 2: Build the new Leave microservice. It reads from its own database
  (replicated from the monolith's leave tables via CDC or ETL).

    Client -> [Proxy] -> [Monolith] (still handles leave)
                    \\--> [Leave Service] (shadow mode: receives traffic but
                                          responses are discarded)

  Step 3: Route read-only leave endpoints to the new service.
  The proxy routes GET /leave/* to the new service, POST /leave/* still
  goes to the monolith.

    Client -> [Proxy] --GET /leave/*--> [Leave Service]
                    \\--POST /leave/*--> [Monolith]

  Step 4: Route all leave traffic to the new service.
  The monolith's leave module is now dormant.

    Client -> [Proxy] --/leave/*-------> [Leave Service]
                    \\--/payroll/*------> [Monolith]
                    \\--/performance/*--> [Monolith]
                    \\--/onboarding/*---> [Monolith]

  Step 5: Remove the leave module from the monolith (the "strangle event").

  STRANGLE EVENT COMPLETION CRITERIA:
  1. Zero traffic reaches the monolith's leave endpoints for 30 days.
  2. All leave data has been migrated to the new service's database.
  3. The new service passes all functional tests that the monolith's
     leave module previously passed.
  4. Performance (latency, throughput) of the new service meets or
     exceeds the monolith.
  5. The leave module's code has been removed from the monolith's
     codebase and the monolith builds successfully without it.
  6. Monitoring confirms no 500 errors from the leave service for 14 days.
""")


# === Exercise 4: Refactoring Practice ===

def exercise_4():
    """Apply named refactoring patterns to obfuscated code."""

    print("EXERCISE 4: Refactoring Practice")
    print("=" * 65)

    print("""
  ORIGINAL CODE:
  def x(d, t, u, s):
      r = 0
      for i in d:
          if i['type'] == t and i['user'] == u and i['status'] == s:
              r += i['amount']
      if r > 1000:
          r = r * 0.95
      return r

  ANALYSIS: This function filters a list of transaction dicts by type, user,
  and status, sums amounts, and applies a 5% discount if the total exceeds 1000.
""")

    # Refactored version
    print("  REFACTORING 1: Rename Method + Rename Variables")
    print("  Pattern: Rename Method (Fowler)")
    print("  Why: Names like x, d, t, u, s, r, i are meaningless.")
    print()

    refactored_1 = '''  def calculate_total_with_discount(transactions, txn_type, user_id, status):
      total = 0
      for txn in transactions:
          if txn['type'] == txn_type and txn['user'] == user_id and txn['status'] == status:
              total += txn['amount']
      if total > 1000:
          total = total * 0.95
      return total'''
    print(refactored_1)

    print("\n  REFACTORING 2: Extract Method (filter + sum)")
    print("  Pattern: Extract Method (Fowler)")
    print("  Why: The filtering and summing logic is a distinct responsibility.")
    print()

    refactored_2 = '''  def _filter_transactions(transactions, txn_type, user_id, status):
      """Filter transactions matching criteria."""
      return [
          txn for txn in transactions
          if txn['type'] == txn_type
          and txn['user'] == user_id
          and txn['status'] == status
      ]

  def calculate_total_with_discount(transactions, txn_type, user_id, status):
      matching = _filter_transactions(transactions, txn_type, user_id, status)
      total = sum(txn['amount'] for txn in matching)
      if total > 1000:
          total = total * 0.95
      return total'''
    print(refactored_2)

    print("\n  REFACTORING 3: Replace Magic Number with Named Constant")
    print("  Pattern: Replace Magic Number with Symbolic Constant (Fowler)")
    print("  Why: 1000 and 0.95 are unexplained magic numbers.")
    print()

    refactored_3 = '''  BULK_DISCOUNT_THRESHOLD = 1000
  BULK_DISCOUNT_RATE = 0.05  # 5% discount

  def _filter_transactions(transactions, txn_type, user_id, status):
      return [
          txn for txn in transactions
          if txn['type'] == txn_type
          and txn['user'] == user_id
          and txn['status'] == status
      ]

  def _apply_bulk_discount(total):
      """Apply 5% discount for totals exceeding threshold."""
      if total > BULK_DISCOUNT_THRESHOLD:
          return total * (1 - BULK_DISCOUNT_RATE)
      return total

  def calculate_total_with_discount(transactions, txn_type, user_id, status):
      matching = _filter_transactions(transactions, txn_type, user_id, status)
      total = sum(txn['amount'] for txn in matching)
      return _apply_bulk_discount(total)'''
    print(refactored_3)

    print("""
  SUMMARY OF PATTERNS APPLIED:
  1. Rename Method + Rename Variables: x -> calculate_total_with_discount
  2. Extract Method: filtering logic -> _filter_transactions()
  3. Replace Magic Number: 1000 -> BULK_DISCOUNT_THRESHOLD, 0.95 -> formula
  4. Extract Method (bonus): discount logic -> _apply_bulk_discount()

  Observable behavior is preserved: same inputs produce same outputs.
  Each function is now independently testable and self-documenting.
""")


# === Exercise 5: Deprecation Policy Design ===

def exercise_5():
    """Deprecation plan for insecure authentication endpoint."""

    print("EXERCISE 5: Deprecation Policy — Authentication Endpoint")
    print("=" * 65)

    print("""
  DEPRECATION PLAN: GET /auth/token?user=X&pass=Y
  =================================================
  Replacement:     POST /auth/token (credentials in request body)
  Consumers:       50 internal services
  Security Issue:  GET parameters appear in access logs, browser history,
                   and proxy caches — credential exposure risk.

  TIMELINE (16 weeks total):

  Week 0:    ANNOUNCEMENT
    - Send company-wide email to all service owners
    - Update API documentation: mark GET endpoint as "DEPRECATED"
    - Add Sunset header to GET responses: `Sunset: <date+16 weeks>`
    - Add Deprecation header: `Deprecation: true`
    - Publish migration guide with code examples for POST endpoint

  Week 1-4:  MIGRATION SUPPORT PERIOD
    - Both endpoints active (GET and POST)
    - Provide a migration helper library (wrapper that switches GET->POST)
    - Hold office hours: 2x/week, 30 minutes, for teams needing help
    - Track migration progress on a dashboard (% of traffic on new endpoint)

  Week 4:    FIRST CHECKPOINT
    - Review dashboard: expect >50% traffic migrated
    - Personally reach out to teams still using GET
    - Escalate to engineering manager if a team refuses to migrate

  Week 8:    SECOND CHECKPOINT — WARNING PHASE
    - GET endpoint returns `Warning: 299 - "Deprecated, use POST /auth/token"`
    - Log all remaining GET consumers (IP, service name, frequency)
    - Direct outreach to remaining consumers (expect <20% traffic on GET)

  Week 12:   BROWNOUT TESTING
    - Disable GET endpoint for 1 hour (planned, announced 1 week ahead)
    - Monitor for failures and identify any unknown consumers
    - Re-enable after 1 hour
    - Contact any newly discovered consumers

  Week 14:   FINAL WARNING
    - GET endpoint returns HTTP 403 with body:
      {"error": "This endpoint is deprecated. Use POST /auth/token.
       Hard shutdown on <date>."}
    - Exception: consumers who have filed an extension request get 2 more weeks

  Week 16:   HARD CUTOVER
    - GET /auth/token returns 410 Gone permanently
    - Remove GET route from codebase
    - Archive the old endpoint code in a tagged Git release for reference

  COMMUNICATION STRATEGY:
    Channels:
    - Email (all-hands announcement + targeted to service owners)
    - Slack #api-announcements channel (weekly reminders)
    - API documentation (deprecation notice with migration guide)
    - Sprint review demos (show migration dashboard)

    Frequency:
    - Week 0: Initial announcement (email + Slack + docs)
    - Weeks 1-8: Weekly Slack reminders with migration progress %
    - Weeks 9-16: Bi-weekly direct outreach to remaining consumers

  MONITORING APPROACH:
    - Dashboard tracking: % of auth requests using GET vs POST
    - Alert: if GET traffic increases (regression detection)
    - Log analysis: unique consumer IPs and service names on GET
    - SLA: POST endpoint latency and error rate (ensure replacement is reliable)

  HARD CUTOVER CRITERIA:
    All of the following must be true:
    1. GET traffic is < 0.1% of total auth traffic for 7 consecutive days
    2. All 50 known internal consumers confirmed migrated (sign-off)
    3. POST endpoint has been stable for 30+ days (no outages)
    4. No extension requests pending
    5. Brownout test (Week 12) passed with zero critical failures
""")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: Maintenance Classification", exercise_1),
        ("Exercise 2: Lehman's Laws Analysis", exercise_2),
        ("Exercise 3: Strangler Fig Design", exercise_3),
        ("Exercise 4: Refactoring Practice", exercise_4),
        ("Exercise 5: Deprecation Policy", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 65}")
        print(f"=== {title} ===")
        print("=" * 65)
        func()

    print("\nAll exercises completed!")
