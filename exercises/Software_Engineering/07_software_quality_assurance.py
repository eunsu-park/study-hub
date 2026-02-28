"""
Exercises for Lesson 07: Software Quality Assurance
Topic: Software_Engineering

Solutions to practice problems from the lesson.
This lesson has two exercise sections: Practice Exercises (5) and Exercises (5).
Includes cyclomatic complexity calculations, Cost of Quality analysis, and code review.
"""


# =====================================================================
# PRACTICE EXERCISES (Section 12)
# =====================================================================

# === Practice Exercise 1: Cyclomatic Complexity ===
# Problem: Calculate CC of process_order(), then refactor to CC <= 4.

def practice_exercise_1():
    """Calculate and reduce cyclomatic complexity."""

    print("PRACTICE EXERCISE 1: Cyclomatic Complexity")
    print("=" * 65)

    original_code = '''
def process_order(order):
    if order is None:                        # Decision 1
        return None
    if order.status == "cancelled":          # Decision 2
        return {"error": "Order is cancelled"}
    if order.items:                           # Decision 3
        for item in order.items:             # Decision 4
            if item.quantity < 0:            # Decision 5
                raise ValueError(...)
            if item.price < 0:               # Decision 6
                raise ValueError(...)
            if item.quantity == 0:            # Decision 7
                continue
            item.total = item.price * item.quantity
    if order.discount_code:                  # Decision 8
        if order.discount_code == "SAVE10":  # Decision 9
            order.discount = 0.10
        elif order.discount_code == "SAVE20":# Decision 10
            order.discount = 0.20
        else:
            return {"error": "Invalid discount code"}
    order.total = sum(...)
    if order.discount:                       # Decision 11
        order.total *= (1 - order.discount)
    return order
'''

    print("  ORIGINAL CODE ANALYSIS:")
    print("  Decision points counted:")
    decisions = [
        "1. if order is None",
        "2. if order.status == 'cancelled'",
        "3. if order.items",
        "4. for item in order.items (loop = decision)",
        "5. if item.quantity < 0",
        "6. if item.price < 0",
        "7. if item.quantity == 0",
        "8. if order.discount_code",
        "9. if order.discount_code == 'SAVE10'",
        "10. elif order.discount_code == 'SAVE20'",
        "11. if order.discount",
    ]
    for d in decisions:
        print(f"    {d}")

    cc = 11 + 1  # CC = decisions + 1
    print(f"\n  Cyclomatic Complexity = {len(decisions)} decisions + 1 = {cc}")
    print(f"  Risk level: HIGH (CC > 10). Difficult to test and maintain.")

    print("\n  REFACTORED VERSION (CC <= 4 per function):")

    refactored = '''
# Strategy: Extract responsibilities into separate functions

DISCOUNT_CODES = {
    "SAVE10": 0.10,
    "SAVE20": 0.20,
}

def validate_order(order):
    """Validate order is processable. CC = 2."""
    if order is None:                              # Decision 1
        return {"error": "Missing order"}
    if order.status == "cancelled":                # Decision 2
        return {"error": "Order is cancelled"}
    return None  # No error

def validate_item(item):
    """Validate a single line item. CC = 3."""
    if item.quantity < 0:                          # Decision 1
        raise ValueError(f"Invalid quantity for {item.name}")
    if item.price < 0:                             # Decision 2
        raise ValueError(f"Invalid price for {item.name}")
    if item.quantity == 0:                          # Decision 3
        return False  # Skip this item
    return True

def apply_discount(order):
    """Apply discount code to order. CC = 2."""
    if not order.discount_code:                    # Decision 1
        return None
    if order.discount_code not in DISCOUNT_CODES:  # Decision 2
        return {"error": "Invalid discount code"}
    order.discount = DISCOUNT_CODES[order.discount_code]
    return None

def process_order(order):
    """Main orchestrator. CC = 3."""
    error = validate_order(order)
    if error:                                      # Decision 1
        return error
    for item in order.items:                       # Decision 2 (loop)
        if validate_item(item):
            item.total = item.price * item.quantity
    error = apply_discount(order)
    if error:                                      # Decision 3
        return error
    order.total = sum(i.total for i in order.items if hasattr(i, 'total'))
    if order.discount:
        order.total *= (1 - order.discount)
    return order
'''
    print(refactored)
    print("  Result: Each function has CC <= 4. Total behavior is preserved.")
    print("  Each function is independently testable.")


# === Practice Exercise 2: Cost of Quality Analysis ===
# Problem: Categorize costs, calculate CoQ, evaluate investment.

def practice_exercise_2():
    """Cost of Quality analysis with investment decision."""

    print("PRACTICE EXERCISE 2: Cost of Quality Analysis")
    print("=" * 65)

    costs = [
        ("Coding standards enforcement", 2000, "Prevention",
         "Prevents defects by establishing quality standards upfront"),
        ("Code review time", 8000, "Appraisal",
         "Detects defects by inspecting work products"),
        ("Unit testing", 5000, "Appraisal",
         "Detects defects by executing test cases"),
        ("Bug fixing before release", 15000, "Internal Failure",
         "Rework caused by defects found before delivery"),
        ("Customer support for production bugs", 25000, "External Failure",
         "Cost of defects discovered by customers"),
        ("Emergency hotfix deployments", 10000, "External Failure",
         "Urgent fixes for production defects"),
    ]

    categories = {"Prevention": 0, "Appraisal": 0,
                  "Internal Failure": 0, "External Failure": 0}

    print("\n  (a) Categorization:")
    print(f"  {'Activity':<35} {'Cost':>8} {'Category':<18} Rationale")
    print(f"  {'-' * 95}")
    for name, cost, cat, rationale in costs:
        categories[cat] += cost
        print(f"  {name:<35} ${cost:>6,} {cat:<18} {rationale}")

    total_coq = sum(categories.values())

    print(f"\n  (b) Total Cost of Quality: ${total_coq:,}/month")
    print(f"  {'-' * 50}")
    for cat, amount in categories.items():
        pct = (amount / total_coq) * 100
        bar = "#" * int(pct / 2)
        print(f"  {cat:<20} ${amount:>7,} ({pct:>5.1f}%) {bar}")

    prevention_appraisal = categories["Prevention"] + categories["Appraisal"]
    failure = categories["Internal Failure"] + categories["External Failure"]
    print(f"\n  Prevention + Appraisal: ${prevention_appraisal:,} "
          f"({prevention_appraisal/total_coq*100:.1f}%)")
    print(f"  Failure costs:          ${failure:,} "
          f"({failure/total_coq*100:.1f}%)")
    print(f"  Ratio: Failure costs are {failure/prevention_appraisal:.1f}x "
          f"Prevention+Appraisal costs")

    # (c) Investment analysis
    additional_prevention = 5000
    internal_reduction = 0.40
    external_reduction = 0.25

    new_internal = categories["Internal Failure"] * (1 - internal_reduction)
    new_external = categories["External Failure"] * (1 - external_reduction)
    savings_internal = categories["Internal Failure"] - new_internal
    savings_external = categories["External Failure"] - new_external
    total_savings = savings_internal + savings_external

    print(f"\n  (c) Investment Analysis: +${additional_prevention:,}/month in Prevention")
    print(f"      Internal failure reduction (40%): "
          f"${categories['Internal Failure']:,} -> ${new_internal:,.0f} "
          f"(saves ${savings_internal:,.0f})")
    print(f"      External failure reduction (25%): "
          f"${categories['External Failure']:,} -> ${new_external:,.0f} "
          f"(saves ${savings_external:,.0f})")
    print(f"      Total savings: ${total_savings:,.0f}/month")
    print(f"      Investment cost: ${additional_prevention:,}/month")
    print(f"      Net benefit: ${total_savings - additional_prevention:,.0f}/month")
    print(f"      ROI: {((total_savings - additional_prevention) / additional_prevention) * 100:.0f}%")
    print(f"\n      RECOMMENDATION: YES — invest. The $5,000 investment yields "
          f"${total_savings:,.0f} in savings, a net gain of "
          f"${total_savings - additional_prevention:,.0f}/month.")


# === Practice Exercise 3: Quality Gate Design ===
# Problem: Design a SonarQube Quality Gate for a fintech API.

def practice_exercise_3():
    """SonarQube Quality Gate for a fintech API."""

    print("PRACTICE EXERCISE 3: SonarQube Quality Gate — Fintech API")
    print("=" * 65)

    conditions = [
        {
            "metric": "Bugs (New Code)",
            "operator": "is greater than",
            "threshold": 0,
            "justification": (
                "Zero tolerance for bugs in new code. In fintech, even minor bugs "
                "can cause incorrect financial calculations or data corruption."
            ),
        },
        {
            "metric": "Vulnerabilities (New Code)",
            "operator": "is greater than",
            "threshold": 0,
            "justification": (
                "Financial systems are high-value targets. Any security vulnerability "
                "in new code must be fixed before merge. Regulatory compliance (PCI-DSS) "
                "demands this."
            ),
        },
        {
            "metric": "Code Coverage on New Code",
            "operator": "is less than",
            "threshold": "90%",
            "justification": (
                "Higher than typical 80% because financial calculations must be "
                "thoroughly tested. Edge cases in rounding, currency conversion, and "
                "boundary conditions are critical."
            ),
        },
        {
            "metric": "Duplicated Lines on New Code",
            "operator": "is greater than",
            "threshold": "3%",
            "justification": (
                "Duplicated financial logic is dangerous: a bug fix in one copy but "
                "not the other leads to inconsistent behavior. Keep duplication low."
            ),
        },
        {
            "metric": "Maintainability Rating (New Code)",
            "operator": "is worse than",
            "threshold": "A",
            "justification": (
                "Fintech systems have long lifetimes and strict audit requirements. "
                "Code must be maintainable for regulatory reviews and future changes."
            ),
        },
        {
            "metric": "Security Hotspots Reviewed",
            "operator": "is less than",
            "threshold": "100%",
            "justification": (
                "All security hotspots must be reviewed before code enters production. "
                "Unreviewed hotspots may contain authentication bypasses or injection "
                "vulnerabilities."
            ),
        },
        {
            "metric": "Reliability Rating (New Code)",
            "operator": "is worse than",
            "threshold": "A",
            "justification": (
                "Financial transactions must be reliable. A rating worse than A "
                "indicates potential runtime errors that could corrupt transactions."
            ),
        },
        {
            "metric": "Cognitive Complexity per Function",
            "operator": "exceeds",
            "threshold": 15,
            "justification": (
                "Complex financial logic must remain understandable for auditors "
                "and future developers. High cognitive complexity leads to bugs "
                "in edge cases."
            ),
        },
    ]

    print("\n  QUALITY GATE: Fintech-API-Gate")
    print(f"  {'#':<3} {'Metric':<35} {'Condition':<25} {'Threshold'}")
    print(f"  {'-' * 80}")
    for i, c in enumerate(conditions, 1):
        print(f"  {i:<3} {c['metric']:<35} {c['operator']:<25} {c['threshold']}")
        print(f"      Justification: {c['justification']}")
        print()


# === Practice Exercise 4: Technical Debt Backlog ===
# Problem: Categorize debt, prioritize, write 3-sprint plan.

def practice_exercise_4():
    """Technical debt categorization and reduction plan."""

    print("PRACTICE EXERCISE 4: Technical Debt Backlog")
    print("=" * 65)

    debt_items = [
        {
            "item": "47 functions with CC > 15 (avg CC = 22)",
            "quadrant": "Prudent-Inadvertent",
            "explanation": (
                "Complexity grew over time as features were added. The team "
                "likely did not track CC metrics. 'Now we know how we should "
                "have done it.'"
            ),
        },
        {
            "item": "Test coverage: 34%",
            "quadrant": "Reckless-Deliberate",
            "explanation": (
                "Low coverage on a production e-commerce system is a conscious "
                "trade-off (shipping fast) that the team knew was risky. "
                "'We know we should test but we chose not to.'"
            ),
        },
        {
            "item": "280 duplicated code blocks",
            "quadrant": "Prudent-Inadvertent",
            "explanation": (
                "Duplication accumulates gradually through copy-paste development. "
                "Teams often don't realize the extent until a tool measures it."
            ),
        },
        {
            "item": "3 potential SQL injection vulnerabilities",
            "quadrant": "Reckless-Inadvertent",
            "explanation": (
                "The developers likely did not know about SQL injection risks. "
                "This is not a deliberate trade-off but a knowledge gap. "
                "'What's parameterized queries?'"
            ),
        },
        {
            "item": "12 uses of deprecated API methods",
            "quadrant": "Prudent-Deliberate",
            "explanation": (
                "The team likely knew the APIs were deprecated but deliberately "
                "deferred migration. 'We'll upgrade later when we have time.'"
            ),
        },
    ]

    print("\n  (a) Fowler's Technical Debt Quadrant Classification:")
    for d in debt_items:
        print(f"\n    Item: {d['item']}")
        print(f"    Quadrant: {d['quadrant']}")
        print(f"    Rationale: {d['explanation']}")

    print("\n\n  (b) Priority Ranking (2 devs at 20% capacity = ~1.6 dev-days/sprint):")
    print("  Assuming 2-week sprints, 20% = ~2 days/sprint per dev = 4 days total/sprint\n")

    priority = [
        ("1 (CRITICAL)", "SQL injection vulnerabilities (3)",
         "Security risk — exploitable in production. Fix immediately. "
         "Estimated: 2 days (parameterize all queries)."),
        ("2 (HIGH)", "Increase test coverage (34% -> target 60%)",
         "Without tests, fixing other debt is risky — you cannot refactor "
         "safely without regression protection."),
        ("3 (MEDIUM)", "Reduce CC in high-complexity functions",
         "High CC correlates with bugs. Focus on the most critical business "
         "logic functions first (checkout, payment)."),
        ("4 (MEDIUM)", "Eliminate duplicated code blocks",
         "Duplicated code makes bug fixes incomplete. Consolidate gradually."),
        ("5 (LOW)", "Update deprecated API methods",
         "Lowest risk — deprecated methods still work. Schedule when "
         "a library upgrade forces the change."),
    ]

    for rank, item, rationale in priority:
        print(f"    Priority {rank}: {item}")
        print(f"      Rationale: {rationale}")

    print("\n\n  (c) 3-Sprint Debt Reduction Plan:")
    sprints = [
        {
            "sprint": "Sprint 1: Security + Test Foundation",
            "tasks": [
                "Fix all 3 SQL injection vulnerabilities (2 days)",
                "Add integration tests for checkout and payment paths (2 days)",
            ],
            "outcome": "Security risk eliminated. Critical paths have regression tests.",
        },
        {
            "sprint": "Sprint 2: Coverage + First Refactoring",
            "tasks": [
                "Write unit tests for 10 highest-CC functions (3 days)",
                "Refactor top 5 highest-CC functions using Extract Method (1 day)",
            ],
            "outcome": "Coverage: 34% -> ~45%. Top 5 functions: CC reduced below 10.",
        },
        {
            "sprint": "Sprint 3: Duplication + Continued Coverage",
            "tasks": [
                "Identify and consolidate top 20 duplicated code blocks (2 days)",
                "Add tests for 10 more functions; refactor 5 more (2 days)",
            ],
            "outcome": "Coverage: ~45% -> ~55%. Duplication reduced by ~40 blocks.",
        },
    ]

    for s in sprints:
        print(f"\n    {s['sprint']}")
        for task in s["tasks"]:
            print(f"      - {task}")
        print(f"      Outcome: {s['outcome']}")


# === Practice Exercise 5: Code Review ===
# Problem: Review get_user_data() function — find 5+ issues.

def practice_exercise_5():
    """Code review of get_user_data() function."""

    print("PRACTICE EXERCISE 5: Code Review — get_user_data()")
    print("=" * 65)

    print("""
  ORIGINAL CODE:
  def get_user_data(user_id):
      conn = sqlite3.connect("app.db")
      query = "SELECT * FROM users WHERE id = " + str(user_id)
      result = conn.execute(query).fetchall()
      user = {}
      for row in result:
          user['id'] = row[0]
          user['name'] = row[1]
          user['email'] = row[2]
          user['password'] = row[3]
          user['admin'] = row[4]
      return user
""")

    issues = [
        {
            "dimension": "SECURITY",
            "issue": "SQL Injection vulnerability",
            "problem": (
                "String concatenation with user_id creates a SQL injection vector. "
                "An attacker can pass user_id='1 OR 1=1' to dump all users."
            ),
            "fix": 'query = "SELECT * FROM users WHERE id = ?"\n'
                   '      result = conn.execute(query, (user_id,)).fetchall()',
        },
        {
            "dimension": "SECURITY",
            "issue": "Returning password hash to caller",
            "problem": (
                "The function returns the password field. If this data reaches "
                "an API response, it leaks password hashes to clients."
            ),
            "fix": "Exclude password from the returned dict:\n"
                   "      user = {'id': row[0], 'name': row[1], 'email': row[2], "
                   "'admin': row[4]}",
        },
        {
            "dimension": "CORRECTNESS",
            "issue": "Connection is never closed",
            "problem": (
                "sqlite3.connect() opens a connection but it is never closed. "
                "This leaks database connections and can exhaust the connection pool."
            ),
            "fix": "Use a context manager:\n"
                   '      with sqlite3.connect("app.db") as conn:\n'
                   "          ...",
        },
        {
            "dimension": "CORRECTNESS",
            "issue": "No handling for user not found",
            "problem": (
                "If no rows are returned (user_id does not exist), the function "
                "returns an empty dict {}. The caller cannot distinguish between "
                "'user not found' and an error."
            ),
            "fix": "if not result:\n"
                   "          return None  # or raise UserNotFoundError(user_id)",
        },
        {
            "dimension": "CORRECTNESS",
            "issue": "Loop overwrites dict on each iteration",
            "problem": (
                "If multiple rows are returned (unlikely for id but possible), "
                "the loop overwrites user dict each time, returning only the last row. "
                "Should use fetchone() since querying by id."
            ),
            "fix": "row = conn.execute(query, (user_id,)).fetchone()\n"
                   "      if row is None: return None\n"
                   "      user = {'id': row[0], ...}",
        },
        {
            "dimension": "PERFORMANCE",
            "issue": "SELECT * retrieves all columns unnecessarily",
            "problem": (
                "SELECT * fetches all columns including password. If the table "
                "has BLOB columns or many fields, this wastes bandwidth."
            ),
            "fix": 'query = "SELECT id, name, email, admin FROM users WHERE id = ?"',
        },
        {
            "dimension": "MAINTAINABILITY",
            "issue": "Hardcoded column indices (row[0], row[1], ...)",
            "problem": (
                "If the table schema changes (columns added/reordered), all indices break. "
                "Fragile and error-prone."
            ),
            "fix": "Use sqlite3.Row or column-name access:\n"
                   "      conn.row_factory = sqlite3.Row\n"
                   "      user = dict(row)  # access by column name",
        },
    ]

    for issue in issues:
        print(f"\n  [{issue['dimension']}] {issue['issue']}")
        print(f"    Problem: {issue['problem']}")
        print(f"    Fix: {issue['fix']}")

    print("\n\n  CORRECTED VERSION:")
    print("""
  import sqlite3

  def get_user_data(user_id):
      with sqlite3.connect("app.db") as conn:
          conn.row_factory = sqlite3.Row
          query = "SELECT id, name, email, admin FROM users WHERE id = ?"
          row = conn.execute(query, (user_id,)).fetchone()
          if row is None:
              return None
          return dict(row)
""")


# =====================================================================
# EXERCISES (End of Lesson)
# =====================================================================

# === Exercise 1: QA vs QC vs Testing ===
# Problem: Classify 6 activities.

def exercise_1():
    """Classify activities as QA, QC, or Testing."""

    print("EXERCISE 1: QA vs QC vs Testing")
    print("=" * 65)

    activities = [
        {
            "activity": "Writing a coding standard requiring all functions to have docstrings",
            "classification": "Quality Assurance (QA)",
            "explanation": (
                "QA is process-oriented and preventive. Writing a standard is defining "
                "a process to prevent quality problems before they occur. No code is "
                "being inspected or executed."
            ),
        },
        {
            "activity": "Running the automated test suite before merging a PR",
            "classification": "Testing (a form of QC)",
            "explanation": (
                "Testing = executing software to find defects. The automated suite "
                "exercises the code to detect regressions. This is also QC because "
                "it inspects a specific work product (the PR)."
            ),
        },
        {
            "activity": "Reviewing a requirements document for completeness and consistency",
            "classification": "Quality Control (QC) — specifically, Static Review/Inspection",
            "explanation": (
                "QC inspects a specific artifact (the requirements doc) to find defects. "
                "This is NOT testing (no code is executed) and NOT QA (it is not defining "
                "a process — it is checking a product)."
            ),
        },
        {
            "activity": "Measuring defect density per KLOC across the last five releases",
            "classification": "QA (Measurement and Process Improvement)",
            "explanation": (
                "Measuring trends across releases is a PROCESS activity aimed at improving "
                "future quality. The data informs process changes (e.g., 'defect density is "
                "rising, we need more code review'). This spans both QA and QC: the "
                "measurement itself is QA; acting on individual findings would be QC."
            ),
        },
        {
            "activity": "Running Bandit to find security issues in Python source before execution",
            "classification": "Quality Control (QC) — Static Analysis",
            "explanation": (
                "Static analysis tools inspect code WITHOUT executing it. This is QC: "
                "inspecting a specific artifact to find defects. It is NOT testing "
                "(which requires execution) though it is sometimes grouped with testing "
                "in CI pipelines."
            ),
        },
        {
            "activity": "Auditing the CI/CD pipeline to confirm security scans run on every commit",
            "classification": "Quality Assurance (QA)",
            "explanation": (
                "Auditing a PROCESS (the CI/CD pipeline) to verify it follows the defined "
                "procedure is QA. The audit checks that the process is in place — not "
                "that any specific code is correct."
            ),
        },
    ]

    for i, a in enumerate(activities, 1):
        print(f"\n  {i}. {a['activity']}")
        print(f"     Classification: {a['classification']}")
        print(f"     Explanation: {a['explanation']}")


# === Exercise 2: Calculate and Interpret Software Metrics ===
# Problem: CC of process_refund(), interpretation, refactoring.

def exercise_2():
    """Calculate Cyclomatic Complexity of process_refund()."""

    print("EXERCISE 2: Cyclomatic Complexity — process_refund()")
    print("=" * 65)

    print("""
  def process_refund(order, reason):
      if order is None or reason is None:          # D1: if, D2: or
          return {"error": "Missing input"}
      if order.status not in ("delivered","shipped"):  # D3
          return {"error": "Order not eligible"}
      if reason == "damaged":                       # D4
          refund_pct = 100
      elif reason == "partial":                     # D5
          if order.days_since_delivery <= 7:        # D6
              refund_pct = 50
          else:
              refund_pct = 25
      elif reason == "wrong_item":                  # D7
          refund_pct = 100
      else:
          return {"error": "Invalid reason"}
      amount = order.total * (refund_pct / 100)
      return {"refund": amount, "pct": refund_pct}
""")

    decisions = [
        "D1: if order is None",
        "D2: or reason is None (compound condition counts as separate decision)",
        "D3: if order.status not in (...)",
        "D4: if reason == 'damaged'",
        "D5: elif reason == 'partial'",
        "D6: if order.days_since_delivery <= 7",
        "D7: elif reason == 'wrong_item'",
    ]

    print("  (a) Decision Points:")
    for d in decisions:
        print(f"      {d}")

    cc = len(decisions) + 1
    print(f"\n      Cyclomatic Complexity = {len(decisions)} decisions + 1 = {cc}")

    print(f"\n  (b) Is CC = {cc} acceptable?")
    print("      CC Risk Table:")
    print("        1-10:  Simple, low risk")
    print("        11-20: Moderate complexity, moderate risk")
    print("        21-50: Complex, high risk")
    print("        > 50:  Untestable")
    print(f"      CC = {cc} falls in the 'simple/low risk' range (1-10).")
    print("      Recommended action: Acceptable. No immediate refactoring needed.")
    print("      However, it is at the upper end of 'simple.' Any additions would push it higher.")

    print("\n  (c) Refactoring Strategy: Strategy Pattern or Lookup Table")
    print("""
  REFACTORED VERSION (Lookup Table):

  REFUND_POLICIES = {
      "damaged": lambda order: 100,
      "wrong_item": lambda order: 100,
      "partial": lambda order: 50 if order.days_since_delivery <= 7 else 25,
  }

  def process_refund(order, reason):
      if order is None or reason is None:
          return {"error": "Missing input"}
      if order.status not in ("delivered", "shipped"):
          return {"error": "Order not eligible for refund"}
      policy = REFUND_POLICIES.get(reason)
      if policy is None:
          return {"error": "Invalid reason"}
      refund_pct = policy(order)
      amount = order.total * (refund_pct / 100)
      return {"refund": amount, "pct": refund_pct}

  New CC = 4 (if/or, if status, if policy is None, compound 'or')
  The reason-matching logic moved to a data structure, eliminating
  the if/elif/elif chain.
""")


# === Exercise 3: Cost of Quality ===
# Problem: Categorize, calculate, and evaluate prevention investment.

def exercise_3():
    """Cost of Quality analysis for a SaaS team."""

    print("EXERCISE 3: Cost of Quality — SaaS Team")
    print("=" * 65)

    costs = [
        ("Automated test infrastructure (CI)", 1200, "Appraisal"),
        ("Weekly code review sessions", 1920, "Appraisal"),
        ("Developer training on secure coding", 800, "Prevention"),
        ("Bug fixing before each release", 9000, "Internal Failure"),
        ("Customer support for production incidents", 14000, "External Failure"),
        ("Emergency hotfixes and rollbacks", 6500, "External Failure"),
    ]

    categories = {"Prevention": 0, "Appraisal": 0,
                  "Internal Failure": 0, "External Failure": 0}

    print("\n  (a) Categorization:")
    for name, cost, cat in costs:
        categories[cat] += cost
        print(f"    ${cost:>6,} — {name:<45} -> {cat}")

    total = sum(categories.values())
    prev_appr = categories["Prevention"] + categories["Appraisal"]
    failure = categories["Internal Failure"] + categories["External Failure"]

    print(f"\n  (b) Total Cost of Quality: ${total:,}/month")
    for cat, amount in categories.items():
        pct = (amount / total) * 100
        print(f"    {cat:<20} ${amount:>7,} ({pct:>5.1f}%)")

    print(f"\n    Prevention + Appraisal: ${prev_appr:,} ({prev_appr/total*100:.1f}%)")
    print(f"    Failure costs:          ${failure:,} ({failure/total*100:.1f}%)")
    print(f"    The team spends {failure/total*100:.0f}% of CoQ on fixing failures —")
    print(f"    a clear signal that more investment in prevention would help.")

    # (c) 1-10-100 analysis
    investment = 3000
    print(f"\n  (c) Break-even Analysis (1-10-100 Rule):")
    print(f"    Additional prevention investment: ${investment:,}/month")
    print(f"    The 1-10-100 rule states: $1 spent on prevention saves $10 in")
    print(f"    internal failure and $100 in external failure.")
    print(f"\n    For ${investment:,} additional prevention:")
    print(f"    - Conservative estimate: $3,000 prevention -> $3,000 minimum")
    print(f"      reduction in failure costs needed to break even.")
    print(f"    - Current failure costs: ${failure:,}/month")
    print(f"    - Break-even requires: ${investment:,} / ${failure:,} = "
          f"{investment/failure*100:.1f}% reduction in failure costs")
    print(f"    - A {investment/failure*100:.1f}% reduction is very achievable with")
    print(f"      better testing tools and linting.")
    print(f"    - Expected reduction (using 1:10 ratio): up to ${investment * 10:,}")
    print(f"      in failure cost savings — far exceeding the investment.")
    print(f"    RECOMMENDATION: Invest. Even a modest 10% failure reduction saves "
          f"${failure * 0.10:,.0f}/month.")


# === Exercise 4: Technical Debt Reduction Plan ===
# Problem: Classify, rank, and plan 4-sprint debt reduction.

def exercise_4():
    """Technical debt reduction plan for legacy payments module."""

    print("EXERCISE 4: Technical Debt Reduction — Payments Module")
    print("=" * 65)

    # (a) Fowler classification
    print("\n  (a) Fowler's Technical Debt Quadrant:")
    items = [
        ("23 functions with CC > 15", "Prudent-Inadvertent",
         "Grew over time; team didn't monitor metrics"),
        ("58% test coverage (target 85%)", "Reckless-Deliberate",
         "Knowingly shipped with low coverage under deadline pressure"),
        ("3 SQL injection vulnerabilities", "Reckless-Inadvertent",
         "Developers lacked security awareness; not a conscious trade-off"),
        ("140 duplicated code blocks", "Prudent-Inadvertent",
         "Accumulated through copy-paste; realized only via static analysis"),
        ("8 deprecated API calls", "Prudent-Deliberate",
         "Knowingly deferred upgrade; still functional"),
    ]
    for item, quadrant, reason in items:
        print(f"\n    {item}")
        print(f"    Quadrant: {quadrant}")
        print(f"    Reason: {reason}")

    # (b) Priority ranking
    print("\n\n  (b) Priority Ranking (20% sprint capacity for debt reduction):")
    ranking = [
        ("1 CRITICAL", "SQL injection (3 vulns)", "Exploitable security risk in a PAYMENTS module"),
        ("2 HIGH", "Test coverage 58% -> 70%+", "Cannot safely refactor without tests"),
        ("3 HIGH", "CC > 15 (23 functions)", "High-CC payment functions are bug-prone"),
        ("4 MEDIUM", "Duplicated code (140 blocks)", "Causes inconsistent bug fixes"),
        ("5 LOW", "Deprecated APIs (8 calls)", "Still functional; upgrade when forced"),
    ]
    for rank, item, justification in ranking:
        print(f"    {rank}: {item}")
        print(f"      Justification: {justification}")

    # (c) 4-sprint plan
    print("\n\n  (c) 4-Sprint Debt Reduction Plan:")
    plan = [
        ("Sprint 1: SECURITY FIRST", [
            "Fix all 3 SQL injection vulnerabilities (parameterize queries)",
            "Add SAST scan to CI pipeline to prevent regression",
            "Write integration tests for the 3 affected code paths",
        ]),
        ("Sprint 2: TEST FOUNDATION", [
            "Write unit tests for top 10 highest-CC functions (payment logic)",
            "Add test coverage tracking to CI (fail build if coverage drops)",
            "Target: 58% -> 65% coverage",
        ]),
        ("Sprint 3: REFACTOR HIGH-CC FUNCTIONS", [
            "Refactor 8 highest-CC functions using Extract Method / Strategy Pattern",
            "Write additional unit tests for refactored functions",
            "Consolidate 30 most-duplicated code blocks",
            "Target: 65% -> 72% coverage, CC < 15 for 8 functions",
        ]),
        ("Sprint 4: CONSOLIDATION", [
            "Refactor 8 more high-CC functions",
            "Consolidate 30 more duplicated blocks",
            "Update 4 of 8 deprecated API calls (prioritize by deprecation urgency)",
            "Target: 72% -> 78% coverage",
        ]),
    ]
    for sprint, tasks in plan:
        print(f"\n    {sprint}")
        for task in tasks:
            print(f"      - {task}")


# === Exercise 5: Pull Request Review ===
# Problem: Review a Flask transfer endpoint — find 6+ issues.

def exercise_5():
    """PR review of a bank transfer endpoint."""

    print("EXERCISE 5: Pull Request Review — /transfer Endpoint")
    print("=" * 65)

    print("""
  ORIGINAL CODE:
  @app.route("/transfer", methods=["POST"])
  def transfer():
      data = request.json
      from_id = data["from_account"]
      to_id = data["to_account"]
      amount = data["amount"]
      conn = db.connect()
      from_bal = conn.execute(
          "SELECT balance FROM accounts WHERE id = " + str(from_id)
      ).fetchone()[0]
      if from_bal >= amount:
          conn.execute(
              "UPDATE accounts SET balance = balance - " + str(amount) +
              " WHERE id = " + str(from_id))
          conn.execute(
              "UPDATE accounts SET balance = balance + " + str(amount) +
              " WHERE id = " + str(to_id))
      return jsonify({"status": "ok"})
""")

    issues = [
        {
            "id": 1,
            "dimension": "SECURITY",
            "issue": "SQL Injection (3 instances)",
            "problem": "All three queries use string concatenation. An attacker can inject SQL.",
            "fix": """Use parameterized queries:
      conn.execute("SELECT balance FROM accounts WHERE id = ?", (from_id,))
      conn.execute("UPDATE accounts SET balance = balance - ? WHERE id = ?", (amount, from_id))""",
        },
        {
            "id": 2,
            "dimension": "CORRECTNESS",
            "issue": "No transaction — race condition",
            "problem": (
                "The two UPDATE statements are not atomic. If the process crashes between "
                "the debit and credit, money disappears. Two concurrent transfers can also "
                "both read the same balance and both succeed (lost update)."
            ),
            "fix": """Wrap in a transaction:
      with conn:  # auto-commit on success, rollback on exception
          conn.execute("UPDATE ... debit ...")
          conn.execute("UPDATE ... credit ...")""",
        },
        {
            "id": 3,
            "dimension": "CORRECTNESS",
            "issue": "Returns 'ok' even when transfer fails",
            "problem": (
                "If from_bal < amount, the function skips the transfer but still "
                "returns {'status': 'ok'}. The client thinks the transfer succeeded."
            ),
            "fix": """Return error for insufficient funds:
      if from_bal < amount:
          return jsonify({"status": "error", "message": "Insufficient funds"}), 400""",
        },
        {
            "id": 4,
            "dimension": "SECURITY",
            "issue": "No input validation",
            "problem": (
                "No validation that amount > 0, from_id != to_id, or that required "
                "fields exist. A negative amount would REVERSE the transfer direction."
            ),
            "fix": """Validate before processing:
      if not all(k in data for k in ("from_account", "to_account", "amount")):
          return jsonify({"error": "Missing fields"}), 400
      if amount <= 0:
          return jsonify({"error": "Amount must be positive"}), 400
      if from_id == to_id:
          return jsonify({"error": "Cannot transfer to same account"}), 400""",
        },
        {
            "id": 5,
            "dimension": "SECURITY",
            "issue": "No authentication or authorization",
            "problem": (
                "Anyone can call POST /transfer and move money between any accounts. "
                "There is no check that the caller owns the from_account."
            ),
            "fix": "Add @login_required decorator and verify request.user.id owns from_account.",
        },
        {
            "id": 6,
            "dimension": "CORRECTNESS",
            "issue": "Connection never closed",
            "problem": "db.connect() opens a connection but it is never closed or returned to pool.",
            "fix": "Use context manager: with db.connect() as conn:",
        },
        {
            "id": 7,
            "dimension": "CORRECTNESS",
            "issue": "No check that accounts exist",
            "problem": (
                "If to_id does not exist, the UPDATE silently succeeds (0 rows affected) — "
                "money is debited but never credited. Lost money."
            ),
            "fix": "Verify both accounts exist before proceeding. Check rows affected by UPDATE.",
        },
        {
            "id": 8,
            "dimension": "PERFORMANCE",
            "issue": "Missing database transaction isolation",
            "problem": (
                "Without proper isolation level (SERIALIZABLE or at least REPEATABLE READ), "
                "concurrent transfers can read stale balances."
            ),
            "fix": "Set appropriate isolation level or use SELECT ... FOR UPDATE to lock rows.",
        },
    ]

    for issue in issues:
        print(f"\n  Issue {issue['id']} [{issue['dimension']}]: {issue['issue']}")
        print(f"    Problem: {issue['problem']}")
        print(f"    Fix: {issue['fix']}")


if __name__ == "__main__":
    print("=" * 65)
    print("=== PRACTICE EXERCISES ===")
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
