"""
Exercises for Lesson 08: Verification and Validation
Topic: Software_Engineering

Solutions to practice problems from the lesson.
This lesson has two exercise sections: Practice Exercises (5) and Exercises (5).
Covers test design, coverage analysis, test planning, inspections, and bug reports.
"""


# =====================================================================
# PRACTICE EXERCISES (Section 13)
# =====================================================================

# === Practice Exercise 1: Equivalence Partitioning + Boundary Value ===
# Problem: Password validation — partitions, boundaries, decision table.

def practice_exercise_1():
    """Equivalence partitioning and boundary value analysis for password validation."""

    print("PRACTICE EXERCISE 1: Password Validation — EP + BVA")
    print("=" * 70)

    print("""
  SPECIFICATION:
  - Length: 8-64 characters
  - Must contain >= 1 uppercase letter
  - Must contain >= 1 digit
  - Must contain >= 1 special character from !@#$%^&*()
  - Must NOT contain spaces

  (a) EQUIVALENCE PARTITIONS:

  Rule 1: Length
    EP1.1 (Invalid): length < 8   (e.g., "Ab1@xyz" = 7 chars)
    EP1.2 (Valid):   8 <= length <= 64
    EP1.3 (Invalid): length > 64

  Rule 2: Uppercase
    EP2.1 (Valid):   contains >= 1 uppercase  (e.g., "Abcdef1@")
    EP2.2 (Invalid): contains 0 uppercase     (e.g., "abcdef1@")

  Rule 3: Digit
    EP3.1 (Valid):   contains >= 1 digit      (e.g., "Abcdefg1@")
    EP3.2 (Invalid): contains 0 digits        (e.g., "Abcdefg@!")

  Rule 4: Special character
    EP4.1 (Valid):   contains >= 1 from !@#$%^&*()
    EP4.2 (Invalid): contains 0 special characters

  Rule 5: Spaces
    EP5.1 (Valid):   no spaces
    EP5.2 (Invalid): contains >= 1 space

  (b) BOUNDARY VALUE TEST SET (Length Rule):

  | Test | Length | Input Example                         | Expected |
  |------|--------|---------------------------------------|----------|
  | BV1  | 7      | "Ab1@xyz"                             | REJECT   |
  | BV2  | 8      | "Ab1@xyzw"                            | ACCEPT   |
  | BV3  | 9      | "Ab1@xyzwq"                           | ACCEPT   |
  | BV4  | 63     | "A" + "a"*58 + "1@zz"                 | ACCEPT   |
  | BV5  | 64     | "A" + "a"*59 + "1@zz"                 | ACCEPT   |
  | BV6  | 65     | "A" + "a"*60 + "1@zz"                 | REJECT   |

  (c) DECISION TABLE (Condition Violations):

  | # | Length OK | Has Upper | Has Digit | Has Special | No Space | Result            |
  |---|----------|-----------|-----------|-------------|----------|-------------------|
  | 1 | Y        | Y         | Y         | Y           | Y        | VALID             |
  | 2 | N (short)| Y         | Y         | Y           | Y        | "Too short"       |
  | 3 | N (long) | Y         | Y         | Y           | Y        | "Too long"        |
  | 4 | Y        | N         | Y         | Y           | Y        | "Need uppercase"  |
  | 5 | Y        | Y         | N         | Y           | Y        | "Need digit"      |
  | 6 | Y        | Y         | Y         | N           | Y        | "Need special"    |
  | 7 | Y        | Y         | Y         | Y           | N        | "No spaces"       |
  | 8 | N        | N         | N         | N           | N        | Multiple errors   |
""")


# === Practice Exercise 2: Coverage Analysis ===
# Problem: shipping_cost() — CFG, CC, branch coverage, path coverage.

def practice_exercise_2():
    """Coverage analysis for shipping_cost() function."""

    print("PRACTICE EXERCISE 2: Coverage Analysis — shipping_cost()")
    print("=" * 70)

    print("""
  def shipping_cost(weight_kg, express, country):
      if weight_kg <= 0:            # Branch A (True/False)
          raise ValueError(...)
      base = weight_kg * 2.50
      if express:                    # Branch B (True/False)
          base *= 1.75
      if country == "domestic":      # Branch C
          return base
      elif country == "canada":      # Branch D
          return base * 1.20
      else:                          # Branch E
          return base * 2.00

  CONTROL FLOW GRAPH:
    [Start] --> (A: weight <= 0?)
                  |T--> [raise ValueError] --> [End]
                  |F--> [base = weight * 2.50]
                        --> (B: express?)
                              |T--> [base *= 1.75]
                              |F--> (skip)
                        --> (C: country == "domestic"?)
                              |T--> [return base]
                              |F--> (D: country == "canada"?)
                                      |T--> [return base * 1.20]
                                      |F--> [return base * 2.00]

  (a) CYCLOMATIC COMPLEXITY:
      Decision nodes: A (weight <= 0), B (express), C (domestic), D (canada)
      CC = 4 decisions + 1 = 5
      (Or: CC = Edges - Nodes + 2 = 10 - 7 + 2 = 5)

  (b) MINIMAL TEST SET FOR 100% BRANCH COVERAGE:
      Must exercise every True/False branch:

      | TC | weight | express | country    | Branches Hit | Expected Output    |
      |----|--------|---------|------------|--------------|--------------------|
      | T1 | -1     | False   | "domestic" | A=True       | ValueError         |
      | T2 | 5      | True    | "domestic" | A=F, B=T, C=T| 5*2.5*1.75 = 21.88|
      | T3 | 3      | False   | "canada"   | A=F, B=F, D=T| 3*2.5*1.20 = 9.00 |
      | T4 | 2      | False   | "japan"    | A=F, B=F, E  | 2*2.5*2.00 = 10.00|

      4 test cases achieve 100% branch coverage.

  (c) ALL INDEPENDENT PATHS (Path Coverage):
      Path 1: A=T -> raise ValueError
      Path 2: A=F -> B=T -> C=T -> return base (express + domestic)
      Path 3: A=F -> B=T -> C=F -> D=T -> return base*1.2 (express + canada)
      Path 4: A=F -> B=T -> C=F -> D=F -> return base*2.0 (express + international)
      Path 5: A=F -> B=F -> C=T -> return base (standard + domestic)
      Path 6: A=F -> B=F -> C=F -> D=T -> return base*1.2 (standard + canada)
      Path 7: A=F -> B=F -> C=F -> D=F -> return base*2.0 (standard + international)

      Path coverage requires 7 test cases (vs 4 for branch coverage).
      Path coverage is stronger: it catches interactions between express
      and country that branch coverage might miss.
""")


# === Practice Exercise 3: Test Plan ===
# Problem: Mobile banking v2.0 — biometric auth, P2P, redesigned history.

def practice_exercise_3():
    """Test plan outline for mobile banking v2.0 release."""

    print("PRACTICE EXERCISE 3: Test Plan — Mobile Banking v2.0")
    print("=" * 70)

    plan = """
  TEST PLAN: Mobile Banking App v2.0
  ====================================

  1. SCOPE
     In Scope:
     - Biometric authentication (fingerprint, Face ID) — new feature
     - Peer-to-peer (P2P) payments — new feature
     - Redesigned transaction history screen — UI overhaul
     - Regression testing of existing features (login, transfers, balance)

     Out of Scope:
     - Backend infrastructure changes (covered by separate test plan)
     - Third-party payment gateway internal testing (covered by vendor SLA)
     - Performance testing of database (separate performance test cycle)

  2. TESTING APPROACH
     Unit Testing:
       - All new business logic (biometric verification, P2P validation)
       - Minimum 85% code coverage on new code
     Integration Testing:
       - Biometric SDK integration with authentication service
       - P2P payment flow: mobile app -> API -> payment processor -> notification
     System Testing:
       - End-to-end workflows on real devices (iOS + Android)
       - Cross-feature interaction (e.g., biometric login then P2P payment)
     Acceptance Testing (UAT):
       - Business stakeholders verify P2P limits and flows
       - Compliance team verifies biometric data handling (GDPR, SOC 2)

  3. ENTRY AND EXIT CRITERIA
     Entry Criteria:
     - All unit tests pass (green CI pipeline)
     - Biometric SDK integrated and building successfully
     - Test environment provisioned with mock payment processor
     - Test data (accounts, users) seeded in test environment

     Exit Criteria:
     - All Critical and High severity defects resolved
     - 95% of test cases passed (remaining 5% must have documented waivers)
     - No open security vulnerabilities (SAST + DAST clean)
     - Performance: login < 2s, P2P transfer < 3s (p95)
     - UAT sign-off from product owner and compliance

  4. RISKS AND MITIGATIONS
     Risk 1: Biometric SDK behaves differently across device models
       Mitigation: Test on a matrix of 10+ devices (top models by market share).
       Use BrowserStack/AWS Device Farm for coverage.

     Risk 2: P2P payments require complex state management (pending, completed,
       failed, reversed) — high defect potential
       Mitigation: Create a state transition test matrix covering all possible
       transitions. Add chaos engineering tests (kill payment service mid-transfer).

     Risk 3: Transaction history redesign may introduce regressions in
       existing sorting/filtering functionality
       Mitigation: Create automated regression suite for all existing history
       features BEFORE redesign begins. Run after every PR.
"""
    print(plan)


# === Practice Exercise 4: Fagan Inspection ===
# Problem: 150-line auth module inspection planning.

def practice_exercise_4():
    """Fagan inspection planning for an authentication module."""

    print("PRACTICE EXERCISE 4: Fagan Inspection — Authentication Module")
    print("=" * 70)

    print("""
  (a) TIME ESTIMATES:

  Recommended inspection rate: 100-200 lines/hour (for critical code).
  For a 150-line authentication module (security-critical → use slower rate):

  Preparation Phase:
    - Each of 4 inspectors reads 150 lines at ~125 lines/hour
    - Preparation time: 150 / 125 = ~1.2 hours per inspector
    - Round to 1.5 hours to allow note-taking

  Inspection Meeting:
    - Recommended: 150 lines at ~100 lines/hour (thorough for auth code)
    - Meeting duration: 1.5 hours (hard max: 2 hours to maintain focus)
    - With 4 inspectors + author + moderator = 6 people × 1.5 hrs = 9 person-hours

  (b) INSPECTION CHECKLIST (Authentication-Specific):

  1. [ ] Are all passwords hashed using a strong algorithm (bcrypt, Argon2)?
         Never stored in plaintext or reversible encryption.
  2. [ ] Is the login function resistant to timing attacks?
         (constant-time comparison for password verification)
  3. [ ] Is there rate limiting on login attempts?
         (prevents brute-force attacks)
  4. [ ] Are session tokens generated with a CSPRNG?
         (cryptographically secure random number generator)
  5. [ ] Do sessions have an expiration time and are they invalidated on logout?
  6. [ ] Are error messages generic?
         ("Invalid credentials" not "Invalid password" — prevents user enumeration)
  7. [ ] Is multi-factor authentication (MFA) enforced for sensitive operations?
  8. [ ] Are all SQL queries parameterized?
         (prevent SQL injection in auth queries)
  9. [ ] Is the authentication state stored server-side?
         (not in client-side cookies that can be tampered with)
  10.[ ] Are sensitive values (passwords, tokens) cleared from memory after use?
         (prevent memory dump attacks)

  (c) MODERATOR RESPONSIBILITY WHEN AUTHOR EXPLAINS DECISIONS:

  The moderator MUST intervene and redirect the discussion. In Fagan's
  process, the author's role during the inspection meeting is to EXPLAIN
  THE CODE (walk through it), not to DEFEND design decisions.

  Why this matters:
  - If the author preemptively explains "why I did it this way," inspectors
    subconsciously shift from "finding defects" to "evaluating design choices."
  - This creates an adversarial dynamic where inspectors feel they need to
    challenge the author's reasoning rather than inspect the code.
  - Time is wasted on design discussions instead of defect detection.

  The moderator should say:
  "Thank you, but let's hold design discussions for after the meeting.
  Right now, let's focus on walking through the code and logging any
  issues we see. We can discuss design alternatives in the rework phase."
""")


# === Practice Exercise 5: Bug Report ===
# Problem: Free items ($0.00) sorted to bottom instead of top.

def practice_exercise_5():
    """Professional bug report for a sorting defect."""

    print("PRACTICE EXERCISE 5: Bug Report — Sorting Defect")
    print("=" * 70)

    report = """
  BUG REPORT
  ===========

  ID:          BUG-2025-0387
  Title:       Products priced at $0.00 appear at bottom when sorting
               "Price: Low to High" instead of at the top

  Reporter:    QA Engineer (Jane Doe)
  Date:        2025-02-15
  Environment: Production (v3.4.2), Chrome 121, macOS 14.3
  Component:   Product Listing / Sorting

  SEVERITY:    Medium
    Justification: Sorting is functionally incorrect for a subset of products.
    Users see wrong ordering, which undermines trust in the product catalog.
    However, no data loss or security impact.

  PRIORITY:    Medium-High
    Justification: Affects all users who sort by price (common action). Free
    products being "hidden" at the bottom means promotional items lose
    visibility, directly impacting business goals. Should be fixed in the
    next sprint.

  STEPS TO REPRODUCE:
  1. Navigate to the product listing page (/products).
  2. Ensure at least 3 products are priced at $0.00 (free items).
  3. Click the "Sort by" dropdown.
  4. Select "Price: Low to High."
  5. Observe the order of products in the list.

  EXPECTED BEHAVIOR:
  Products priced at $0.00 should appear FIRST in the list (lowest price),
  followed by products in ascending price order.

  ACTUAL BEHAVIOR:
  Products priced at $0.00 appear at the BOTTOM of the list, after all
  priced products. The remaining products are correctly sorted in ascending
  order.

  ADDITIONAL DATA:
  - Sorting "Price: High to Low" correctly places $0.00 items at the bottom.
  - Products priced at $0.01 sort correctly.
  - The issue occurs on all browsers tested (Chrome, Firefox, Safari).
  - The issue does NOT occur when using the mobile app (API returns correct order).

  HYPOTHESIS (Root Cause):
  The sorting algorithm likely treats $0.00 as a special value (null, empty,
  or falsy). In many languages, `0` is falsy, and a sort comparator that
  checks `if price:` before comparing will push zero-priced items to the
  end. Alternatively, the database query may use `ORDER BY NULLIF(price, 0)`
  or a similar construct that treats zero as null, sorting it last.

  Likely fix: Ensure the sort comparator uses strict numeric comparison
  (`price_a < price_b`) rather than truthiness checks.

  ATTACHMENTS:
  - screenshot_sort_low_to_high.png (showing $0.00 items at bottom)
  - network_tab.har (API response shows correct order → frontend sorting bug)
"""
    print(report)


# =====================================================================
# EXERCISES (End of Lesson)
# =====================================================================

# === Exercise 1: Black-Box Test Cases ===
# Problem: Discount calculation — EP + BVA test set.

def exercise_1():
    """Black-box test cases for discount calculation."""

    print("EXERCISE 1: Black-Box Test Cases — Discount Calculation")
    print("=" * 70)

    print("""
  SPECIFICATION:
  - cart_total (float, > 0), coupon_code (string, optional)
  - Valid coupons: "SAVE10" (10%), "SAVE20" (20%), "FREESHIP" (no price change)
  - If cart_total < 10.00, no discount applied even with valid coupon
  - If coupon_code is None or empty, no discount
  - Invalid coupon codes return error

  EQUIVALENCE PARTITIONS:
  Cart total:
    EP1: cart_total <= 0 (invalid)
    EP2: 0 < cart_total < 10.00 (valid but no discount)
    EP3: cart_total >= 10.00 (valid, discount eligible)

  Coupon:
    EP4: coupon_code is None
    EP5: coupon_code is "" (empty string)
    EP6: coupon_code = "SAVE10"
    EP7: coupon_code = "SAVE20"
    EP8: coupon_code = "FREESHIP"
    EP9: coupon_code is invalid (e.g., "BOGUS")

  MINIMAL TEST SET:

  | # | cart_total | coupon   | Partition      | Expected Output               |
  |---|-----------|----------|----------------|-------------------------------|
  | 1 |    -5.00  | None     | EP1            | Error: cart_total must be > 0  |
  | 2 |     0.00  | None     | EP1 boundary   | Error: cart_total must be > 0  |
  | 3 |     0.01  | None     | EP2 boundary   | No discount. Total = $0.01    |
  | 4 |     9.99  | "SAVE10" | EP2+EP6 (< 10) | No discount. Total = $9.99    |
  | 5 |    10.00  | "SAVE10" | EP3+EP6 bndry  | 10% off. Total = $9.00        |
  | 6 |    10.01  | "SAVE10" | EP3+EP6        | 10% off. Total = $9.01        |
  | 7 |    50.00  | "SAVE20" | EP3+EP7        | 20% off. Total = $40.00       |
  | 8 |    50.00  | "FREESHIP"| EP3+EP8       | Total = $50.00, freeship=True  |
  | 9 |    50.00  | None     | EP3+EP4        | No discount. Total = $50.00   |
  |10 |    50.00  | ""       | EP3+EP5        | No discount. Total = $50.00   |
  |11 |    50.00  | "BOGUS"  | EP3+EP9        | Error: Invalid coupon code     |
  |12 |     9.99  | "BOGUS"  | EP2+EP9        | Error: Invalid coupon code     |

  Boundary values tested: 0, 0.01, 9.99, 10.00, 10.01
  12 test cases cover all partitions and critical boundaries.
""")


# === Exercise 2: Branch Coverage ===
# Problem: classify_bmi() — branches, minimal test set, path analysis.

def exercise_2():
    """Branch coverage analysis for classify_bmi()."""

    print("EXERCISE 2: Branch Coverage — classify_bmi()")
    print("=" * 70)

    print("""
  def classify_bmi(weight_kg, height_m):
      if height_m <= 0 or weight_kg <= 0:     # Branch 1T/1F (compound: 2 sub-branches)
          raise ValueError(...)
      bmi = weight_kg / (height_m ** 2)
      if bmi < 18.5:                           # Branch 2T/2F
          return "underweight"
      elif bmi < 25.0:                         # Branch 3T/3F
          return "normal"
      elif bmi < 30.0:                         # Branch 4T/4F
          return "overweight"
      else:                                    # Branch 5 (else)
          return "obese"

  (a) BRANCHES (8 branch outcomes):
      B1T: height_m <= 0 is True   -> raise ValueError
      B1F: height_m <= 0 is False  -> check weight
      B1a: weight_kg <= 0 is True  -> raise ValueError (short-circuit)
      B1b: weight_kg <= 0 is False -> continue
      B2T: bmi < 18.5             -> "underweight"
      B2F: bmi >= 18.5            -> check next
      B3T: bmi < 25.0             -> "normal"
      B3F: bmi >= 25.0            -> check next
      B4T: bmi < 30.0             -> "overweight"
      B4F: bmi >= 30.0            -> "obese"

      Total: ~10 branch outcomes (including compound condition sub-branches)

  (b) MINIMUM TEST CASES FOR 100% BRANCH COVERAGE:

  | TC | weight_kg | height_m | BMI     | Branches Hit     | Result        |
  |----|-----------|----------|---------|------------------|---------------|
  | T1 | 60        | -1.0     | N/A     | B1T (h<=0)       | ValueError    |
  | T2 | -5        | 1.70     | N/A     | B1F, B1a (w<=0)  | ValueError    |
  | T3 | 50        | 1.80     | 15.43   | B1F, B1b, B2T    | "underweight" |
  | T4 | 70        | 1.75     | 22.86   | B1F, B1b, B2F,B3T| "normal"      |
  | T5 | 85        | 1.75     | 27.76   | B2F, B3F, B4T    | "overweight"  |
  | T6 | 120       | 1.75     | 39.18   | B2F, B3F, B4F    | "obese"       |

  Minimum: 5 test cases (T1+T3+T4+T5+T6) for basic branch coverage.
  6 test cases if we want to cover both sub-branches of the compound condition.

  (c) DOES THIS ACHIEVE 100% PATH COVERAGE?
      NO. Branch coverage does not equal path coverage.

      Paths through the function:
      P1: ValueError (height <= 0)
      P2: ValueError (weight <= 0, height > 0)
      P3: underweight
      P4: normal
      P5: overweight
      P6: obese

      In this case, the number of paths equals the number of branches
      because there are no loops and conditions are sequential (not nested
      in complex ways). So branch coverage HAPPENS to equal path coverage
      here. But this is a coincidence of the simple control flow — in
      general, path coverage requires exponentially more test cases than
      branch coverage (due to combinatorial explosion of loop iterations
      and nested conditions).
""")


# === Exercise 3: Integration Testing ===
# Problem: Microservices order system — dependency graph, bottom-up plan.

def exercise_3():
    """Integration test plan for microservices order system."""

    print("EXERCISE 3: Integration Testing — Order System")
    print("=" * 70)

    print("""
  (a) INTEGRATION DEPENDENCY GRAPH:

      OrderService
        |-------> InventoryService
        |-------> PaymentService
        |-------> NotificationService (after successful payment)

      NotificationService is called BY OrderService (not the reverse).
      InventoryService and PaymentService are independent of each other.

  (b) INCREMENTAL BOTTOM-UP INTEGRATION TEST PLAN:

  Phase 1: Test Leaf Services Independently
    Services: InventoryService, PaymentService, NotificationService
    Stubs/Drivers: Test driver for each service (simulates OrderService calls)
    Tests:
      - InventoryService: check stock, reserve stock, release stock
      - PaymentService: authorize, capture, refund
      - NotificationService: send email, send SMS, handle delivery failures
    Interface contracts verified:
      - Each service's API contract (request/response schemas)
      - Error responses (400, 404, 500) follow consistent format

  Phase 2: Integrate OrderService with InventoryService + PaymentService
    Services: OrderService + InventoryService + PaymentService
    Stubs: NotificationService is STUBBED (returns success)
    Tests:
      - Place order: inventory reserved, payment authorized, order created
      - Place order with insufficient stock: payment NOT attempted, error returned
      - Place order with payment failure: inventory reservation RELEASED
      - Concurrent orders for last item: one succeeds, one fails gracefully
    Interface contracts verified:
      - OrderService correctly interprets inventory check responses
      - OrderService handles payment timeout (retry vs fail)
      - Inventory reservation + payment are consistent (no orphaned reservations)

  Phase 3: Full Integration (Add NotificationService)
    Services: All four services
    Stubs: None
    Tests:
      - Successful order: customer receives confirmation email
      - Failed order: customer receives failure notification
      - NotificationService down: order still completes (notification is async/non-blocking)
    Interface contracts verified:
      - Notification payload contains correct order details
      - Notification failure does not block order completion

  (c) THREE INTEGRATION-SPECIFIC FAILURE MODES:

  1. Distributed Transaction Inconsistency:
     OrderService deducts inventory AND charges payment. If payment succeeds
     but inventory reservation fails due to a race condition, the customer
     is charged but the order cannot be fulfilled. This can only be caught
     at integration level — each service works correctly individually.

  2. Timeout Cascade:
     PaymentService takes 10 seconds to respond (external gateway slow).
     OrderService has a 5-second timeout. OrderService returns error to user.
     PaymentService eventually charges the customer anyway. The payment
     succeeded but OrderService thinks it failed. Unit tests of each service
     pass; the bug only appears when real network delays exist.

  3. Schema Mismatch:
     InventoryService expects {"product_id": "ABC-123"} but OrderService
     sends {"productId": "ABC-123"} (camelCase vs snake_case). Unit tests
     with mocked responses pass because the mock matches the caller's
     expectation. Real integration fails with a 400 Bad Request.
""")


# === Exercise 4: Test Plan for Biometric Login ===
# Problem: Mobile banking biometric login test plan outline.

def exercise_4():
    """Test plan for biometric login feature."""

    print("EXERCISE 4: Test Plan — Biometric Login")
    print("=" * 70)

    plan = """
  TEST PLAN: Biometric Login (Fingerprint + Face ID)
  ====================================================

  SCOPE:
    In Scope:
    - Fingerprint enrollment and authentication flow
    - Face ID enrollment and authentication flow
    - Fallback to PIN when biometric fails
    - Biometric settings management (enable/disable per method)
    - Biometric token storage and refresh

    Out of Scope:
    - Biometric hardware testing (vendor responsibility)
    - Server-side authentication API (separate test plan)
    - Non-biometric login flows (existing, already tested)

  TEST LEVELS:
    Unit Testing:
    - Biometric token generation and validation logic
    - Timeout and retry logic
    - Input validation (biometric prompt responses)

    Integration Testing:
    - Mobile biometric SDK ↔ authentication service API
    - Biometric token ↔ session management service
    - Fallback path: biometric failure → PIN prompt → auth service

    System Testing:
    - End-to-end login on physical devices (real fingerprint + face)
    - Login under various conditions (wet fingers, glasses, low light)
    - Cross-feature: biometric login → immediate P2P payment

    Acceptance Testing:
    - Business: UX meets design specifications
    - Compliance: biometric data handling meets GDPR/SOC 2

  TESTING TYPES (Non-Functional):
    1. Security Testing:
       - Biometric spoofing resistance (silicone fingerprint, photo)
       - Biometric data encryption at rest and in transit
       - Token not extractable from device keychain
       Why: Financial app; biometric bypass = account takeover

    2. Performance Testing:
       - Biometric authentication < 2 seconds (p95)
       - No UI freeze during biometric prompt
       Why: Slow biometric = users abandon and revert to PIN

    3. Usability Testing:
       - Tested with 10+ users across age groups
       - Accessibility: VoiceOver/TalkBack support during biometric prompts
       Why: Biometric must be easier than PIN, or adoption will be low

    4. Compatibility Testing:
       - iOS: iPhone 8+ (Touch ID), iPhone X+ (Face ID)
       - Android: fingerprint API level 23+, face unlock level 29+
       - Test on at least 10 device models
       Why: Biometric APIs vary significantly across devices

  ENTRY CRITERIA:
    - Biometric SDK version finalized and documented
    - Test devices procured (minimum 10 models)
    - Test accounts with various biometric configurations created
    - CI pipeline builds and deploys to test devices successfully

  EXIT CRITERIA:
    - 100% of Critical test cases pass
    - 95% of all test cases pass
    - Zero open Critical or High severity defects
    - Biometric auth latency < 2s (p95) on all target devices
    - Security penetration test report shows no biometric bypass vulnerabilities
    - Compliance team sign-off on biometric data handling

  TOP 3 RISKS AND MITIGATIONS:
    1. Risk: Device-specific biometric API differences cause crashes
       Mitigation: Maintain a device compatibility matrix. Test on each
       device in CI using cloud device farms. Have a feature flag to
       disable biometric on specific device models if needed.

    2. Risk: Biometric SDK update changes API behavior mid-testing
       Mitigation: Pin SDK version in build. Test with pinned version.
       Only upgrade SDK in a separate, dedicated test cycle.

    3. Risk: Users cannot enroll biometrics due to unclear UX
       Mitigation: Conduct usability testing in Sprint 1 with paper
       prototypes. Iterate on enrollment flow before development.
"""
    print(plan)


# === Exercise 5: Bug Lifecycle Analysis ===
# Problem: 10,000-row PDF export causes 504 timeout.

def exercise_5():
    """Bug report and analysis for PDF export timeout."""

    print("EXERCISE 5: Bug Report + Root Cause Analysis — PDF Export Timeout")
    print("=" * 70)

    print("""
  (a) BUG REPORT:

  ID:          BUG-2025-0412
  Title:       Exporting 10,000-row report to PDF returns 504 Gateway Timeout
               after 32 seconds

  Reporter:    QA Engineer
  Date:        2025-02-20
  Environment: Staging (v4.1.0-rc2), Chrome 122, Ubuntu 22.04
  Component:   Reporting / Export

  SEVERITY:    High
    Justification: Data loss — the user's report is not generated. For large
    reports (10K+ rows), the feature is completely non-functional. This is
    not a crash, but the feature fails to deliver its core purpose.

  PRIORITY:    High
    Justification: Report export is a core feature used by finance teams
    for monthly reconciliation. The deadline for March reports is approaching.
    Enterprise customers with large datasets are blocked.

  STEPS TO REPRODUCE:
  1. Log in as user with access to the "Monthly Transactions" report.
  2. Set date range to generate ~10,000 rows (e.g., 2024-01-01 to 2024-12-31).
  3. Click "Export to PDF."
  4. Wait for the export to complete.

  EXPECTED BEHAVIOR:
  The system generates a PDF file and offers it for download. For large
  reports, a progress indicator or background job notification is shown.

  ACTUAL BEHAVIOR:
  After 32 seconds, the browser displays a "504 Gateway Timeout" error.
  No PDF file is generated. No error is logged in the application logs
  (only the nginx access log shows the 504).

  ADDITIONAL DATA:
  - Export works correctly for reports with < 5,000 rows (~15 seconds).
  - Export to CSV (same 10K rows) completes in 3 seconds.
  - The 504 comes from the nginx reverse proxy (default timeout: 30s).
  - Server CPU spikes to 100% during the export attempt.

  (b) SEVERITY vs PRIORITY:
  Severity: HIGH — Feature is broken for a significant use case (large reports).
  Priority: HIGH — Business-critical feature with an upcoming deadline.
  These are INDEPENDENT ratings:
  - A cosmetic bug (Low severity) on the login page might be High priority
    (first impression for all users).
  - A crash in an admin tool used once/year (High severity) might be
    Low priority (rare usage).

  (c) ROOT CAUSE HYPOTHESES:

  Hypothesis 1 (Application Layer):
    The PDF generation library renders all 10,000 rows synchronously in
    a single HTTP request thread. PDF rendering is CPU-intensive (layout
    calculation, font rendering, page breaking). 10K rows exceeds the
    time budget for a synchronous HTTP response. The fix: move PDF
    generation to an async background job (Celery/RQ) and return a
    "your report is being generated" response with a download link.

  Hypothesis 2 (Infrastructure Layer):
    The nginx reverse proxy has a 30-second proxy_read_timeout. Even if
    the application could generate the PDF in 45 seconds, nginx kills
    the connection at 30s. The fix: increase nginx timeout (short-term)
    AND implement async generation (long-term). Simply increasing the
    timeout is not sustainable as data grows.

  (d) VERIFICATION STEPS AFTER FIX:

  1. Reproduce the original bug: export 10K-row report to PDF.
     Verify: PDF is generated successfully (not a 504).
  2. Verify file integrity: open the PDF and confirm all 10,000 rows
     are present and correctly formatted.
  3. Regression test: export reports of various sizes (100, 1K, 5K, 10K,
     50K rows) and verify all succeed.
  4. Performance test: measure export time for 10K rows. Verify it
     completes within acceptable time (e.g., < 60s or background job).
  5. Edge case: export an empty report (0 rows). Verify it generates
     a valid PDF with headers but no data rows.
  6. Concurrent test: trigger 5 simultaneous PDF exports and verify
     all complete without server crashes or timeouts.
  7. Monitor: check application logs for any new errors or warnings
     during the export process.
""")


if __name__ == "__main__":
    print("=" * 70)
    print("=== PRACTICE EXERCISES ===")
    print("=" * 70)

    for i, func in enumerate([practice_exercise_1, practice_exercise_2,
                               practice_exercise_3, practice_exercise_4,
                               practice_exercise_5], 1):
        print(f"\n{'=' * 70}")
        print(f"=== Practice Exercise {i} ===")
        print("=" * 70)
        func()

    print("\n\n" + "=" * 70)
    print("=== EXERCISES (End of Lesson) ===")
    print("=" * 70)

    for i, func in enumerate([exercise_1, exercise_2, exercise_3,
                               exercise_4, exercise_5], 1):
        print(f"\n{'=' * 70}")
        print(f"=== Exercise {i} ===")
        print("=" * 70)
        func()

    print("\nAll exercises completed!")
