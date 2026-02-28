"""
Exercises for Lesson 04: Requirements Engineering
Topic: Software_Engineering

Solutions to practice problems from the lesson.
This lesson has two exercise sections: Practice Exercises (5) and Exercises (5).
"""


# =====================================================================
# PRACTICE EXERCISES (Section 12)
# =====================================================================

# === Practice Exercise 1: Classifying Requirements ===
# Problem: Classify each statement as FR, NFR, or Constraint, and identify defects.

def practice_exercise_1():
    """Classify requirements and identify defects."""

    requirements = [
        {
            "statement": '"The system should be fast."',
            "classification": "NFR (Performance)",
            "defect": (
                "Vague/Ambiguous: 'fast' is not measurable. What operation? "
                "What threshold? For whom?"
            ),
            "rewrite": (
                '"The system shall display search results within 2 seconds '
                'for 95% of queries under normal load (up to 500 concurrent users)."'
            ),
        },
        {
            "statement": '"The application shall authenticate users via OAuth 2.0."',
            "classification": "FR (Functional Requirement)",
            "defect": "None — this is well-formed. It specifies a concrete "
                      "behavior with a verifiable standard (OAuth 2.0).",
            "rewrite": "No rewrite needed. Could add: '...supporting Google "
                       "and GitHub identity providers at launch.'",
        },
        {
            "statement": '"The system shall be implemented using React 18."',
            "classification": "Constraint (Technology Constraint)",
            "defect": "None — this is a legitimate constraint, likely from "
                      "organizational standards or existing infrastructure.",
            "rewrite": "No rewrite needed. This constrains implementation, "
                       "not behavior.",
        },
        {
            "statement": ('"The checkout page shall load in under 1.5 seconds '
                         'for 95% of users on a 4G connection."'),
            "classification": "NFR (Performance)",
            "defect": "None — this is SMART: Specific (checkout page), "
                      "Measurable (1.5s), has conditions (4G, 95th percentile).",
            "rewrite": "No rewrite needed. Excellent requirement.",
        },
        {
            "statement": '"All data shall be encrypted and backed up daily."',
            "classification": "NFR (Security + Reliability) — actually TWO requirements",
            "defect": (
                "Compound/Non-atomic: Contains two independent requirements "
                "(encryption AND backup) in one statement. Also vague on "
                "encryption: at rest? in transit? What algorithm?"
            ),
            "rewrite": (
                "Split into:\n"
                '      NFR-SEC-01: "All data at rest shall be encrypted using AES-256."\n'
                '      NFR-SEC-02: "All data in transit shall use TLS 1.3."\n'
                '      NFR-REL-01: "Full database backups shall be performed daily at '
                '02:00 UTC and retained for 30 days."'
            ),
        },
    ]

    print("PRACTICE EXERCISE 1: Classifying Requirements")
    print("=" * 65)
    for i, req in enumerate(requirements, 1):
        label = chr(96 + i)  # a, b, c, d, e
        print(f"\n  {label}. {req['statement']}")
        print(f"     Classification: {req['classification']}")
        print(f"     Defect: {req['defect']}")
        print(f"     Rewrite: {req['rewrite']}")


# === Practice Exercise 2: Rewriting Bad Requirements ===
# Problem: Rewrite poorly formed requirements to meet SMART/ISO 29148.

def practice_exercise_2():
    """Rewrite bad requirements to meet SMART criteria."""

    rewrites = [
        {
            "original": '"The system shall handle a lot of concurrent users."',
            "defect": "Vague: 'a lot' is not measurable",
            "rewritten": (
                '"The system shall support at least 10,000 concurrent users with '
                'response times not exceeding 500ms at the 95th percentile, as '
                'measured by load testing with Apache JMeter."'
            ),
            "smart_check": "Specific (10K users), Measurable (500ms p95), "
                           "Achievable (realistic for modern architectures), "
                           "Relevant (scalability), Time-bound (verifiable at load test)",
        },
        {
            "original": '"The UI shall be intuitive."',
            "defect": "Subjective: 'intuitive' cannot be measured",
            "rewritten": (
                '"New users shall be able to complete the core workflow (create account, '
                'search product, add to cart, checkout) without external assistance '
                'within 5 minutes, as measured by usability testing with 10 '
                'representative users."'
            ),
            "smart_check": "Specific (core workflow), Measurable (5 min, no assistance), "
                           "verifiable via usability testing",
        },
        {
            "original": '"The system shall use a database to store information."',
            "defect": "Trivially true / Implementation detail masquerading as requirement",
            "rewritten": (
                '"The system shall persist all user data, transaction records, and '
                'configuration settings such that no data is lost in the event of '
                'application restart. Data shall be recoverable to within 1 hour of '
                'any failure (RPO = 1 hour)."'
            ),
            "smart_check": "Focuses on WHAT (data persistence, recovery) rather "
                           "than HOW (database). Measurable via RPO.",
        },
        {
            "original": '"Errors should be handled gracefully."',
            "defect": "Vague: 'gracefully' is subjective; 'should' is ambiguous (use 'shall')",
            "rewritten": (
                '"When an unrecoverable error occurs, the system SHALL: '
                '(1) display a user-friendly error message with a unique error ID, '
                '(2) log the full stack trace to the centralized logging system, '
                '(3) return an appropriate HTTP status code (4xx or 5xx), and '
                '(4) not expose internal implementation details to the user."'
            ),
            "smart_check": "Specific (four concrete behaviors), Measurable "
                           "(each can be verified in testing), uses 'shall'",
        },
    ]

    print("PRACTICE EXERCISE 2: Rewriting Bad Requirements")
    print("=" * 65)
    for i, rw in enumerate(rewrites, 1):
        label = chr(96 + i)
        print(f"\n  {label}. Original: {rw['original']}")
        print(f"     Defect: {rw['defect']}")
        print(f"     Rewritten: {rw['rewritten']}")
        print(f"     SMART Check: {rw['smart_check']}")


# === Practice Exercise 3: User Stories and Acceptance Criteria ===
# Problem: Write 3 user stories for a library system with Gherkin acceptance criteria.

def practice_exercise_3():
    """User stories with Gherkin acceptance criteria for a library system."""

    stories = [
        {
            "role": "Library Member",
            "story": ("As a library member, I want to search for books by title, "
                      "author, or ISBN, so that I can quickly find the book I need."),
            "acceptance_criteria": [
                {
                    "name": "Successful title search",
                    "gherkin": (
                        "Given I am on the library search page\n"
                        "When I enter 'Clean Code' in the search field and select 'Title'\n"
                        "Then I see a list of books with 'Clean Code' in their title\n"
                        "And each result shows the title, author, availability status, and call number"
                    ),
                },
                {
                    "name": "No results found",
                    "gherkin": (
                        "Given I am on the library search page\n"
                        "When I search for 'xyznonexistentbook123'\n"
                        "Then I see a message 'No books found matching your search'\n"
                        "And I am offered suggestions: 'Try a different search term or browse by category'"
                    ),
                },
                {
                    "name": "Empty search field",
                    "gherkin": (
                        "Given I am on the library search page\n"
                        "When I click 'Search' without entering any text\n"
                        "Then I see a validation message 'Please enter a search term'\n"
                        "And no search is performed"
                    ),
                },
            ],
        },
        {
            "role": "Librarian",
            "story": ("As a librarian, I want to check out a book to a member, "
                      "so that the system records the loan and calculates the due date."),
            "acceptance_criteria": [
                {
                    "name": "Successful checkout",
                    "gherkin": (
                        "Given the book 'ISBN 978-0132350884' is available\n"
                        "And the member 'M-1001' has fewer than 5 active loans\n"
                        "When I scan the book's barcode and the member's library card\n"
                        "Then the book status changes to 'Checked Out'\n"
                        "And the due date is set to 14 days from today\n"
                        "And the member receives a confirmation email"
                    ),
                },
                {
                    "name": "Member has reached loan limit",
                    "gherkin": (
                        "Given the member 'M-1002' already has 5 active loans\n"
                        "When I attempt to check out another book to them\n"
                        "Then the system displays 'Loan limit reached (5/5)'\n"
                        "And the checkout is blocked\n"
                        "And I am shown the member's current loans with due dates"
                    ),
                },
            ],
        },
        {
            "role": "Administrator",
            "story": ("As a library administrator, I want to generate a monthly "
                      "overdue report, so that I can identify members with overdue "
                      "books and trigger reminder notifications."),
            "acceptance_criteria": [
                {
                    "name": "Report with overdue items",
                    "gherkin": (
                        "Given there are 15 overdue loans in the system\n"
                        "When I generate the monthly overdue report for January 2025\n"
                        "Then the report lists all 15 overdue items\n"
                        "And each entry shows: member name, book title, due date, days overdue\n"
                        "And the report is sorted by days overdue (descending)"
                    ),
                },
                {
                    "name": "Report generation fails due to database timeout",
                    "gherkin": (
                        "Given the reporting database is experiencing high load\n"
                        "When I generate the monthly overdue report and the query exceeds 30 seconds\n"
                        "Then the system displays 'Report generation timed out. Please try again later.'\n"
                        "And the partial results are NOT displayed\n"
                        "And an alert is sent to the ops team"
                    ),
                },
            ],
        },
    ]

    print("PRACTICE EXERCISE 3: User Stories with Gherkin Acceptance Criteria")
    print("=" * 65)
    for story in stories:
        print(f"\n  ROLE: {story['role']}")
        print(f"  STORY: {story['story']}")
        for ac in story["acceptance_criteria"]:
            print(f"\n    Scenario: {ac['name']}")
            for line in ac["gherkin"].split("\n"):
                print(f"      {line}")


# === Practice Exercise 4: Requirements Traceability Matrix ===
# Problem: Design a minimal RTM for given library system requirements.

def practice_exercise_4():
    """RTM for library system requirements."""

    rtm = [
        {
            "req_id": "FR-001",
            "description": "Users can search books by title, author, or ISBN",
            "source": "Stakeholder interview (Librarian, 2024-01-15)",
            "design_ref": "DD-3.2 Search Module Architecture",
            "impl_module": "search_service.py, book_repository.py",
            "test_cases": ["TC-001: Title search returns matching books",
                           "TC-002: Author search returns matching books",
                           "TC-003: ISBN search returns exact match",
                           "TC-004: Empty search returns validation error"],
        },
        {
            "req_id": "FR-002",
            "description": "Users can place a hold on a checked-out book",
            "source": "Use case UC-05 (Library Member places hold)",
            "design_ref": "DD-4.1 Hold Queue Design",
            "impl_module": "hold_service.py, notification_service.py",
            "test_cases": ["TC-010: Place hold on checked-out book succeeds",
                           "TC-011: Hold when book is available redirects to checkout",
                           "TC-012: Duplicate hold is rejected"],
        },
        {
            "req_id": "NFR-001",
            "description": "Search results within 2s for up to 1,000 results",
            "source": "SLA document (Operations, 2024-01-20)",
            "design_ref": "DD-3.3 Search Index (Elasticsearch)",
            "impl_module": "search_service.py, es_config.py",
            "test_cases": ["TC-020: Search latency < 2s for 1,000 results (load test)",
                           "TC-021: Search latency < 2s at 100 concurrent users"],
        },
    ]

    print("PRACTICE EXERCISE 4: Requirements Traceability Matrix")
    print("=" * 75)

    # Print as formatted table
    header = f"{'Req ID':<10} {'Description':<42} {'Source':<30}"
    print(f"\n  {header}")
    print(f"  {'-' * 82}")
    for row in rtm:
        print(f"\n  {row['req_id']:<10} {row['description'][:42]:<42} {row['source'][:30]}")
        print(f"  {'Design:':<10} {row['design_ref']}")
        print(f"  {'Code:':<10} {row['impl_module']}")
        print(f"  {'Tests:':<10}")
        for tc in row['test_cases']:
            print(f"    - {tc}")

    print("\n\n  KEY OBSERVATIONS:")
    print("  - Every requirement has at least one test case (forward traceability)")
    print("  - NFR-001 requires load testing, not just functional testing")
    print("  - Source column provides audit trail for 'why' each requirement exists")
    print("  - Design references link to specific sections of the design document")


# === Practice Exercise 5: MoSCoW Prioritization ===
# Problem: Prioritize 10 ride-sharing app features using MoSCoW.

def practice_exercise_5():
    """MoSCoW prioritization for ride-sharing app MVP."""

    features = [
        ("Ride booking from current location", "Must Have",
         "Core function — without this, there is no product."),
        ("Fare estimate before booking", "Should Have",
         "Important for trust, but Uber launched without it initially. "
         "Users will accept if pricing is transparent post-ride."),
        ("In-app chat between driver and passenger", "Could Have",
         "Phone calls or SMS work as alternatives. Nice for UX but not essential."),
        ("Driver ratings and reviews", "Should Have",
         "Critical for trust and safety at scale, but not strictly needed for "
         "day-one launch if the driver pool is small and vetted."),
        ("Multi-stop rides", "Won't Have (this time)",
         "Complex routing and pricing logic. Not needed for MVP; address in v2."),
        ("Ride history and receipts", "Must Have",
         "Required for expense reporting, dispute resolution, and legal compliance. "
         "Cannot operate a paid service without receipts."),
        ("Promotional discount codes", "Won't Have (this time)",
         "Marketing feature. Can be added after user acquisition begins."),
        ("Real-time GPS tracking during ride", "Must Have",
         "Essential for safety — both passenger and driver need to know location. "
         "Also required for regulatory compliance in many jurisdictions."),
        ("Scheduled future rides", "Could Have",
         "Useful but not essential for MVP. On-demand is the primary use case."),
        ("Carbon offset tracking per ride", "Won't Have (this time)",
         "Nice differentiator but zero impact on core functionality."),
    ]

    print("PRACTICE EXERCISE 5: MoSCoW Prioritization — Ride-Sharing App MVP")
    print("=" * 75)

    # Group by category
    categories = {"Must Have": [], "Should Have": [], "Could Have": [],
                  "Won't Have (this time)": []}
    for feature, category, justification in features:
        categories[category].append((feature, justification))

    for category, items in categories.items():
        print(f"\n  [{category}]")
        for feature, justification in items:
            print(f"    - {feature}")
            print(f"      Justification: {justification}")

    print("\n  SUMMARY:")
    print("  Must Have (3):   Ride booking, GPS tracking, Ride history/receipts")
    print("  Should Have (2): Fare estimate, Driver ratings")
    print("  Could Have (2):  In-app chat, Scheduled rides")
    print("  Won't Have (3):  Multi-stop, Promo codes, Carbon offsets")


# =====================================================================
# EXERCISES (Section at end of lesson)
# =====================================================================

# === Exercise 1: Classify and Repair Requirements ===
# Problem: Classify, identify defect, and rewrite 5 requirements.

def exercise_1():
    """Classify, find defects, and rewrite requirements to be SMART."""

    items = [
        {
            "statement": '"The system shall be easy to use."',
            "classification": "NFR (Usability)",
            "defect_type": "Vague/Ambiguous — 'easy to use' is subjective and unmeasurable",
            "rewrite": (
                '"A new user shall complete the onboarding flow (account creation '
                'through first successful action) within 3 minutes without consulting '
                'help documentation, as validated by usability testing with 8 '
                'representative users."'
            ),
        },
        {
            "statement": '"All passwords must be secure."',
            "classification": "NFR (Security)",
            "defect_type": "Vague — 'secure' is undefined. No measurable criteria.",
            "rewrite": (
                '"The system shall enforce password complexity: minimum 12 characters, '
                'at least one uppercase letter, one lowercase letter, one digit, and '
                'one special character. Passwords shall be stored using bcrypt with a '
                'work factor of 12. The system shall reject passwords found in the '
                'HaveIBeenPwned top-100,000 list."'
            ),
        },
        {
            "statement": '"The report module shall display and export and archive data."',
            "classification": "FR (Functional Requirement)",
            "defect_type": "Compound/Non-atomic — three distinct behaviors in one requirement",
            "rewrite": (
                "Split into three:\n"
                '      FR-RPT-01: "The report module shall display filtered data in a '
                'tabular format with sortable columns."\n'
                '      FR-RPT-02: "The report module shall export data to CSV and PDF '
                'formats."\n'
                '      FR-RPT-03: "The report module shall archive generated reports '
                'for a minimum of 7 years, accessible via the report history page."'
            ),
        },
        {
            "statement": '"The application shall run on the cloud."',
            "classification": "Constraint (Deployment Constraint)",
            "defect_type": "Vague — which cloud? Public/private/hybrid? Any specific services?",
            "rewrite": (
                '"The application shall be deployed on AWS us-east-1 region using '
                'containerized services (ECS Fargate). All infrastructure shall be '
                'defined as code using Terraform."'
            ),
        },
        {
            "statement": '"Users should be able to log in quickly."',
            "classification": "NFR (Performance) or FR (Authentication)",
            "defect_type": (
                "Two defects: (1) 'should' is ambiguous (use 'shall' for mandatory), "
                "(2) 'quickly' is not measurable."
            ),
            "rewrite": (
                '"The system shall complete user authentication (from credentials '
                'submission to session establishment) within 1 second at the 99th '
                'percentile under normal load."'
            ),
        },
    ]

    print("EXERCISE 1: Classify and Repair Requirements")
    print("=" * 65)
    for i, item in enumerate(items, 1):
        print(f"\n  {i}. {item['statement']}")
        print(f"     (a) Classification: {item['classification']}")
        print(f"     (b) Defect: {item['defect_type']}")
        print(f"     (c) Rewrite: {item['rewrite']}")


# === Exercise 2: Gherkin Acceptance Criteria ===
# Problem: Write 3 Gherkin scenarios for a ride-sharing driver location story.

def exercise_2():
    """Gherkin acceptance criteria for driver location tracking."""

    story = ("As a passenger, I want to view the driver's real-time location on "
             "a map, so that I know how far away they are.")

    scenarios = [
        {
            "name": "Happy path — driver is en route",
            "gherkin": """\
Scenario: Driver location shown while en route
  Given I have booked a ride and the driver "John" has accepted
  And the driver's status is "En Route"
  When I open the ride tracking screen
  Then I see a map centered on my pickup location
  And I see the driver's car icon on the map at their current GPS position
  And I see an ETA label showing estimated arrival time (e.g., "3 min away")
  And the driver's position updates every 5 seconds""",
        },
        {
            "name": "Driver cancels the ride",
            "gherkin": """\
Scenario: Driver cancels while passenger is tracking
  Given I am viewing the driver's real-time location on the map
  And the driver's status is "En Route"
  When the driver cancels the ride
  Then the driver's car icon is removed from the map
  And I see a notification: "Your driver has cancelled. Finding a new driver..."
  And the system automatically begins searching for a replacement driver
  And I am no longer charged for the cancelled ride""",
        },
        {
            "name": "GPS data unavailable",
            "gherkin": """\
Scenario: Driver GPS signal is lost
  Given I am viewing the driver's real-time location on the map
  And the driver's GPS data has not been received for more than 30 seconds
  When the tracking screen refreshes
  Then the driver's car icon shows a grey "stale" indicator
  And a message appears: "Driver location temporarily unavailable"
  And the last known position and timestamp are displayed
  And the system retries fetching GPS data every 10 seconds
  And if GPS data resumes, the icon returns to normal and the message disappears""",
        },
    ]

    print("EXERCISE 2: Gherkin Acceptance Criteria for Driver Location")
    print("=" * 65)
    print(f"\n  User Story: {story}")

    for sc in scenarios:
        print(f"\n  --- {sc['name']} ---")
        for line in sc["gherkin"].split("\n"):
            print(f"    {line}")


# === Exercise 3: Build an RTM ===
# Problem: RTM for a task management API.

def exercise_3():
    """RTM for a task management API."""

    rtm = [
        {
            "req_id": "FR-001",
            "description": "Users can create a task with title, description, due date",
            "source": "Product Owner (Sprint 1 Planning)",
            "design_ref": "API Design Doc §3.1 — POST /tasks endpoint",
            "code_module": "api/tasks.py :: create_task(), models/task.py :: Task",
            "test_cases": [
                "TC-FR001-01: Create task with all fields — returns 201 + task JSON",
                "TC-FR001-02: Create task missing title — returns 400 validation error",
                "TC-FR001-03: Create task with past due date — returns 400 or warning",
            ],
        },
        {
            "req_id": "FR-002",
            "description": "Users can mark a task as complete",
            "source": "Product Owner (Sprint 1 Planning)",
            "design_ref": "API Design Doc §3.2 — PATCH /tasks/{id}/complete",
            "code_module": "api/tasks.py :: complete_task(), models/task.py :: Task.complete()",
            "test_cases": [
                "TC-FR002-01: Mark existing open task as complete — returns 200",
                "TC-FR002-02: Mark already-completed task — returns 409 conflict or idempotent 200",
                "TC-FR002-03: Mark nonexistent task — returns 404",
            ],
        },
        {
            "req_id": "NFR-001",
            "description": "API response within 300ms at 95th percentile (normal load)",
            "source": "SLA document / Technical Lead",
            "design_ref": "Architecture Doc §5 — Performance Requirements",
            "code_module": "middleware/metrics.py (latency tracking), db/indexes.py",
            "test_cases": [
                "TC-NFR001-01: Load test 100 req/s for 10 min — p95 < 300ms",
                "TC-NFR001-02: Load test 500 req/s for 5 min — p95 < 300ms",
            ],
        },
        {
            "req_id": "C-001",
            "description": "System implemented in Python 3.11+",
            "source": "Technical Standards Committee",
            "design_ref": "Architecture Doc §2 — Technology Stack",
            "code_module": "pyproject.toml (requires-python = '>=3.11')",
            "test_cases": [
                "TC-C001-01: CI pipeline runs on Python 3.11 — build succeeds",
                "TC-C001-02: Verify no Python 3.10-incompatible syntax warnings",
            ],
        },
    ]

    print("EXERCISE 3: Requirements Traceability Matrix — Task Management API")
    print("=" * 80)
    print(f"\n  {'Req ID':<10} {'Description':<50} {'Source'}")
    print(f"  {'-' * 80}")

    for row in rtm:
        print(f"\n  {row['req_id']:<10} {row['description'][:50]:<50} {row['source']}")
        print(f"  {'  Design:':<12} {row['design_ref']}")
        print(f"  {'  Code:':<12} {row['code_module']}")
        print(f"  {'  Tests:':<12}")
        for tc in row["test_cases"]:
            print(f"    - {tc}")

    print("\n\n  TRACEABILITY BENEFITS:")
    print("  1. Forward: Every requirement has at least one test → no untested requirements")
    print("  2. Backward: Every test links to a requirement → no purposeless tests")
    print("  3. Impact analysis: If FR-001 changes, we know exactly which code and tests to update")
    print("  4. Coverage: C-001 (constraint) is verified through CI, not manual testing")


# === Exercise 4: MoSCoW + Kano Prioritization ===
# Problem: Classify personal finance app features using both frameworks.

def exercise_4():
    """Prioritize features using MoSCoW and Kano model."""

    features = [
        {
            "feature": "Link a bank account",
            "moscow": "Must Have",
            "kano": "Must-Be (Basic)",
            "analysis": (
                "Without linking an account, the app has no data. Both frameworks agree: "
                "this is a non-negotiable baseline feature."
            ),
        },
        {
            "feature": "View account balance",
            "moscow": "Must Have",
            "kano": "Must-Be (Basic)",
            "analysis": (
                "Core functionality. Users expect to see their balance — its absence "
                "causes dissatisfaction but its presence alone does not delight."
            ),
        },
        {
            "feature": "Categorize transactions automatically",
            "moscow": "Should Have",
            "kano": "Performance (One-Dimensional)",
            "analysis": (
                "Important for value but not strictly needed for MVP. The better "
                "the categorization, the more satisfied users are (linear relationship). "
                "MoSCoW and Kano agree on importance."
            ),
        },
        {
            "feature": "Set a monthly budget and receive alerts",
            "moscow": "Should Have",
            "kano": "Performance (One-Dimensional)",
            "analysis": (
                "Key differentiator from a simple bank app. More accuracy in alerts → "
                "more satisfaction. Not MVP-critical but high priority for v1.1."
            ),
        },
        {
            "feature": "Export transactions to CSV",
            "moscow": "Could Have",
            "kano": "Must-Be (Basic)",
            "analysis": (
                "CONFLICT: Kano says 'Must-Be' (users expect data portability — "
                "its absence causes complaints). MoSCoW says 'Could Have' because "
                "MVP can survive without it. Resolution: Move to 'Should Have' — "
                "Kano's user-expectation insight overrides pure MVP scoping."
            ),
        },
        {
            "feature": "Social sharing of savings goals",
            "moscow": "Won't Have",
            "kano": "Attractive (Delighter)",
            "analysis": (
                "Both agree it is not essential. Kano classifies it as a delighter — "
                "unexpected and pleasing — but MoSCoW correctly defers it. Many "
                "users may actually dislike sharing financial data socially."
            ),
        },
        {
            "feature": "AI-powered spending insights",
            "moscow": "Could Have",
            "kano": "Attractive (Delighter)",
            "analysis": (
                "Both frameworks agree: nice to have, not essential. If done well, "
                "it could be a key differentiator. But poor AI recommendations "
                "could erode trust. Defer until sufficient data is collected."
            ),
        },
        {
            "feature": "Dark mode",
            "moscow": "Could Have",
            "kano": "Must-Be (Basic) — trending toward it",
            "analysis": (
                "CONFLICT: MoSCoW says 'Could Have' (not critical). Kano is shifting: "
                "dark mode was once a delighter but is now expected by many users — "
                "its absence causes complaints on app stores. Resolution: Implement "
                "early (move to 'Should Have') because the cost is low and the "
                "user-satisfaction impact is disproportionately high."
            ),
        },
    ]

    print("EXERCISE 4: MoSCoW + Kano Prioritization — Personal Finance App")
    print("=" * 75)
    print(f"\n  {'#':<3} {'Feature':<35} {'MoSCoW':<15} {'Kano'}")
    print(f"  {'-' * 75}")

    for i, f in enumerate(features, 1):
        print(f"\n  {i:<3} {f['feature']:<35} {f['moscow']:<15} {f['kano']}")
        print(f"      Analysis: {f['analysis']}")

    print("\n\n  FRAMEWORK CONFLICTS AND RESOLUTION:")
    print("  - CSV Export: Kano (Must-Be) > MoSCoW (Could Have) → promote to Should Have")
    print("  - Dark Mode: Kano (trending Must-Be) > MoSCoW (Could Have) → promote to Should Have")
    print("  - Lesson: MoSCoW focuses on MVP scope; Kano focuses on user satisfaction.")
    print("    Using both prevents shipping an MVP that technically works but disappoints users.")


# === Exercise 5: Change Control Process ===
# Problem: Design a change control plan for adding VAT display to an e-commerce system.

def exercise_5():
    """Change control plan for VAT display requirement."""

    plan = """
CHANGE CONTROL PLAN: VAT Display Requirement
=============================================

--- (a) CHANGE REQUEST DOCUMENT FIELDS ---

  CR-ID:          CR-2025-042
  Title:          Display customer country and local VAT rules on purchase receipts
  Requestor:      Legal Department (Jane Smith, Head of Compliance)
  Date Submitted: 2025-01-15
  Priority:       HIGH — Legal/regulatory mandate
  Deadline:       30 days from submission (2025-02-14)
  Description:    All purchase receipts (email, PDF, in-app) must display the
                  customer's registered country and comply with local VAT display
                  rules (rate, amount, registration number where required).
  Affected Modules: Checkout, Receipt Generation, Invoice System
  Business Justification: EU VAT Directive 2006/112/EC compliance; non-compliance
                  risk includes fines up to 10% of annual turnover per jurisdiction.
  Attachments:    Legal requirements document, VAT rule matrix by country

--- (b) CHANGE CONTROL BOARD (CCB) ---

  Members:
  1. Product Owner (Chair) — final decision authority
  2. Technical Lead — assesses implementation complexity and risk
  3. QA Lead — assesses testing scope and timeline
  4. Legal Representative — validates compliance requirements
  5. DevOps Lead — assesses deployment and rollback risk

  Questions the CCB Must Answer:
  1. Is this change necessary? (Yes — legal mandate, not optional)
  2. What is the scope of impact? (3 modules, multiple services)
  3. Can we meet the 30-day deadline with current team capacity?
  4. What is the risk of NOT doing this? (Legal fines, service suspension)
  5. Are there dependencies on external systems (tax APIs, country databases)?
  6. What is the rollback strategy if the deployment fails?
  7. Does this require changes to the database schema? (migration risk)

--- (c) IMPACT ANALYSIS DIMENSIONS ---

  1. TECHNICAL IMPACT:
     - Checkout module: Add country field to order model if not present
     - Receipt module: New VAT calculation engine (rules vary by country)
     - Invoice module: Update invoice template, add VAT registration number
     - Database: May need new tax_rules table; migration required
     - External dependencies: Country-to-VAT-rate lookup (internal DB or API?)

  2. SCHEDULE IMPACT:
     - Estimated effort: 15-20 developer-days
     - With 30-day deadline and 2-dev team: TIGHT but achievable if started immediately
     - Testing: 5 days minimum (multi-country, multi-currency edge cases)
     - Buffer for legal review of output: 3 days

  3. QUALITY IMPACT:
     - High risk: VAT calculation errors could cause legal issues
     - Need country-specific test cases (EU, UK, US states, etc.)
     - Regression risk: receipt changes could break existing formatting

  4. RESOURCE IMPACT:
     - Requires 2 developers dedicated for 3 weeks
     - Legal team must review and sign off on VAT display format
     - QA needs access to a VAT rule reference for validation

  5. COST IMPACT:
     - Direct: ~3 person-weeks of development (~$15K-$25K at loaded cost)
     - Indirect: Other features in Sprint N and N+1 will be delayed
     - Risk of NOT doing it: Fines potentially in millions

--- (d) BASELINE AND ROLLOUT STRATEGY ---

  BASELINE:
  - Current receipt format (v3.2) is baselined in Git tag receipt-v3.2
  - Database schema v42 is baselined before migration
  - All VAT rules are version-controlled in a tax_rules.yaml config file

  ROLLOUT STRATEGY:
  Phase 1 (Days 1-5):   Design + Tax rule database setup
  Phase 2 (Days 6-15):  Implementation (checkout, receipt, invoice modules)
  Phase 3 (Days 16-22): Testing (unit, integration, UAT with legal team)
  Phase 4 (Days 23-25): Staging deployment + Legal sign-off
  Phase 5 (Days 26-28): Production deployment (canary → full)
  Phase 6 (Days 29-30): Monitoring + buffer

  DEPLOYMENT:
  - Feature flag: 'enable_vat_display' — allows toggling without rollback
  - Canary deployment: 5% of traffic for 24 hours, monitor for errors
  - Rollback plan: Disable feature flag (instant), revert to receipt-v3.2
  - Post-deployment: Verify receipts for 5 sample countries (UK, DE, FR, US, AU)
"""
    print(plan)


if __name__ == "__main__":
    print("=" * 65)
    print("=== PRACTICE EXERCISES (Section 12) ===")
    print("=" * 65)

    print("\n" + "=" * 65)
    print("=== Practice Exercise 1: Classifying Requirements ===")
    print("=" * 65)
    practice_exercise_1()

    print("\n" + "=" * 65)
    print("=== Practice Exercise 2: Rewriting Bad Requirements ===")
    print("=" * 65)
    practice_exercise_2()

    print("\n" + "=" * 65)
    print("=== Practice Exercise 3: User Stories with Gherkin ===")
    print("=" * 65)
    practice_exercise_3()

    print("\n" + "=" * 65)
    print("=== Practice Exercise 4: Requirements Traceability Matrix ===")
    print("=" * 65)
    practice_exercise_4()

    print("\n" + "=" * 65)
    print("=== Practice Exercise 5: MoSCoW Prioritization ===")
    print("=" * 65)
    practice_exercise_5()

    print("\n\n" + "=" * 65)
    print("=== EXERCISES (End of Lesson) ===")
    print("=" * 65)

    print("\n" + "=" * 65)
    print("=== Exercise 1: Classify and Repair Requirements ===")
    print("=" * 65)
    exercise_1()

    print("\n" + "=" * 65)
    print("=== Exercise 2: Gherkin Acceptance Criteria ===")
    print("=" * 65)
    exercise_2()

    print("\n" + "=" * 65)
    print("=== Exercise 3: Build an RTM ===")
    print("=" * 65)
    exercise_3()

    print("\n" + "=" * 65)
    print("=== Exercise 4: MoSCoW + Kano Prioritization ===")
    print("=" * 65)
    exercise_4()

    print("\n" + "=" * 65)
    print("=== Exercise 5: Change Control Process ===")
    print("=" * 65)
    exercise_5()

    print("\nAll exercises completed!")
