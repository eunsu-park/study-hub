"""
Exercises for Lesson 16: Developer Practices and Ethics
Topic: Programming

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Identify Technical Debt ===
# Problem: Identify deliberate and accidental technical debt.

def exercise_1():
    """Solution: Classify and prioritize technical debt examples."""

    print("  Example Project: E-commerce web application")
    print()

    debt_examples = {
        "Deliberate, Prudent Technical Debt": {
            "example": (
                "We chose to use SQLite instead of PostgreSQL for the MVP. "
                "We know it won't scale past 100 concurrent users, but we need "
                "to ship in 2 weeks to validate the business model. Migration "
                "plan is documented in ADR-003."
            ),
            "characteristics": [
                "Team is AWARE it's suboptimal",
                "Decision was conscious and documented",
                "There's a plan to address it later",
                "The trade-off was worth it (speed to market)",
            ],
        },
        "Accidental Technical Debt": {
            "example": (
                "The authentication module grew organically over 18 months. "
                "Password hashing, JWT generation, OAuth, RBAC, and session "
                "management are all in a single 2000-line auth.py file. "
                "Nobody planned this; each feature just got added to the existing file."
            ),
            "characteristics": [
                "Team wasn't aware debt was accumulating",
                "Happened gradually (boiling frog problem)",
                "No documentation of the coupling",
                "Now it's risky to change (everything depends on it)",
            ],
        },
        "Prioritization": {
            "address_first": "Accidental (auth.py monolith)",
            "reasoning": [
                "Security-critical code that's hard to audit in its current form",
                "Every new auth feature increases risk of introducing bugs",
                "High coupling means changes ripple unpredictably",
                "Refactoring now prevents compounding debt",
            ],
            "address_second": "Deliberate (SQLite)",
            "reasoning_second": [
                "Only matters when we approach 100 concurrent users",
                "Migration path is already documented",
                "Can be planned and scheduled proactively",
            ],
        },
    }

    for category, info in debt_examples.items():
        print(f"  {category}:")
        if "example" in info:
            print(f"    Example: {info['example']}")
            for char in info["characteristics"]:
                print(f"      - {char}")
        else:
            print(f"    Address first: {info['address_first']}")
            for r in info["reasoning"]:
                print(f"      - {r}")
            print(f"    Address second: {info['address_second']}")
            for r in info["reasoning_second"]:
                print(f"      - {r}")
        print()


# === Exercise 2: Write a README ===
# Problem: Write a README for a CLI TODO list tool.

def exercise_2():
    """Solution: Well-structured README for a CLI tool."""

    readme = """
  # todo-cli

  A fast, lightweight command-line tool for managing TODO lists.
  Store tasks locally, organize with tags, and never forget what
  needs doing.

  ## Features

  - Add, complete, and delete tasks from the terminal
  - Tag tasks for organization (#work, #personal, #urgent)
  - Priority levels (low, medium, high)
  - Filter and search across all tasks
  - Data stored locally in ~/.todo/tasks.json

  ## Installation

  ### From PyPI (recommended)
  ```bash
  pip install todo-cli
  ```

  ### From source
  ```bash
  git clone https://github.com/user/todo-cli.git
  cd todo-cli
  pip install -e .
  ```

  ## Usage

  ```bash
  # Add a task
  todo add "Write unit tests" --tag work --priority high

  # List all tasks
  todo list

  # List filtered tasks
  todo list --tag work --status pending

  # Complete a task
  todo done 3

  # Delete a task
  todo delete 3

  # Search tasks
  todo search "unit tests"
  ```

  ## Configuration

  Configuration file: `~/.todo/config.yaml`
  ```yaml
  default_priority: medium
  date_format: "%Y-%m-%d"
  color_output: true
  ```

  ## Contributing

  1. Fork the repository
  2. Create a feature branch (`git checkout -b feature/amazing-feature`)
  3. Write tests for your changes
  4. Ensure all tests pass (`pytest`)
  5. Submit a pull request

  Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

  ## License

  MIT License - see [LICENSE](LICENSE) for details.
"""

    print(readme)


# === Exercise 3: Choose a License ===
# Problem: Choose a license for an open-source ML library.

def exercise_3():
    """Solution: License selection for an ML library."""

    print("  Requirements:")
    print("    - Anyone can use it freely")
    print("    - Derivatives must also be open source")
    print("    - Protection against patent lawsuits")
    print()

    licenses = {
        "Apache 2.0 (RECOMMENDED)": {
            "meets_requirements": True,
            "free_use": "Yes - permissive, anyone can use, modify, distribute",
            "copyleft": "No - derivatives CAN be proprietary (doesn't meet req #2)",
            "patent_protection": "Yes - explicit patent grant and retaliation clause",
            "note": "Does NOT require derivatives to be open source",
        },
        "GPL v3 (BEST FIT)": {
            "meets_requirements": True,
            "free_use": "Yes - anyone can use freely",
            "copyleft": "Yes - strong copyleft: derivatives MUST be open source under GPL",
            "patent_protection": "Yes - includes patent protection (Section 11)",
            "note": "Some companies avoid GPL due to copyleft obligations",
        },
        "LGPL v3 (COMPROMISE)": {
            "meets_requirements": True,
            "free_use": "Yes - anyone can use freely",
            "copyleft": "Partial - modifications to the library must be open source, "
                        "but applications using it as a library don't need to be",
            "patent_protection": "Yes - same patent protection as GPL v3",
            "note": "Good middle ground: protects the library while allowing proprietary apps",
        },
        "MIT": {
            "meets_requirements": False,
            "free_use": "Yes",
            "copyleft": "No - extremely permissive, no copyleft",
            "patent_protection": "No - no explicit patent grant",
            "note": "Doesn't meet requirements 2 or 3",
        },
    }

    print("  License Analysis:")
    for name, info in licenses.items():
        meets = "MEETS ALL" if info["meets_requirements"] else "INCOMPLETE"
        print(f"\n  {name} [{meets}]:")
        print(f"    Free use: {info['free_use']}")
        print(f"    Copyleft (derivatives open source): {info['copyleft']}")
        print(f"    Patent protection: {info['patent_protection']}")
        print(f"    Note: {info['note']}")

    print("\n  RECOMMENDATION: GPL v3")
    print("    It's the only license that satisfies ALL three requirements:")
    print("    free use + mandatory open source derivatives + patent protection.")
    print("    If adoption by corporations matters, LGPL v3 is the compromise.")


# === Exercise 4: Ethical Scenario Analysis ===
# Problem: Analyze ethical issues in credit scoring with ZIP codes.

def exercise_4():
    """Solution: Ethical analysis of biased credit scoring algorithm."""

    print("  Scenario: Credit scoring model uses ZIP code as a feature.")
    print("  ZIP code is highly correlated with race; model gives lower")
    print("  scores to predominantly Black neighborhoods.")
    print()

    analysis = {
        "1. What is the ethical issue?": [
            "Proxy discrimination: ZIP code acts as a proxy for race,",
            "creating disparate impact even without explicitly using race.",
            "This is a form of algorithmic bias that perpetuates historical",
            "patterns of residential segregation (redlining).",
            "The model learned societal bias from historical data.",
        ],
        "2. What are potential harms?": [
            "Denial of credit to qualified individuals based on where they live",
            "Perpetuation of wealth inequality across racial lines",
            "Reinforcement of historical redlining patterns",
            "Legal liability: violates Fair Housing Act and ECOA",
            "Loss of public trust in algorithmic decision-making",
            "Self-fulfilling prophecy: denied credit -> lower property values -> lower scores",
        ],
        "3. What would you do?": [
            "IMMEDIATE: Remove ZIP code as a direct feature",
            "INVESTIGATE: Test for disparate impact across protected groups",
            "  - Calculate approval rates by race, gender, age",
            "  - Use the 80% rule (4/5ths rule) as a baseline check",
            "MITIGATE: Apply fairness constraints during model training",
            "  - Use techniques like adversarial debiasing or equalized odds",
            "  - Replace ZIP code with non-discriminatory features (income, debt-to-income)",
            "AUDIT: Implement ongoing fairness monitoring",
            "  - Regular bias audits by independent reviewers",
            "  - Publish model fairness metrics",
            "ESCALATE: Report findings to management and compliance",
            "  - Document the issue and your recommendation",
            "  - If company refuses to act, consider external reporting",
            "PREVENT: Establish a model ethics review process",
            "  - Require bias impact assessment before deploying any model",
        ],
    }

    for question, points in analysis.items():
        print(f"  {question}")
        for point in points:
            print(f"    {point}")
        print()


# === Exercise 5: Career Reflection ===
# Problem: Reflect on career direction (IC vs management, T-shaped skills).

def exercise_5():
    """Solution: Framework for career self-assessment."""

    print("  Career Reflection Framework")
    print("  (Fill in your own answers using this structure)")
    print()

    framework = {
        "1. IC Track vs Management Track": {
            "IC (Individual Contributor)": [
                "Strengths: deep technical skills, architecture decisions, mentoring",
                "Growth path: Senior -> Staff -> Principal -> Distinguished Engineer",
                "You enjoy: solving hard problems, writing code, technical excellence",
                "You prefer: depth over breadth, making things work over managing people",
            ],
            "Management": [
                "Strengths: people skills, strategic thinking, cross-team coordination",
                "Growth path: Tech Lead -> Engineering Manager -> Director -> VP",
                "You enjoy: growing people, removing blockers, setting technical direction",
                "You prefer: team success over personal output, influence over execution",
            ],
            "reflection_prompt": "Which energizes you more: a day of deep coding, "
                                 "or a day of 1:1s and planning meetings?",
        },
        "2. T-Shaped Skills": {
            "Vertical bar (deep expertise)": [
                "What is the ONE area you want to be the go-to expert in?",
                "Examples: distributed systems, ML infrastructure, frontend performance,",
                "database internals, security engineering, compiler design",
            ],
            "Horizontal bars (broad knowledge)": [
                "What complementary areas would make you more effective?",
                "Examples: product sense, data analysis, DevOps, UX research,",
                "technical writing, project management, business strategy",
            ],
        },
        "3. Learning Goal (next 3 months)": {
            "template": [
                "Specific: 'Learn [skill/technology]'",
                "Measurable: 'Complete [course/project/certification]'",
                "Achievable: 'Dedicate [X hours/week]'",
                "Relevant: 'This helps me [career goal]'",
                "Time-bound: 'By [date]'",
            ],
            "example": "Learn Kubernetes by deploying a multi-service application "
                       "on a local cluster, completing the CKA study guide, "
                       "dedicating 5 hours/week for 12 weeks.",
        },
    }

    for section, content in framework.items():
        print(f"  {section}:")
        for key, value in content.items():
            if isinstance(value, list):
                print(f"    {key}:")
                for item in value:
                    print(f"      - {item}")
            else:
                print(f"    {key}: {value}")
        print()

    print("  Key insight: The best career path is the one that aligns your")
    print("  strengths with work that energizes you. There's no wrong choice")
    print("  between IC and management -- only the wrong fit for you.")


if __name__ == "__main__":
    print("=== Exercise 1: Identify Technical Debt ===")
    exercise_1()
    print("\n=== Exercise 2: Write a README ===")
    exercise_2()
    print("\n=== Exercise 3: Choose a License ===")
    exercise_3()
    print("\n=== Exercise 4: Ethical Scenario Analysis ===")
    exercise_4()
    print("\n=== Exercise 5: Career Reflection ===")
    exercise_5()
    print("\nAll exercises completed!")
