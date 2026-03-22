# Developer Practices and Ethics

**Previous**: [Software Architecture Basics](./15_Software_Architecture.md)

> **Topic**: Programming
> **Lesson**: 16 of 16
> **Prerequisites**: Software development experience, version control, basic understanding of software lifecycle

## Learning Objectives

After completing this lesson, you will be able to:

1. Classify technical debt using Martin Fowler's quadrant (deliberate/accidental, prudent/reckless) and prioritize repayment strategies
2. Apply documentation best practices: code comments that explain "why," API docstrings, architecture diagrams (C4 model), and project READMEs
3. Contribute to open source projects following standard workflows (fork, branch, PR) and community etiquette
4. Select appropriate open source licenses (MIT, Apache 2.0, GPL, AGPL) based on project goals and compatibility requirements
5. Apply ethical principles to software development: privacy by design, algorithmic fairness, accessibility (WCAG), and security responsibility
6. Recognize and avoid dark patterns in UI/UX design
7. Build sustainable career practices including continuous learning habits, T-shaped skill development, and imposter syndrome management
8. Apply the pragmatic programmer mindset: care about craft, DRY principle, prototype to learn, and refactor continuously

---

## Introduction

Writing code is only part of being a professional developer. This lesson covers the **practices, principles, and ethics** that distinguish great developers from merely competent ones:

- **Technical debt management**: Balance speed and quality
- **Documentation**: Make your work understandable and maintainable
- **Open source**: Contribute to and create public software
- **Ethics**: Build software responsibly
- **Career growth**: Develop sustainable, fulfilling careers

These practices shape how you work, collaborate, and contribute to the broader software community.

## Technical Debt

**Technical debt** is the implied cost of future rework caused by choosing quick solutions today instead of better approaches that would take longer.

### Definition

Technical debt is a **metaphor**:
- Like financial debt, it can be **strategic** (borrow now, pay later)
- It accrues **interest**: the longer you wait, the harder it is to fix
- Too much debt leads to **bankruptcy**: codebase becomes unmaintainable

```python
# Quick solution (creates technical debt)
def process_data(data):
    # TODO: Handle edge cases, add validation
    return data.split(',')[0]  # Fragile, assumes format

# Better solution (takes more time upfront)
def process_data(data):
    if not isinstance(data, str):
        raise TypeError("Data must be a string")
    if not data:
        raise ValueError("Data cannot be empty")

    parts = data.split(',')
    if len(parts) == 0:
        raise ValueError("Invalid format")

    return parts[0].strip()
```

### Types of Technical Debt

**Deliberate vs Accidental**:

| Deliberate | Accidental |
|------------|------------|
| "We know this is hacky but we need to ship" | "We didn't know a better way" |
| Conscious decision | Lack of knowledge/skill |
| Can be planned for | Discovered later |

**Prudent vs Reckless** (Martin Fowler's quadrant):

```
              Deliberate
                 |
    Prudent      |     Reckless
   "Ship now,    |    "We don't have
    refactor     |     time for design"
     later"      |
-----------------+------------------
   "What's       |     "We didn't
   layering?"    |      realize we'd
                 |    duplicate code"
              Accidental
```

**Examples**:

```javascript
// Deliberate + Prudent: Hardcode now, make configurable later
const API_URL = 'https://api.example.com';  // TODO: Move to config

// Deliberate + Reckless: Copy-paste without thinking
function calculateDiscount1(price) { return price * 0.1; }
function calculateDiscount2(price) { return price * 0.1; }
function calculateDiscount3(price) { return price * 0.1; }

// Accidental + Prudent: Learned better pattern, need to refactor
// Old code: global variables (didn't know better)
let userId = 123;
// New code: dependency injection (learned the pattern)
class UserService {
    constructor(userId) { this.userId = userId; }
}

// Accidental + Reckless: Didn't understand problem, made it worse
// Mixing concerns without realizing
function saveUser(user) {
    db.save(user);
    sendEmail(user.email);  // Side effect!
    logAnalytics(user.id);  // Side effect!
}
```

### Managing Technical Debt

**1. Track it**: Document debt in code or issue tracker
```java
// TODO-TECH-DEBT: This uses N+1 queries. Batch fetch orders.
public List<Order> getUserOrders(int userId) {
    User user = userRepository.findById(userId);
    List<Order> orders = new ArrayList<>();
    for (int orderId : user.getOrderIds()) {
        orders.add(orderRepository.findById(orderId));  // N queries
    }
    return orders;
}
```

**2. Prioritize it**: Not all debt is equal
- **High priority**: Security vulnerabilities, performance bottlenecks
- **Medium priority**: Code duplication, poor naming
- **Low priority**: Missing comments, minor style issues

**3. Pay it down**: Allocate time for refactoring
```
# Sprint planning
User stories: 60% of capacity
Tech debt:    20% of capacity
Bugs:         20% of capacity
```

**4. Prevent it**: Code review, automated testing, architectural guidelines

### When Technical Debt Is Acceptable

✅ **Good reasons** to incur debt:
- **Time-to-market**: Ship MVP to validate product hypothesis
- **Learning**: Build prototype to understand problem better
- **Experimentation**: A/B test before committing to full implementation

```python
# Acceptable: Quick prototype to test feature adoption
# If users like it, we'll rewrite properly
@app.route('/experimental-search')
def search():
    # PROTOTYPE: Replace with proper search engine if successful
    results = [item for item in items if query in item['name']]
    return jsonify(results)
```

❌ **Bad reasons**:
- "We'll fix it later" (without plan)
- "It works, ship it!" (ignoring quality)
- Ignoring known best practices out of laziness

### Technical Bankruptcy

When debt becomes **unbearable**, the codebase is unmaintainable:
- Every change breaks something
- Developers afraid to touch code
- Onboarding new developers takes months

**Solutions**:
1. **Incremental rewrite**: Strangle pattern (replace piece by piece)
2. **Big rewrite**: Risky, often fails (see Netscape rewrite disaster)
3. **Refactor aggressively**: Dedicate sprints to cleanup

## Documentation

Good documentation makes software **understandable**, **usable**, and **maintainable**.

### Types of Documentation

**1. Code comments**: Explain **why**, not what
```cpp
// Bad: Obvious what
int x = y + 5;  // Add 5 to y

// Good: Explain why
int maxRetries = baseRetries + 5;  // Add 5 buffer retries for flaky network

// Good: Explain non-obvious logic
// Use binary search for O(log n) instead of O(n) since array is sorted
int index = binarySearch(sortedArray, target);
```

**2. API documentation**: Reference + guides + tutorials
```python
def calculate_discount(price, customer_tier):
    """
    Calculate discount based on customer tier.

    Args:
        price (float): Original price in USD
        customer_tier (str): One of 'bronze', 'silver', 'gold', 'platinum'

    Returns:
        float: Discounted price

    Raises:
        ValueError: If customer_tier is invalid

    Examples:
        >>> calculate_discount(100, 'gold')
        85.0
        >>> calculate_discount(100, 'platinum')
        75.0
    """
    tiers = {'bronze': 0, 'silver': 0.05, 'gold': 0.15, 'platinum': 0.25}
    if customer_tier not in tiers:
        raise ValueError(f"Invalid tier: {customer_tier}")
    return price * (1 - tiers[customer_tier])
```

**3. Architecture documentation**: High-level design decisions
```markdown
# Architecture: C4 Model

## Context Diagram
Shows how system fits in the world (users, external systems).

## Container Diagram
Shows high-level tech choices (web app, API, database).

## Component Diagram
Shows components within each container.

## Code Diagram
Shows classes/modules (usually autogenerated from code).
```

**4. README**: The front door of your project
```markdown
# Project Name

Brief description of what this project does.

## Features
- Feature 1
- Feature 2

## Installation
```bash
pip install myproject
```

## Quick Start
```python
from myproject import main
main()
```

## Documentation
Full documentation at https://docs.example.com

## Contributing
See CONTRIBUTING.md

## License
MIT License
```

### When to Write Documentation

**Write documentation when**:
- Public API: Users need to understand how to use it
- Complex algorithm: Future you will forget why it works
- Onboarding: New team members need to get started
- Architecture decision: Record why you made choice

**Don't write documentation when**:
- Code is self-explanatory (good naming, clear structure)
- Documentation would just repeat code

### Documentation as Code

Store documentation in **version control** alongside code:
```
project/
├── src/
├── docs/
│   ├── architecture.md
│   ├── api-reference.md
│   └── deployment.md
├── README.md
└── CONTRIBUTING.md
```

**Benefits**:
- Reviewed in pull requests
- Versioned with code (docs match code version)
- Easy to update

**Tools**: Sphinx, MkDocs, Docusaurus, JSDoc

### The C4 Model for Architecture Documentation

**C4** stands for Context, Containers, Components, Code.

```
Level 1: System Context
┌──────────────────────────────────────┐
│  User  →  [System]  →  [Database]    │
└──────────────────────────────────────┘

Level 2: Containers
┌──────────────────────────────────────┐
│  User  →  [Web App]  →  [API]  →  DB│
└──────────────────────────────────────┘

Level 3: Components (within API container)
┌──────────────────────────────────────┐
│  [Controller] → [Service] → [Repo]   │
└──────────────────────────────────────┘

Level 4: Code (classes, functions)
```

Use diagrams sparingly—they become outdated quickly. Focus on Levels 1-2.

## Open Source

**Open source software** is software with source code that anyone can inspect, modify, and distribute.

### How Open Source Works

**Licenses**: Legal terms for using, modifying, distributing software
**Contributions**: Submit bug fixes, features via pull requests
**Community**: Maintainers, contributors, users collaborate

### Contributing to Open Source

**Why contribute**:
- **Learn**: Study real-world codebases
- **Build portfolio**: Show your work to employers
- **Give back**: Improve tools you use
- **Network**: Meet developers worldwide

**How to start**:
1. **Find a project**: Look for repos with `good-first-issue` label
2. **Read CONTRIBUTING.md**: Understand project workflow
3. **Start small**: Fix typos, improve docs, add tests
4. **Communicate**: Ask questions, be respectful

**Example: First contribution**:
```bash
# 1. Fork repository on GitHub
# 2. Clone your fork
git clone https://github.com/yourname/projectname.git
cd projectname

# 3. Create branch
git checkout -b fix-typo-readme

# 4. Make changes
# Edit README.md

# 5. Commit
git commit -m "docs: fix typo in installation instructions"

# 6. Push
git push origin fix-typo-readme

# 7. Open pull request on GitHub
```

**Etiquette**:
- Be respectful: Maintainers are volunteers
- Follow guidelines: CONTRIBUTING.md, code style
- Be patient: Reviews take time
- Accept feedback: Code changes are requested for good reasons

### Creating Open Source Projects

**When to open source**:
- Tool is useful beyond your company
- You want community contributions
- You're building a portfolio

**Steps**:
1. **Choose a license** (see next section)
2. **Write README**: What it does, how to install, how to use
3. **Add CONTRIBUTING.md**: How to contribute
4. **Add CODE_OF_CONDUCT.md**: Community behavior expectations
5. **Publish**: Push to GitHub, GitLab, etc.

**Example README**:
```markdown
# AwesomeTool

A CLI tool for automating deployments.

## Installation
```bash
npm install -g awesome-tool
```

## Usage
```bash
awesome-tool deploy --env production
```

## Contributing
See CONTRIBUTING.md

## License
MIT License
```

### Popular Open Source Licenses

**Permissive licenses**: Minimal restrictions

| License | Description |
|---------|-------------|
| **MIT** | Very permissive. Use, modify, distribute freely. |
| **Apache 2.0** | Like MIT, but includes patent grant. |
| **BSD** | Like MIT, with slight variations (2-clause, 3-clause). |

**Copyleft licenses**: Derivatives must also be open source

| License | Description |
|---------|-------------|
| **GPL v3** | Strong copyleft. Any derivative must be GPL. |
| **LGPL v3** | Weak copyleft. Linking to library is allowed. |
| **AGPL v3** | Like GPL, but also covers network use (SaaS). |

**When to use**:
- **MIT**: Most projects (simple, widely understood)
- **Apache 2.0**: If patents are a concern
- **GPL v3**: If you want to ensure derivatives stay open
- **AGPL v3**: If you want to ensure SaaS stays open

**Example: Adding license**:
```bash
# Add LICENSE file with chosen license text
# Add to package.json or setup.py
"license": "MIT"
```

### License Compatibility

Some licenses are incompatible:

```
✅ OK: MIT library in Apache project (permissive → permissive)
✅ OK: MIT library in GPL project (permissive → copyleft)
❌ NOT OK: GPL library in MIT project (copyleft → permissive)
```

**Rule of thumb**: Permissive licenses are compatible with everything. Copyleft licenses are not compatible with permissive.

### Dual Licensing

Some projects offer **two licenses**:
- **Open source license** (GPL): Free for open source projects
- **Commercial license**: Paid for proprietary projects

**Example**: MySQL (GPL or commercial license)

## Ethics in Software Development

Software has **real-world impact**. Developers have **ethical responsibilities**.

### Privacy and Data Protection

**Principles**:
- **Minimize collection**: Only collect data you need
- **Purpose limitation**: Use data only for stated purpose
- **User control**: Let users access, delete their data
- **Security**: Protect data from breaches

**Regulations**:
- **GDPR** (EU): Strict data protection, user rights (access, deletion, portability)
- **CCPA** (California): Similar to GDPR for California residents

**Example: Privacy-friendly design**:
```python
# Bad: Collect everything
user = {
    'name': name,
    'email': email,
    'birthday': birthday,
    'location': location,
    'browsing_history': history,  # Do you need this?
}

# Good: Collect only what's needed
user = {
    'email': email,  # For login
}
```

### Algorithmic Bias and Fairness

**Bias** in algorithms can perpetuate discrimination.

**Sources of bias**:
1. **Training data bias**: Data reflects historical inequities
2. **Feature selection bias**: Choosing features correlated with protected attributes
3. **Label bias**: Labels reflect human biases

**Example: Hiring algorithm**:
```python
# Biased: Training data has more male engineers
model.fit(historical_resumes, historical_hires)
# Model learns: male → higher score

# Mitigation: Remove gender-correlated features, balanced training data
```

**Principles**:
- **Fairness**: Treat similar individuals similarly
- **Transparency**: Explain how decisions are made
- **Accountability**: Who is responsible for outcomes?

### Accessibility

**Accessibility** ensures software is usable by everyone, including people with disabilities.

**Web accessibility (WCAG)**:
- **Perceivable**: Text alternatives for images, captions for videos
- **Operable**: Keyboard navigation, no flashing content (seizures)
- **Understandable**: Clear language, consistent navigation
- **Robust**: Compatible with assistive technologies (screen readers)

**Example: Accessible HTML**:
```html
<!-- Bad: Image without alt text -->
<img src="chart.png">

<!-- Good: Descriptive alt text -->
<img src="chart.png" alt="Bar chart showing sales growth from 2020 to 2024">

<!-- Bad: Button without label -->
<button><span class="icon-close"></span></button>

<!-- Good: Button with accessible label -->
<button aria-label="Close dialog">
  <span class="icon-close" aria-hidden="true"></span>
</button>
```

**Benefits beyond disability**:
- Captions help in noisy environments
- Keyboard navigation helps power users
- Clear language helps non-native speakers

### Security Responsibility

**Developers are responsible** for protecting user data.

**Principles**:
- **Defense in depth**: Multiple layers of security
- **Least privilege**: Grant minimal permissions
- **Fail securely**: Errors should not expose data
- **Keep secrets secret**: Never hardcode passwords, API keys

**Example: Secure password storage**:
```javascript
// Bad: Plain text passwords
db.save({ username, password });

// Bad: Reversible encryption
db.save({ username, password: encrypt(password) });

// Good: Hashed with salt
const bcrypt = require('bcrypt');
const hashedPassword = await bcrypt.hash(password, 10);
db.save({ username, password: hashedPassword });
```

**Example: SQL injection prevention**:
```python
# Bad: String concatenation (SQL injection risk)
query = f"SELECT * FROM users WHERE email = '{email}'"
db.execute(query)

# Good: Parameterized query
query = "SELECT * FROM users WHERE email = ?"
db.execute(query, (email,))
```

### Dark Patterns

**Dark patterns** are manipulative UI/UX designs that trick users.

**Examples**:
- **Forced continuity**: Hard to cancel subscriptions
- **Hidden costs**: Fees revealed at checkout
- **Bait and switch**: Button says one thing, does another
- **Confirmshaming**: "No thanks, I don't want to save money"

**Ethical stance**: Don't build dark patterns. Respect user autonomy.

### AI Ethics

As AI becomes prevalent, developers must consider:

**Transparency**: Can users understand how AI decisions are made?
```python
# Black box: Neural network decision
prediction = model.predict(features)

# Transparent: Explain features used
prediction = model.predict(features)
explanation = explainer.explain_instance(features)
# "Decision based on: credit score (0.6), income (0.3), ..."
```

**Accountability**: Who is responsible when AI makes mistakes?
- Self-driving car crash: Driver? Manufacturer? Developer?

**Bias**: Is AI fair across demographic groups?
- Face recognition works worse for darker skin tones (training data bias)

**Privacy**: How is training data collected? Can models leak training data?

## Developer Well-Being

Software development is a **marathon**, not a sprint. Sustainable careers require well-being.

### Sustainable Pace

**Avoid burnout**:
- **Work-life balance**: Don't work nights/weekends regularly
- **Breaks**: Step away from screen, take walks
- **Vacations**: Fully disconnect (don't check Slack)

**Crunch time is harmful**:
- Productivity drops after 40-50 hours/week
- Bugs increase with fatigue
- Burnout leads to attrition

**Red flags**:
- "We're a family" (code for overwork)
- Constant firefighting (poor planning)
- Hero culture (rewarding overwork)

### Continuous Learning

Technology changes rapidly. **Continuous learning** is essential.

**Strategies**:
- **Read code**: Study open source projects
- **Side projects**: Build tools for fun
- **Courses**: Online courses, books, conferences
- **Teach**: Writing, mentoring solidifies knowledge

**Avoid**:
- **Hype-driven development**: Don't chase every new framework
- **Tutorial hell**: Balance learning with building
- **Comparison**: Everyone progresses at different rates

### Imposter Syndrome

**Imposter syndrome**: Feeling like a fraud despite evidence of competence.

**It's normal**:
- Experienced developers feel it too
- Industry changes quickly (everyone is learning)

**Strategies**:
- **Document wins**: Keep a "brag document"
- **Talk about it**: You'll find others feel the same
- **Perspective**: You don't need to know everything

## Career Growth

Software careers offer two primary tracks.

### IC Track vs Management Track

**Individual Contributor (IC) Track**:
- **Focus**: Technical depth
- **Progression**: Junior → Mid → Senior → Staff → Principal
- **Responsibilities**: Write code, design systems, mentor

**Management Track**:
- **Focus**: People, processes, strategy
- **Progression**: Team Lead → Manager → Director → VP
- **Responsibilities**: Hiring, performance reviews, roadmap planning

**Neither is better**: Choose based on what energizes you.

### T-Shaped Skills

**T-shaped developer**:
- **Vertical bar**: Deep expertise in one area (e.g., backend systems)
- **Horizontal bar**: Broad knowledge across areas (frontend, databases, DevOps)

```
   Frontend   Backend   DevOps   ML   Mobile
      |         |||       |      |      |
      |         |||       |      |      |
      |         |||       |      |      |
             (Deep in backend, broad elsewhere)
```

**How to build**:
- **Depth**: Master one domain (3-5 years)
- **Breadth**: Side projects, cross-functional work

### Building a Learning Habit

**Techniques**:
- **Daily reading**: 30 minutes on blogs, docs
- **Weekly experiments**: Try new tool/language
- **Monthly projects**: Build something end-to-end
- **Quarterly deep dives**: Read a book, take a course

**Resources**:
- **Blogs**: Martin Fowler, Joel Spolsky, engineering blogs (Uber, Netflix)
- **Books**: "Clean Code", "Designing Data-Intensive Applications"
- **Podcasts**: Software Engineering Daily, Changelog
- **Conferences**: Local meetups, online conferences

## The Pragmatic Programmer Mindset

From the book *The Pragmatic Programmer* by Andrew Hunt and David Thomas:

**1. Care about your craft**
- Take pride in your work
- Don't tolerate broken windows (bad code)

**2. Think about your work**
- Don't code on autopilot
- Question decisions: "Why are we doing it this way?"

**3. Be a catalyst for change**
- Don't wait for permission to improve things
- Show, don't tell (build prototype, then convince)

**4. Remember the big picture**
- Don't get lost in details
- Understand business context

**5. DRY: Don't Repeat Yourself**
- Duplication is waste (code, knowledge, documentation)

**6. Make it easy to reuse**
- Write modular, decoupled code

**7. Prototype to learn**
- Build throwaway code to explore ideas

**8. Estimate to avoid surprises**
- Learn to estimate time (practice, improve)

**9. Refactor early, refactor often**
- Code rots—keep it clean

**10. Test your software, or your users will**
- Automated tests catch regressions

## Exercises

### Exercise 1: Identify Technical Debt

Review one of your projects and identify:
1. One example of **deliberate, prudent** technical debt
2. One example of **accidental** technical debt
3. Prioritize: Which should you address first? Why?

### Exercise 2: Write a README

Write a README for a hypothetical project: a command-line tool for managing TODO lists. Include:
- Project description
- Installation instructions
- Usage examples
- Contributing guidelines
- License

### Exercise 3: Choose a License

You're releasing an open source library for machine learning. You want:
- Anyone to use it freely
- Derivatives to also be open source
- Protection against patent lawsuits

Which license would you choose? Justify your answer.

### Exercise 4: Ethical Scenario Analysis

Analyze this scenario:

> Your company is building a credit scoring algorithm. The model uses ZIP code as a feature. You notice that ZIP code is highly correlated with race, and the model gives lower scores to predominantly Black neighborhoods.

Questions:
1. What is the ethical issue?
2. What are potential harms?
3. What would you do?

### Exercise 5: Career Reflection

Reflect on your own career:
1. Are you more interested in the IC track or management track? Why?
2. What is your "vertical bar" (area of deep expertise)?
3. What "horizontal bars" (broad knowledge) do you want to develop?
4. What is one learning goal for the next 3 months?

## Summary

Professional software development extends far beyond writing code:

- **Technical Debt**: Balance speed and quality; manage debt intentionally
- **Documentation**: Make software understandable through code comments, API docs, architecture diagrams, and READMEs
- **Open Source**: Contribute to and create open source projects; understand licenses (MIT, Apache, GPL)
- **Ethics**: Prioritize privacy, fairness, accessibility, security, and transparency
- **Well-Being**: Maintain sustainable pace, continuous learning, and manage imposter syndrome
- **Career Growth**: Choose IC or management track, develop T-shaped skills, build learning habits
- **Pragmatic Mindset**: Care about craft, think critically, refactor continuously, test thoroughly

Great developers combine **technical skill** with **professionalism, ethics, and continuous growth**. The code you write today becomes the legacy you leave tomorrow—make it count.

---

**Previous**: [Software Architecture Basics](./15_Software_Architecture.md)
