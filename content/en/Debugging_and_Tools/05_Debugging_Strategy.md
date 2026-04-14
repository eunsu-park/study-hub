# Debugging Strategy

**Previous**: [Common Bug Patterns](./04_Common_Bug_Patterns.md) | **Next**: [Logging](./06_Logging.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply the scientific method to debugging: hypothesize, test, refine
2. Use binary search to locate bugs in large codebases efficiently
3. Create a minimal reproducible example that isolates a bug
4. Distinguish between correlation and causation in debugging
5. Use the rubber duck debugging technique to verbalize assumptions
6. Apply the "work backward from the symptom" strategy
7. Know when to take a break and when to ask for help
8. Document bugs systematically for effective communication

---

Random guessing is the slowest way to debug. Professional developers follow systematic strategies that dramatically reduce the time from "something's wrong" to "found it and fixed it." This lesson teaches you the mental frameworks and techniques that experienced developers use to hunt bugs efficiently.

> **Key Insight:** Debugging is not about finding the fix -- it's about finding the **cause**. Once you truly understand the cause, the fix is usually obvious.

---

## 1. The Scientific Method for Debugging

Debugging is fundamentally a scientific process:

```
┌─────────────────────────────────────────────────┐
│           The Debugging Scientific Method        │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. OBSERVE    → What is the actual behavior?    │
│       │                                          │
│       ▼                                          │
│  2. HYPOTHESIZE → What could cause this?         │
│       │                                          │
│       ▼                                          │
│  3. PREDICT    → If my hypothesis is correct,    │
│       │          what should I see when I test?   │
│       │                                          │
│       ▼                                          │
│  4. TEST       → Run the experiment              │
│       │                                          │
│       ├─ Prediction correct → Hypothesis likely  │
│       │   right → Try to fix                     │
│       │                                          │
│       └─ Prediction wrong → Revise hypothesis    │
│           → Go back to step 2                    │
│                                                  │
└─────────────────────────────────────────────────┘
```

### Example in Practice

**Observation**: "The function returns 0 instead of the expected sum."

**Hypothesis 1**: "Maybe the input list is empty."
- **Test**: Add `print(f"input: {numbers}")` before the calculation.
- **Result**: Input is `[10, 20, 30]` -- not empty. Hypothesis rejected.

**Hypothesis 2**: "Maybe the accumulator variable is being reset."
- **Test**: Print the accumulator inside the loop.
- **Result**: The accumulator stays at 0. Found it -- the `+=` is inside an `if` block that never executes.

---

## 2. Binary Search Debugging

When you have a large chunk of code and the bug is somewhere in it, use binary search to find it in O(log n) time.

### 2.1 The Technique

```
Code block: 100 lines
│
├── Lines 1-50:   Add a check/print here. Bug present? → Bug is in 1-50
│   ├── Lines 1-25:   Check here. Bug present? → Bug is in 1-25
│   │   └── ...narrow down to exact line
│   └── Lines 26-50:  Bug not in 1-25? → It's here
│
└── Lines 51-100: Bug not in 1-50? → It's here
```

### 2.2 Practical Application

```python
def complex_pipeline(data):
    step1 = clean_data(data)
    step2 = validate(step1)
    step3 = transform(step2)
    step4 = aggregate(step3)
    step5 = format_output(step4)
    return step5
```

Binary search approach:
```python
def complex_pipeline(data):
    step1 = clean_data(data)
    step2 = validate(step1)
    step3 = transform(step2)
    
    # CHECKPOINT: Is the data correct at this point?
    print(f"After step 3: {step3!r}")
    # If correct → bug is in steps 4-5
    # If wrong  → bug is in steps 1-3
    
    step4 = aggregate(step3)
    step5 = format_output(step4)
    return step5
```

### 2.3 Binary Search with Comments

Another variant: comment out half the code and see if the bug persists.

```python
def process(data):
    result = step_a(data)
    result = step_b(result)
    # result = step_c(result)   # Commented out
    # result = step_d(result)   # Commented out
    return result
    
# If bug disappears → it was in step_c or step_d
# If bug remains → it's in step_a or step_b
```

---

## 3. Creating a Minimal Reproducible Example (MRE)

An MRE is the **smallest possible code** that still reproduces the bug. This is the single most important debugging technique.

### 3.1 Why Create an MRE?

- Forces you to **understand** the bug (often reveals the cause during creation)
- Eliminates irrelevant code that distracts from the root cause
- Makes it easy to **share** with others for help
- Lets you **test fixes** quickly without running the full application

### 3.2 How to Create an MRE

```
Start with the full buggy code
          │
          ▼
Remove one component at a time
          │
          ▼
   Bug still present?
    ┌──────┴──────┐
    │             │
   Yes           No
    │             │
    ▼             ▼
Continue       Put it back;
removing       it's relevant
    │
    ▼
Repeat until nothing more can be removed
    │
    ▼
You have your MRE
```

### 3.3 Example: From 200 Lines to 10

Original (200 lines, database, API, logging, etc.):
```python
# Can't share this -- it needs a database, config files, and API keys
class OrderProcessor:
    def __init__(self, db, api_client, logger):
        ...
    def process(self, order_id):
        order = self.db.fetch(order_id)
        items = self.api_client.get_items(order["items"])
        total = self.calculate_total(items)  # BUG: wrong total
        ...
```

MRE (10 lines):
```python
# Bug: calculate_total returns wrong result for items with discount
def calculate_total(items):
    total = 0
    for item in items:
        total += item["price"] - item.get("discount", 0)
    return total

items = [{"price": 100, "discount": 10}, {"price": 50}]
print(calculate_total(items))  # Expected: 140, Got: 140
# Actually the bug appears only with percentage discounts...

items = [{"price": 100, "discount": "10%"}, {"price": 50}]
print(calculate_total(items))  # TypeError! discount is a string
```

---

## 4. Work Backward from the Symptom

Instead of reading code top-to-bottom, start at the **wrong output** and trace backward.

### 4.1 The Process

```
Wrong Output
    │
    ▼
What line produces this output?
    │
    ▼
What variables feed into that line?
    │
    ▼
Where do those variables get their values?
    │
    ▼
Are those values correct?
    ├── Yes → The bug is in how they're combined
    └── No  → Trace further back to find where they went wrong
```

### 4.2 Example

```python
def generate_report(sales_data):
    # ... many lines of code ...
    report = {
        "total_revenue": total_revenue,      # This is wrong! 
        "avg_sale": avg_sale,
        "count": count,
    }
    return report
```

Trace backward:
1. `total_revenue` is wrong → Where is it calculated?
2. `total_revenue = sum(amounts)` → What's in `amounts`?
3. `amounts = [sale["amount"] for sale in filtered_sales]` → What's `filtered_sales`?
4. Found it: the filter excluded valid sales because of a date comparison bug.

---

## 5. Rubber Duck Debugging

Explain your code, line by line, to an inanimate object (a rubber duck, a pet, a wall). The act of verbalizing forces you to think clearly about each step.

### 5.1 How to Do It

1. Put a rubber duck (or anything) on your desk
2. Explain what the code is **supposed** to do
3. Go through the code line by line, explaining what each line **actually** does
4. When your explanation of what it "should do" diverges from what it "actually does," you've found the bug

### 5.2 Why It Works

- **Forces precision**: You can't wave your hands when talking to a duck
- **Exposes assumptions**: "This line increments the counter... wait, does it?"
- **Slows you down**: Bugs hide when you skim; they reveal themselves when you read carefully

### 5.3 Structured Self-Questioning

If you don't have a duck, ask yourself these questions:

1. "What should happen at this line?"
2. "What actually happens?"
3. "What assumptions am I making?"
4. "Have I verified those assumptions?"
5. "What would I tell a colleague about this code?"

---

## 6. Common Debugging Anti-Patterns

### 6.1 Shotgun Debugging

**Anti-pattern**: Changing random things and hoping the bug goes away.

```python
# "Maybe if I add +1 here... no... maybe -1... let me try abs()..."
result = abs(total + 1) - 1  # Why? No idea, but it seems to work
```

**Problem**: Even if you stumble onto a fix, you don't understand the bug. It will come back or cause new issues.

### 6.2 Blame-the-Tool

**Anti-pattern**: "It must be a bug in Python/the library/the OS."

**Reality**: 99.9% of the time, the bug is in your code. Check your code thoroughly before suspecting external tools.

### 6.3 The Fix-Without-Understanding

**Anti-pattern**: Applying a fix from Stack Overflow without understanding why it works.

**Problem**: You might fix the symptom but not the cause, or introduce new bugs.

### 6.4 Tunnel Vision

**Anti-pattern**: Staring at the same 5 lines for an hour, convinced the bug is there.

**Fix**: Take a step back. The bug might be in a completely different part of the code. The wrong variable might have been set incorrectly 100 lines earlier.

---

## 7. When to Take a Break

### 7.1 Signs You Need a Break

- You've been staring at the same code for more than 30 minutes without progress
- You're making changes without a clear hypothesis
- You're frustrated or angry at the code
- You keep re-reading the same lines

### 7.2 What to Do

- Walk away for 15 minutes (seriously -- do something else)
- Explain the problem to someone (even in a message you never send)
- Switch to a different task and come back later
- Sleep on it (your subconscious is better at debugging than you think)

> **Experience:** Almost every senior developer has a story about spending hours on a bug, going home frustrated, and solving it in 5 minutes the next morning. This is not a coincidence -- fresh eyes see what tired eyes cannot.

---

## 8. When and How to Ask for Help

### 8.1 Before Asking

1. Read the error message carefully
2. Search for the error message online
3. Create a minimal reproducible example
4. List what you've already tried
5. Check the documentation

### 8.2 How to Write a Good Bug Report

```
## What I Expected
The function should return the sum of all positive numbers.

## What Actually Happened
It returns 0 for any input containing negative numbers.

## Minimal Reproducible Example
```python
def sum_positive(numbers):
    total = 0
    for n in numbers:
        if n > 0:
            total += n
        else:
            return 0   # BUG: returns immediately instead of skipping
    return total

print(sum_positive([1, -2, 3]))  # Expected: 4, Got: 0
`` `

## What I've Tried
- Verified the input is correct
- Added print statements to trace execution
- The function returns at the first negative number instead of skipping it

## Environment
Python 3.12, macOS 14.0
```

---

## 9. Debugging Checklists

### Quick Debugging Checklist

```
□ Read the error message (bottom-to-top for tracebacks)
□ Check the exact line mentioned in the error
□ Print the values going into that line
□ Check types (is it a string when you expect an int?)
□ Check for None
□ Check loop boundaries (off-by-one?)
□ Check recent changes (what did you change last?)
```

### Systematic Debugging Checklist

```
□ Can you reproduce the bug consistently?
□ What is the expected behavior vs actual behavior?
□ What is the minimal input that triggers the bug?
□ Where in the code does the behavior diverge from expectations?
□ What changed recently? (git diff, git log)
□ Are there any edge cases being handled incorrectly?
□ Are there any assumptions that might be wrong?
```

---

## 10. Putting It All Together

The debugging workflow for any bug:

```
1. REPRODUCE      Can you make it happen reliably?
       │
       ▼
2. ISOLATE        Create an MRE. What's the minimal trigger?
       │
       ▼
3. LOCATE         Binary search + tracing to find the exact line
       │
       ▼
4. UNDERSTAND     Why does this line produce the wrong result?
       │
       ▼
5. FIX            Change the code to produce the correct result
       │
       ▼
6. VERIFY         Does the fix work? Does it break anything else?
       │
       ▼
7. PREVENT        Add a test so this bug never returns
```

---

## Summary

- Debugging is a scientific process: observe, hypothesize, predict, test
- Binary search narrows down the bug location in logarithmic time
- Creating an MRE is the most powerful debugging technique -- it often reveals the bug during creation
- Work backward from the wrong output to find where values go wrong
- Rubber duck debugging forces you to verbalize (and question) your assumptions
- Avoid anti-patterns: shotgun debugging, blame-the-tool, fix-without-understanding
- Take breaks -- fresh eyes find bugs that tired eyes miss
- When asking for help, provide an MRE and document what you've tried

---

## Exercises

1. Apply binary search debugging to locate a bug in a multi-step pipeline
2. Create a minimal reproducible example from a buggy module
3. Practice rubber duck debugging by writing out your reasoning for a bug
4. Write a structured bug report for a given bug scenario

**Previous**: [Common Bug Patterns](./04_Common_Bug_Patterns.md) | **Next**: [Logging](./06_Logging.md)
