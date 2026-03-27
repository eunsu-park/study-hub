"""
Exercises for Lesson 08: Prompt Engineering
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""

import re
from collections import Counter


# === Exercise 1: Zero-shot vs Few-shot Comparison ===
# Problem: Write both a zero-shot and a few-shot prompt for product review
# sentiment classification (Positive, Negative, Neutral).

def exercise_1():
    """Zero-shot vs few-shot prompt design for sentiment classification."""

    review = "The battery life is decent but the screen quality is disappointing."

    zero_shot = f"""Classify the sentiment of the following product review as Positive, Negative, or Neutral.

Review: "{review}"

Sentiment:"""

    few_shot = f"""Classify the sentiment of the following product reviews.

Review: "Absolutely love this! Best purchase I've made all year."
Sentiment: Positive

Review: "Terrible quality. Broke after two days of use."
Sentiment: Negative

Review: "It does what it says. Nothing special."
Sentiment: Neutral

Review: "{review}"
Sentiment:"""

    print("Exercise 1: Zero-shot vs Few-shot Comparison")
    print("=" * 60)

    print("\n--- Zero-shot Prompt ---")
    print(zero_shot)

    print("\n--- Few-shot Prompt ---")
    print(few_shot)

    print("\n--- When to prefer each ---")
    print("Zero-shot: Use when the task is straightforward and the model's")
    print("  training already covers it well. Faster to write and uses fewer tokens.")
    print("Few-shot: Use when you need a specific output format, the task is")
    print("  ambiguous, or you want consistent style. Especially valuable for")
    print("  rare label sets or domain-specific tasks.")
    print("\nRule of thumb: start with zero-shot, switch to few-shot if results")
    print("are inconsistent.")


# === Exercise 2: Chain-of-Thought Prompt Design ===
# Problem: Design a zero-shot CoT prompt for a budgeting calculation.

def exercise_2():
    """Chain-of-Thought prompt design for financial reasoning."""

    cot_prompt = """A user wants to know if they can save enough for a vacation.
- Monthly income (after tax): $4,200
- Fixed monthly expenses: $2,800
- Vacation cost: $1,500
- Months until vacation: 3

Can they afford it? Let's think step by step."""

    # Simulated model reasoning
    expected_reasoning = """
1. Monthly savings = Income - Expenses = $4,200 - $2,800 = $1,400
2. Total savings in 3 months = $1,400 x 3 = $4,200
3. Vacation cost = $1,500
4. $4,200 > $1,500, so yes, they can afford it.
   They'll have $4,200 - $1,500 = $2,700 left over.

Answer: Yes, they can afford the vacation."""

    # Verify the math programmatically
    income = 4200
    expenses = 2800
    vacation_cost = 1500
    months = 3

    monthly_savings = income - expenses
    total_savings = monthly_savings * months
    can_afford = total_savings >= vacation_cost
    leftover = total_savings - vacation_cost

    print("Exercise 2: Chain-of-Thought Prompt Design")
    print("=" * 60)

    print("\n--- CoT Prompt ---")
    print(cot_prompt)

    print("\n--- Expected Model Reasoning ---")
    print(expected_reasoning)

    print("\n--- Verification ---")
    print(f"  Monthly savings: ${monthly_savings:,}")
    print(f"  Total savings (3 months): ${total_savings:,}")
    print(f"  Vacation cost: ${vacation_cost:,}")
    print(f"  Can afford: {can_afford}")
    print(f"  Leftover: ${leftover:,}")

    print("\nWhy CoT helps: Without step-by-step reasoning, the model might")
    print("jump to an answer and make arithmetic errors. The phrase 'Let's")
    print("think step by step' elicits structured reasoning that reduces")
    print("errors on multi-step math problems.")


# === Exercise 3: Structured Output Extraction ===
# Problem: Write a prompt that extracts structured information from a job
# posting as valid JSON.

def exercise_3():
    """Structured output extraction from job postings."""

    extraction_prompt = """Extract job information from the posting below and return it as JSON.
Use null for any fields not mentioned.

Required JSON structure:
{{
  "title": "job title",
  "company": "company name",
  "location": "city/remote",
  "salary_range": "e.g. $80,000-$100,000 or null",
  "required_skills": ["skill1", "skill2"],
  "experience_years": 3
}}

Job posting:
{posting}

JSON:"""

    posting = """
Senior ML Engineer at DataCorp
San Francisco, CA (hybrid)
We're looking for someone with 5+ years of experience in Python and PyTorch.
Knowledge of distributed training and MLflow is a plus.
"""

    # Simulated extraction result
    expected_output = {
        "title": "Senior ML Engineer",
        "company": "DataCorp",
        "location": "San Francisco, CA (hybrid)",
        "salary_range": None,
        "required_skills": ["Python", "PyTorch", "distributed training", "MLflow"],
        "experience_years": 5,
    }

    print("Exercise 3: Structured Output Extraction")
    print("=" * 60)

    print("\n--- Extraction Prompt Template ---")
    print(extraction_prompt.format(posting=posting.strip()))

    print("\n--- Expected JSON Output ---")
    import json
    print(json.dumps(expected_output, indent=2))

    print("\n--- Key Design Decisions ---")
    print("1. Use double braces {{ }} to escape literal braces in format strings")
    print("2. Providing the exact schema reduces hallucinated fields")
    print("3. 'null' instruction prevents inventing data for missing fields")
    print("4. Adding 'JSON:' at the end primes the model to output JSON directly")


# === Exercise 4: Self-Consistency Implementation ===
# Problem: Implement self-consistency prompting that generates multiple
# reasoning paths and returns the most common answer.

def exercise_4():
    """Self-consistency implementation with majority voting."""

    def extract_final_answer(response):
        """Extract the answer from a CoT response."""
        match = re.search(
            r"(?:answer is|therefore)[:\s]+(.+?)(?:\.|$)",
            response, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
        return lines[-1] if lines else response.strip()

    def self_consistency(responses):
        """Take majority vote from multiple reasoning paths."""
        answers = [extract_final_answer(r) for r in responses]
        vote_counts = Counter(answers)
        final_answer = vote_counts.most_common(1)[0][0]
        return final_answer, dict(vote_counts), answers

    # Simulated reasoning paths for: "What is the average speed if a train
    # travels 120 km in 1.5 hours, and then 80 km in 1 hour?"
    simulated_responses = [
        "Total distance = 120 + 80 = 200 km. Total time = 1.5 + 1 = 2.5 hours. "
        "Average speed = 200/2.5 = 80 km/h. The answer is 80 km/h",

        "First leg: 120/1.5 = 80 km/h. Second leg: 80/1 = 80 km/h. "
        "Average = (80 + 80)/2 = 80 km/h. The answer is 80 km/h",

        "Speed1 = 120/1.5 = 80. Speed2 = 80/1 = 80. "
        "Average speed = 200/2.5 = 80. The answer is 80 km/h",

        "Total = 200 km in 2.5 hours. 200/2.5 = 80. Therefore 80 km/h",

        # One path with an error (arithmetic mistake)
        "120 + 80 = 190 km. 1.5 + 1 = 2.5 hours. 190/2.5 = 76. "
        "The answer is 76 km/h",
    ]

    final, votes, all_answers = self_consistency(simulated_responses)

    print("Exercise 4: Self-Consistency Implementation")
    print("=" * 60)

    print(f"\nQuestion: Average speed for 120km/1.5h then 80km/1h")
    print(f"\nReasoning paths ({len(simulated_responses)} samples):")
    for i, (response, answer) in enumerate(zip(simulated_responses, all_answers)):
        print(f"\n  Path {i + 1}: {response[:70]}...")
        print(f"  Extracted answer: {answer}")

    print(f"\nVote counts: {votes}")
    print(f"Final answer (majority): {final}")
    print(f"Confidence: {votes[final]}/{len(simulated_responses)} paths agree")

    print("\nWhy this works: Temperature > 0 causes the model to explore")
    print("different reasoning paths. Incorrect paths tend to disagree,")
    print("while correct paths consistently converge on the same answer.")


# === Exercise 5: Prompt Template Evaluation ===
# Problem: Identify weaknesses in a code review prompt and write an
# improved version.

def exercise_5():
    """Prompt template evaluation and improvement."""

    original_prompt = 'Review this code: {code}'

    improved_prompt = """You are a senior software engineer conducting a thorough code review.

Review the following {language} code and provide feedback organized by severity.

Code to review:
```{language}
{code}
```

Provide your review in this exact format:

## Critical Issues (must fix before merging)
- [issue]: [explanation and suggested fix]

## Warnings (should fix)
- [issue]: [explanation and suggested fix]

## Suggestions (nice to have)
- [issue]: [explanation and suggested fix]

## Summary
[2-3 sentence overall assessment including what the code does well]"""

    print("Exercise 5: Prompt Template Evaluation")
    print("=" * 60)

    print("\n--- Original (weak) prompt ---")
    print(f'  "{original_prompt}"')

    print("\n--- Problems identified ---")
    problems = [
        "1. No role/expertise context -- model doesn't know what reviewer to be",
        "2. No output structure -- response format is unpredictable",
        "3. No review criteria -- security? performance? style? all of them?",
        "4. No language specification -- different languages have different practices",
        "5. No severity guidance -- critical bugs not distinguished from nits",
    ]
    for p in problems:
        print(f"  {p}")

    print("\n--- Improved prompt ---")
    print(improved_prompt)

    # Demonstrate with example code
    test_code = '''def get_user(id):
    return db.execute(f"SELECT * FROM users WHERE id = {id}")'''

    print("\n--- Test with example code ---")
    print(f"  Code: {test_code}")
    print(f"  Expected: Critical issue flagged for SQL injection vulnerability")

    print("\nThe improved prompt enforces consistent structure, focuses the")
    print("model's expertise, and separates critical issues from minor")
    print("suggestions -- making the output actionable for developers.")


if __name__ == "__main__":
    print("=== Exercise 1: Zero-shot vs Few-shot ===")
    exercise_1()
    print("\n=== Exercise 2: Chain-of-Thought Prompt ===")
    exercise_2()
    print("\n=== Exercise 3: Structured Output Extraction ===")
    exercise_3()
    print("\n=== Exercise 4: Self-Consistency ===")
    exercise_4()
    print("\n=== Exercise 5: Prompt Template Evaluation ===")
    exercise_5()
    print("\nAll exercises completed!")
