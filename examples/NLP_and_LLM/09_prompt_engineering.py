"""
08. Prompt Engineering Example

Various prompting techniques and optimization strategies
"""

print("=" * 60)
print("Prompt Engineering")
print("=" * 60)


# ============================================
# 1. Prompt Template Class
# ============================================
print("\n[1] Prompt Template")
print("-" * 40)

class PromptTemplate:
    """Reusable prompt template"""

    def __init__(self, template: str, input_variables: list = None):
        self.template = template
        self.input_variables = input_variables or []

    def format(self, **kwargs) -> str:
        """Fill variables to generate a prompt"""
        return self.template.format(**kwargs)

    @classmethod
    def from_file(cls, path: str):
        """Load template from file"""
        with open(path, 'r', encoding='utf-8') as f:
            return cls(f.read())

# Basic template
basic_template = PromptTemplate(
    template="""You are a {role}.
Task: {task}
Input: {input}
Output:""",
    input_variables=["role", "task", "input"]
)

prompt = basic_template.format(
    role="helpful assistant",
    task="translate to Korean",
    input="Hello, world!"
)
print("Basic template example:")
print(prompt)


# ============================================
# 2. Zero-shot vs Few-shot
# ============================================
print("\n[2] Zero-shot vs Few-shot")
print("-" * 40)

# Zero-shot prompt
zero_shot = """Analyze the sentiment of the following review.
Review: "This movie was really boring."
Sentiment:"""

print("Zero-shot:")
print(zero_shot)

# Few-shot prompt
few_shot = """Analyze the sentiment of the following review.

Review: "What a really fun movie!"
Sentiment: Positive

Review: "Worst movie ever, a waste of time"
Sentiment: Negative

Review: "It was just okay"
Sentiment: Neutral

Review: "This movie was really boring."
Sentiment:"""

print("\nFew-shot:")
print(few_shot)


# ============================================
# 3. Few-shot Prompt Builder
# ============================================
print("\n[3] Few-shot Prompt Builder")
print("-" * 40)

class FewShotPromptTemplate:
    """Few-shot prompt generator"""

    def __init__(
        self,
        examples: list,
        example_template: str,
        prefix: str = "",
        suffix: str = "",
        separator: str = "\n\n"
    ):
        self.examples = examples
        self.example_template = example_template
        self.prefix = prefix
        self.suffix = suffix
        self.separator = separator

    def format(self, **kwargs) -> str:
        # Format examples
        formatted_examples = [
            self.example_template.format(**ex)
            for ex in self.examples
        ]

        # Combine
        parts = []
        if self.prefix:
            parts.append(self.prefix)
        parts.extend(formatted_examples)
        if self.suffix:
            parts.append(self.suffix.format(**kwargs))

        return self.separator.join(parts)

# Sentiment analysis Few-shot
sentiment_examples = [
    {"text": "I really love it!", "sentiment": "Positive"},
    {"text": "Not great", "sentiment": "Negative"},
    {"text": "It's just average", "sentiment": "Neutral"},
]

sentiment_prompt = FewShotPromptTemplate(
    examples=sentiment_examples,
    example_template="Text: {text}\nSentiment: {sentiment}",
    prefix="Analyze the sentiment of the following text.",
    suffix="Text: {text}\nSentiment:"
)

result = sentiment_prompt.format(text="I'm feeling great today")
print("Few-shot sentiment analysis prompt:")
print(result)


# ============================================
# 4. Chain-of-Thought (CoT)
# ============================================
print("\n[4] Chain-of-Thought (CoT)")
print("-" * 40)

# Zero-shot CoT
zero_shot_cot = """Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
   How many balls does he have now?

Let's think step by step."""

print("Zero-shot CoT:")
print(zero_shot_cot)

# Few-shot CoT
few_shot_cot = """Q: There are 15 trees in the grove. Grove workers plant trees today.
   After they are done, there will be 21 trees. How many trees did they plant?

A: Let's think step by step.
1. Started with 15 trees.
2. After planting, there are 21 trees.
3. Trees planted = 21 - 15 = 6 trees.
The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive,
   how many cars are in the parking lot?

A: Let's think step by step.
1. Started with 3 cars.
2. 2 more cars arrive.
3. Total = 3 + 2 = 5 cars.
The answer is 5.

Q: Roger has 5 tennis balls. He buys 2 more cans of 3 balls each.
   How many balls does he have now?

A: Let's think step by step."""

print("\nFew-shot CoT:")
print(few_shot_cot)


# ============================================
# 5. Role-based Prompting
# ============================================
print("\n[5] Role-based Prompting")
print("-" * 40)

class RolePrompt:
    """Role-based prompt generation"""

    ROLES = {
        "developer": """You are a senior software developer with 10 years of experience.
You write clean, efficient, and well-documented code.
You always consider edge cases and security implications.""",

        "teacher": """You are a patient and encouraging teacher.
You explain complex concepts using simple analogies.
You always check for understanding and provide examples.""",

        "reviewer": """You are a thorough code reviewer.
You check for:
- Code readability
- Potential bugs
- Performance issues
- Security vulnerabilities
You provide constructive feedback.""",

        "translator": """You are a professional translator.
You translate text while preserving:
- Original meaning
- Tone and style
- Cultural context
You provide notes for idiomatic expressions."""
    }

    @classmethod
    def get_system_prompt(cls, role: str) -> str:
        return cls.ROLES.get(role, "You are a helpful assistant.")

    @classmethod
    def create_prompt(cls, role: str, task: str) -> dict:
        return {
            "system": cls.get_system_prompt(role),
            "user": task
        }

# Role prompt example
prompt = RolePrompt.create_prompt(
    role="reviewer",
    task="""Please review the following code:

def get_user(id):
    return db.execute(f"SELECT * FROM users WHERE id = {id}")
"""
)
print("Code reviewer role:")
print(f"System: {prompt['system'][:100]}...")
print(f"User: {prompt['user']}")


# ============================================
# 6. Structured Output Prompts
# ============================================
print("\n[6] Structured Output")
print("-" * 40)

# JSON output prompt
json_prompt = """Extract persons and locations from the following text.

Text: "Alice met Bob in Seoul and they traveled to Busan."

Respond in the following JSON format:
{
  "persons": ["person1", "person2"],
  "locations": ["location1", "location2"]
}"""

print("JSON output prompt:")
print(json_prompt)

# Markdown structured output
markdown_prompt = """Analyze the following article.

## Summary
(Summarize in 2-3 sentences)

## Key Points
- Point 1
- Point 2
- Point 3

## Sentiment
(Choose from Positive/Negative/Neutral)

## Confidence
(Choose from High/Medium/Low, explain the reason)"""

print("\nMarkdown structured output:")
print(markdown_prompt)


# ============================================
# 7. Output Parser
# ============================================
print("\n[7] Output Parser")
print("-" * 40)

import json
import re
from typing import Any, Optional

class OutputParser:
    """LLM output parsing"""

    @staticmethod
    def parse_json(text: str) -> Optional[dict]:
        """Extract and parse JSON"""
        # Find JSON block
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, text, re.DOTALL)

        if match:
            json_str = match.group(1)
        else:
            # Find JSON object directly
            json_pattern = r'\{[^{}]*\}'
            match = re.search(json_pattern, text, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                return None

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def parse_list(text: str) -> list:
        """Extract list items"""
        # Numbered items
        numbered = re.findall(r'^\d+\.\s*(.+)$', text, re.MULTILINE)
        if numbered:
            return numbered

        # Bulleted items
        bulleted = re.findall(r'^[-*]\s*(.+)$', text, re.MULTILINE)
        return bulleted

    @staticmethod
    def parse_key_value(text: str) -> dict:
        """Extract key-value pairs"""
        pattern = r'^([^:]+):\s*(.+)$'
        matches = re.findall(pattern, text, re.MULTILINE)
        return {k.strip(): v.strip() for k, v in matches}

# Test
sample_output = """Analysis results:
- Topic: Artificial Intelligence
- Sentiment: Positive
- Confidence: High

1. First point
2. Second point
3. Third point"""

parser = OutputParser()
print("List parsing:", parser.parse_list(sample_output))
print("Key-value parsing:", parser.parse_key_value(sample_output))


# ============================================
# 8. Self-Consistency
# ============================================
print("\n[8] Self-Consistency")
print("-" * 40)

from collections import Counter

class SelfConsistency:
    """Self-Consistency: Majority vote across multiple reasoning paths"""

    def __init__(self, model_fn, n_samples: int = 5):
        self.model_fn = model_fn
        self.n_samples = n_samples

    def generate_with_consistency(self, prompt: str) -> tuple:
        """Generate multiple samples and take majority vote"""
        responses = []

        for _ in range(self.n_samples):
            # Generate diverse responses with temperature > 0
            response = self.model_fn(prompt, temperature=0.7)
            answer = self._extract_answer(response)
            responses.append(answer)

        # Majority vote
        counter = Counter(responses)
        most_common = counter.most_common(1)[0]

        return most_common[0], most_common[1] / self.n_samples

    def _extract_answer(self, response: str) -> str:
        """Extract final answer from response"""
        # "The answer is X" pattern
        match = re.search(r'answer is[:\s]*(\d+)', response, re.IGNORECASE)
        if match:
            return match.group(1)

        # Last number
        numbers = re.findall(r'\d+', response)
        return numbers[-1] if numbers else response

# Mock function
def mock_model(prompt, temperature=0.7):
    import random
    # In practice, call LLM
    answers = ["42", "42", "42", "41", "42"]
    return f"The answer is {random.choice(answers)}"

sc = SelfConsistency(mock_model, n_samples=5)
print("Self-Consistency example (mock):")
print("Generate multiple reasoning paths and select final answer by majority vote")


# ============================================
# 9. ReAct Pattern
# ============================================
print("\n[9] ReAct (Reasoning + Acting)")
print("-" * 40)

react_prompt = """Answer the following question using this format:

Question: {question}

Thought: (your reasoning about what to do)
Action: (one of: Search[query], Calculate[expression], Lookup[term], Finish[answer])
Observation: (result of the action)

Repeat Thought/Action/Observation until you have the answer.

Example:
Question: What is the capital of the country where the Eiffel Tower is located?

Thought: I need to find where the Eiffel Tower is located.
Action: Search[Eiffel Tower location]
Observation: The Eiffel Tower is located in Paris, France.

Thought: Now I know it's in France. I need to find the capital of France.
Action: Search[capital of France]
Observation: The capital of France is Paris.

Thought: I have the answer now.
Action: Finish[Paris]

Now answer:
Question: {question}
"""

print("ReAct prompt pattern:")
print(react_prompt.format(question="What year was Python created?"))


# ============================================
# 10. Tree of Thoughts
# ============================================
print("\n[10] Tree of Thoughts")
print("-" * 40)

class TreeOfThoughts:
    """Tree of Thoughts: Explore multiple reasoning paths"""

    def __init__(self, model_fn, evaluator_fn):
        self.model_fn = model_fn
        self.evaluator_fn = evaluator_fn

    def solve(self, problem: str, depth: int = 3, branches: int = 3) -> str:
        """Solve problem via tree search"""
        thoughts = self._generate_thoughts(problem, branches)

        # Evaluate each thought
        scored_thoughts = [
            (thought, self.evaluator_fn(problem, thought))
            for thought in thoughts
        ]

        # Select top thoughts
        scored_thoughts.sort(key=lambda x: x[1], reverse=True)
        best_thoughts = scored_thoughts[:2]

        if depth > 1:
            # Recursively expand
            for thought, _ in best_thoughts:
                extended = self.solve(
                    f"{problem}\n\nPartial solution: {thought}",
                    depth - 1,
                    branches
                )
                thoughts.append(extended)

        # Final selection
        return scored_thoughts[0][0]

    def _generate_thoughts(self, problem: str, n: int) -> list:
        """Generate n different approaches"""
        prompt = f"""Problem: {problem}

Generate {n} different approaches to solve this problem.
Each approach should be a distinct strategy.

Approach 1:"""

        # In practice, call LLM
        return [f"Approach {i+1}: ..." for i in range(n)]

print("Tree of Thoughts pattern:")
print("- Explore multiple reasoning paths in a tree structure")
print("- Evaluate each node (thought) and expand promising paths")
print("- Effective for complex reasoning problems")


# ============================================
# 11. Prompt Optimization Strategies
# ============================================
print("\n[11] Prompt Optimization")
print("-" * 40)

optimization_strategies = """
Prompt Optimization Strategies:

1. Clarity
   Bad:  "Clean up the text"
   Good: "Summarize the following text in 3 sentences and extract 5 key keywords."

2. Specificity
   Bad:  "Write good code"
   Good: "Python 3.10+, use type hints, PEP 8 compliant, include error handling"

3. Constraints
   - Output length: "Within 100 words"
   - Output format: "In JSON format"
   - Style: "In a formal tone"

4. Provide Examples
   - Provide 1-3 examples of desired output
   - Clearly convey format and style

5. Step-by-step Decomposition
   - Break complex tasks into smaller steps
   - Clear instructions for each step

6. Negative Prompting
   - Add "Do not..." instructions
   - Prevent unwanted output
"""
print(optimization_strategies)


# ============================================
# 12. Prompt A/B Testing
# ============================================
print("\n[12] Prompt A/B Testing")
print("-" * 40)

class PromptABTest:
    """Prompt A/B testing framework"""

    def __init__(self, model_fn, evaluator_fn):
        self.model_fn = model_fn
        self.evaluator_fn = evaluator_fn

    def run_test(
        self,
        prompt_a: str,
        prompt_b: str,
        test_cases: list,
        n_trials: int = 1
    ) -> dict:
        """Run A/B test"""
        results = {"A": 0, "B": 0, "tie": 0}
        details = []

        for case in test_cases:
            scores_a = []
            scores_b = []

            for _ in range(n_trials):
                # Prompt A
                response_a = self.model_fn(prompt_a.format(**case))
                score_a = self.evaluator_fn(response_a, case.get("expected"))
                scores_a.append(score_a)

                # Prompt B
                response_b = self.model_fn(prompt_b.format(**case))
                score_b = self.evaluator_fn(response_b, case.get("expected"))
                scores_b.append(score_b)

            avg_a = sum(scores_a) / len(scores_a)
            avg_b = sum(scores_b) / len(scores_b)

            if avg_a > avg_b:
                results["A"] += 1
                winner = "A"
            elif avg_b > avg_a:
                results["B"] += 1
                winner = "B"
            else:
                results["tie"] += 1
                winner = "tie"

            details.append({
                "case": case,
                "score_a": avg_a,
                "score_b": avg_b,
                "winner": winner
            })

        return {
            "summary": results,
            "details": details,
            "winner": "A" if results["A"] > results["B"] else "B"
        }

print("Prompt A/B testing framework")
print("- Compare performance of two prompts")
print("- Evaluate across various test cases")
print("- Derive statistically significant results")


# ============================================
# 13. Domain-specific Prompt Templates
# ============================================
print("\n[13] Domain-specific Prompt Templates")
print("-" * 40)

PROMPT_TEMPLATES = {
    "classification": """Classify the following text into one of these categories: {categories}

Text: {text}

Think step by step:
1. Identify key features of the text
2. Match features to categories
3. Select the best category

Category:""",

    "summarization": """Summarize the following text in {num_sentences} sentences.
Focus on the key points and main arguments.
Maintain the original tone.

Text:
{text}

Summary:""",

    "qa": """Answer the question based on the context below.
If the answer cannot be found in the context, say "I don't know."
Do not make up information.

Context: {context}

Question: {question}

Answer:""",

    "code_generation": """Write a {language} function that {task_description}.

Requirements:
{requirements}

Include:
- Type hints (if applicable)
- Docstring
- Example usage
- Error handling

Code:
```{language}
""",

    "translation": """Translate the following {source_lang} text to {target_lang}.
Preserve the original tone and meaning.
For idiomatic expressions, provide a note.

Original ({source_lang}):
{text}

Translation ({target_lang}):""",

    "extraction": """Extract the following information from the text:
{fields}

Text:
{text}

Output as JSON:
{{
{json_template}
}}"""
}

# Usage example
classification_prompt = PROMPT_TEMPLATES["classification"].format(
    categories="Positive, Negative, Neutral",
    text="The weather is really nice today!"
)
print("Classification prompt:")
print(classification_prompt[:200] + "...")


# ============================================
# 14. Prompt Chaining
# ============================================
print("\n[14] Prompt Chaining")
print("-" * 40)

class PromptChain:
    """Chain prompts to perform complex tasks"""

    def __init__(self, model_fn):
        self.model_fn = model_fn
        self.steps = []

    def add_step(self, name: str, prompt_template: str, parser=None):
        """Add a step to the chain"""
        self.steps.append({
            "name": name,
            "template": prompt_template,
            "parser": parser
        })
        return self

    def run(self, initial_input: dict) -> dict:
        """Run the chain"""
        context = initial_input.copy()
        results = {}

        for step in self.steps:
            # Generate prompt
            prompt = step["template"].format(**context)

            # Call LLM
            response = self.model_fn(prompt)

            # Parse (optional)
            if step["parser"]:
                response = step["parser"](response)

            # Save result
            results[step["name"]] = response
            context[step["name"]] = response

        return results

# Chain example
chain = PromptChain(lambda x: "Mock response")
chain.add_step(
    "summary",
    "Summarize this text: {text}"
).add_step(
    "keywords",
    "Extract keywords from: {summary}"
).add_step(
    "title",
    "Create a title based on keywords: {keywords}"
)

print("Prompt chaining example:")
print("1. Text summarization")
print("2. Keyword extraction")
print("3. Title generation")
print("-> Each step's output is used as the next step's input")


# ============================================
# Summary
# ============================================
print("\n" + "=" * 60)
print("Prompt Engineering Summary")
print("=" * 60)

summary = """
Prompting Technique Selection Guide:

| Situation                | Recommended Technique  |
|--------------------------|------------------------|
| Simple task              | Zero-shot              |
| Specific format needed   | Few-shot + format spec |
| Complex reasoning        | Chain-of-Thought       |
| Reliability needed       | Self-Consistency       |
| Tool usage needed        | ReAct                  |
| Very complex problem     | Tree of Thoughts       |

Core Principles:
1. Clear and specific instructions
2. Provide appropriate examples
3. Specify output format
4. Encourage step-by-step thinking
5. Iterative improvement and testing

Prompt Structure:
    [System instruction] + [Context] + [Task] + [Output format]
"""
print(summary)
