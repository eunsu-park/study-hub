"""
Exercises for Lesson 10: LangChain Basics
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


# === Exercise 1: LCEL Chain Composition ===
# Problem: Build a two-step LCEL chain that first identifies the programming
# language in a code snippet, then explains what the code does.

def exercise_1():
    """LCEL chain composition: language detection + code explanation."""

    # Simulated chain components (no API calls)
    # In real LangChain: ChatPromptTemplate | ChatOpenAI | StrOutputParser

    def detect_language(code):
        """Simulated language detector based on syntax heuristics."""
        code_lower = code.lower().strip()
        if 'def ' in code_lower and ':' in code_lower:
            return "Python"
        elif 'function ' in code_lower or '=>' in code_lower:
            return "JavaScript"
        elif '#include' in code_lower:
            return "C/C++"
        elif 'public static void' in code_lower:
            return "Java"
        elif 'fn ' in code_lower and '->' in code_lower:
            return "Rust"
        return "Unknown"

    def explain_code(code, language):
        """Simulated code explainer (in production, this would call an LLM)."""
        explanations = {
            "fibonacci": (
                f"This {language} function computes the nth Fibonacci number "
                "using recursion. It handles the base cases (n=0 returns 0, "
                "n=1 returns 1) and recursively sums the two preceding "
                "Fibonacci numbers."
            ),
            "binary_search": (
                f"This {language} function performs binary search on a sorted array. "
                "It maintains left/right pointers and narrows the search range by "
                "comparing the target with the midpoint element."
            ),
        }
        for key, explanation in explanations.items():
            if key in code.lower().replace("_", ""):
                return explanation
        return f"This {language} code performs a computation (full analysis requires LLM)."

    # Simulated LCEL chain using function composition
    class SimulatedLCELChain:
        """
        Demonstrates the LCEL pattern:
          RunnableParallel(language=detect_chain, code=passthrough)
          | explain_prompt
          | llm
          | output_parser
        """
        def invoke(self, inputs):
            code = inputs["code"]
            # Step 1: Detect language (parallel with code passthrough)
            language = detect_language(code)
            # Step 2: Explain code using detected language
            explanation = explain_code(code, language)
            return {
                "language": language,
                "explanation": explanation,
            }

    chain = SimulatedLCELChain()

    # Test with multiple code snippets
    test_snippets = [
        {
            "code": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
        },
        {
            "code": """
function binarySearch(arr, target) {
    let left = 0, right = arr.length - 1;
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
""",
        },
    ]

    print("Exercise 1: LCEL Chain Composition")
    print("=" * 60)

    print("\nLCEL Chain Architecture:")
    print("  Input: {code}")
    print("    |")
    print("    +-- RunnableParallel --+")
    print("    |   detect_language    |  RunnablePassthrough (code)")
    print("    +----------+----------+")
    print("               |")
    print("    {language, code}")
    print("               |")
    print("    explain_prompt | llm | StrOutputParser")
    print("               |")
    print("    Output: explanation")

    for i, snippet in enumerate(test_snippets):
        result = chain.invoke(snippet)
        print(f"\n--- Snippet {i + 1} ---")
        print(f"  Code: {snippet['code'].strip()[:60]}...")
        print(f"  Detected language: {result['language']}")
        print(f"  Explanation: {result['explanation']}")

    print("\nKey LCEL concepts:")
    print("  - RunnableParallel: runs sub-chains simultaneously")
    print("  - RunnablePassthrough: passes input unchanged to next step")
    print("  - The | operator chains components: prompt | llm | parser")
    print("  - Output of RunnableParallel is a dict matching prompt variables")


# === Exercise 2: Pydantic Output Parser ===
# Problem: Create a chain that extracts structured product information
# from unstructured text using a Pydantic model.

def exercise_2():
    """Pydantic output parser for structured product extraction."""

    @dataclass
    class Product:
        """Pydantic-like product model (using dataclass for self-containment)."""
        name: str
        price: float
        in_stock: bool
        features: List[str]

        def validate(self):
            """Basic validation (Pydantic does this automatically)."""
            if not isinstance(self.name, str) or not self.name:
                raise ValueError("name must be a non-empty string")
            if not isinstance(self.price, (int, float)) or self.price < 0:
                raise ValueError("price must be a non-negative number")
            if not isinstance(self.in_stock, bool):
                raise ValueError("in_stock must be a boolean")
            if not isinstance(self.features, list):
                raise ValueError("features must be a list")
            return True

    def extract_product(text):
        """
        Simulated product extraction (in production, LLM + JsonOutputParser).

        In real LangChain:
            parser = JsonOutputParser(pydantic_object=Product)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract product info. {format_instructions}"),
                ("human", "{text}")
            ]).partial(format_instructions=parser.get_format_instructions())
            chain = prompt | llm | parser
        """
        # Simple rule-based extraction for demonstration
        text_lower = text.lower()

        # Extract product name (first capitalized phrase)
        import re
        name_match = re.search(r'(?:The |the )?([A-Z][A-Za-z0-9\s]+?)(?:\s+is|\s+comes)', text)
        name = name_match.group(1).strip() if name_match else "Unknown Product"

        # Extract price
        price_match = re.search(r'\$([0-9,]+\.?\d*)', text)
        price = float(price_match.group(1).replace(',', '')) if price_match else 0.0

        # Check stock status
        in_stock = any(phrase in text_lower for phrase in
                       ['in stock', 'available', 'ready to ship'])

        # Extract features (look for comma-separated items after keywords)
        features = []
        feature_patterns = [
            r'(?:include|highlights|features)[s]?\s+(?:a\s+)?(.+?)(?:\.|$)',
            r'(\d+(?:-\w+)?\s+\w+(?:\s+\w+)?)',  # "15-inch 4K display"
        ]
        # Simple: split on commas and "and"
        feature_section = re.search(
            r'(?:include|highlights|features)[s]?\s+(.+?)(?:\.\s|$)',
            text, re.IGNORECASE
        )
        if feature_section:
            raw = feature_section.group(1)
            parts = re.split(r',\s*(?:and\s+)?', raw)
            features = [p.strip().strip('.') for p in parts if p.strip()]

        return Product(name=name, price=price, in_stock=in_stock, features=features)

    # Test texts
    test_texts = [
        (
            "The UltraBook Pro 15 is now available for $1,299.99. This laptop is "
            "in stock and ready to ship. Key highlights include a 15-inch 4K display, "
            "16GB RAM, 512GB NVMe SSD, and 18-hour battery life."
        ),
        (
            "The SoundWave Max comes with impressive specs for $299.99. Currently "
            "in stock. Features include noise cancellation, 40-hour battery, "
            "and Bluetooth 5.3 connectivity."
        ),
    ]

    print("Exercise 2: Pydantic Output Parser")
    print("=" * 60)

    print("\nFormat instructions (what Pydantic generates):")
    print('  {"name": str, "price": float, "in_stock": bool, "features": [str]}')

    for i, text in enumerate(test_texts):
        product = extract_product(text)
        product.validate()

        print(f"\n--- Product {i + 1} ---")
        print(f"  Input: \"{text[:70]}...\"")
        print(f"  Extracted:")
        print(f"    Name:     {product.name}")
        print(f"    Price:    ${product.price:.2f}")
        print(f"    In stock: {product.in_stock}")
        print(f"    Features: {product.features}")

    print(f"\nWhy Pydantic output parsing matters:")
    print(f"  - JsonOutputParser with Pydantic provides schema validation")
    print(f"  - If the LLM returns 'price': 'expensive' instead of a float,")
    print(f"    Pydantic raises a validation error")
    print(f"  - This makes output reliable for downstream processing")
    print(f"  - format_instructions auto-generates the JSON schema for the prompt")


# === Exercise 3: RunnableWithMessageHistory ===
# Problem: Implement multi-turn customer support using
# RunnableWithMessageHistory pattern.

def exercise_3():
    """RunnableWithMessageHistory for multi-turn conversation."""

    class ChatMessageHistory:
        """Simulated in-memory message history store."""
        def __init__(self):
            self.messages = []

        def add_user_message(self, content):
            self.messages.append({"role": "user", "content": content})

        def add_ai_message(self, content):
            self.messages.append({"role": "assistant", "content": content})

        def get_messages(self):
            return self.messages.copy()

    class SimulatedSupportBot:
        """
        Simulated RunnableWithMessageHistory pattern.

        In real LangChain:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful customer support agent..."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])
            chain = prompt | llm | StrOutputParser()
            chatbot = RunnableWithMessageHistory(
                chain, get_session_history,
                input_messages_key="input",
                history_messages_key="history",
            )
        """
        def __init__(self):
            self.store: Dict[str, ChatMessageHistory] = {}

        def get_session_history(self, session_id):
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            return self.store[session_id]

        def generate_response(self, user_input, history):
            """Simulated LLM response based on conversation history."""
            all_text = " ".join(
                m["content"] for m in history
            ) + " " + user_input
            all_lower = all_text.lower()

            # Extract order number from context
            import re
            order_match = re.search(r'#(\d+)', all_text)
            order_num = order_match.group(1) if order_match else None

            # Simple response logic
            if "hasn't arrived" in all_lower or "not arrived" in all_lower:
                return (
                    f"I'm sorry to hear your order hasn't arrived yet. "
                    f"{'I can see order #' + order_num + '. ' if order_num else ''}"
                    f"Let me look into this for you. Can I help you track it?"
                )
            elif "track" in all_lower:
                return (
                    f"I'll track {'order #' + order_num if order_num else 'your order'} "
                    f"right away. It shows it's currently in transit and should "
                    f"arrive within 2-3 business days."
                )
            elif "lost" in all_lower:
                return (
                    f"If {'order #' + order_num if order_num else 'your order'} "
                    f"doesn't arrive within 5 business days, we can issue a full "
                    f"refund or send a replacement. Would you like me to set a reminder?"
                )
            else:
                return "How can I help you today?"

        def invoke(self, inputs, session_id):
            """Main entry point with session management."""
            history = self.get_session_history(session_id)
            user_input = inputs["input"]

            # Generate response using history context
            response = self.generate_response(
                user_input, history.get_messages()
            )

            # Store messages
            history.add_user_message(user_input)
            history.add_ai_message(response)

            return response

    bot = SimulatedSupportBot()
    session_id = "user_42"

    # Simulate multi-turn conversation
    conversation = [
        "My order #12345 hasn't arrived yet.",
        "Can you track it?",
        "What if it's lost?",
    ]

    print("Exercise 3: RunnableWithMessageHistory")
    print("=" * 60)

    print(f"\nSession ID: {session_id}")
    print(f"System prompt: 'You are a helpful customer support agent...'")

    for turn, user_msg in enumerate(conversation, 1):
        response = bot.invoke({"input": user_msg}, session_id=session_id)
        print(f"\n  Turn {turn}:")
        print(f"    User: {user_msg}")
        print(f"    Bot:  {response}")

    # Show stored history
    history = bot.get_session_history(session_id)
    print(f"\n--- Stored message history ---")
    print(f"  Total messages: {len(history.messages)} "
          f"({len(conversation)} user + {len(conversation)} assistant)")
    for msg in history.messages:
        role = msg["role"]
        content = msg["content"][:60]
        print(f"    [{role}] {content}...")

    print(f"\nKey points:")
    print(f"  - MessagesPlaceholder injects all previous messages at that position")
    print(f"  - session_id allows multiple isolated conversations")
    print(f"  - In production, replace in-memory store with Redis/PostgreSQL")
    print(f"  - History grows with every turn -- use summary memory for long chats")


# === Exercise 4: LangChain vs LangGraph Selection ===
# Problem: For each scenario, decide whether to use LCEL chains or LangGraph.

def exercise_4():
    """LangChain LCEL vs LangGraph selection for different scenarios."""

    scenarios = [
        {
            "scenario": "A. Summarize a PDF document",
            "choice": "LCEL",
            "justification": (
                "Linear workflow: load -> split -> summarize. No cycles or "
                "complex state needed. chain = loader | splitter | "
                "summarize_prompt | llm | parser"
            ),
            "architecture": "PDF -> TextSplitter -> SummarizePrompt -> LLM -> Output",
        },
        {
            "scenario": "B. Web browsing agent (search until answer found)",
            "choice": "LangGraph",
            "justification": (
                "Requires cycles (search -> evaluate -> search again if needed). "
                "State must persist across iterations. Conditional edges decide "
                "when to stop."
            ),
            "architecture": (
                "SearchNode -> EvaluateNode -> [conditional: "
                "found? -> END, not found? -> SearchNode]"
            ),
        },
        {
            "scenario": "C. Translate text to 5 languages in parallel",
            "choice": "LCEL",
            "justification": (
                "RunnableParallel handles this perfectly: "
                "{fr: fr_chain, de: de_chain, ja: ja_chain, ko: ko_chain, "
                "es: es_chain} -- all in one call."
            ),
            "architecture": (
                "Input -> RunnableParallel(fr=chain, de=chain, ja=chain, "
                "ko=chain, es=chain) -> {5 translations}"
            ),
        },
        {
            "scenario": "D. Code review pipeline (loop until tests pass)",
            "choice": "LangGraph",
            "justification": (
                "Requires a cycle: review -> run tests -> if tests fail, "
                "route back to review. Stateful graph tracks iteration count "
                "and test results."
            ),
            "architecture": (
                "ReviewNode -> TestNode -> [conditional: "
                "pass? -> END, fail? -> ReviewNode, max_iter? -> END]"
            ),
        },
    ]

    print("Exercise 4: LangChain LCEL vs LangGraph Selection")
    print("=" * 70)

    for s in scenarios:
        print(f"\n  {s['scenario']}")
        print(f"  Choice: {s['choice']}")
        print(f"  Why: {s['justification']}")
        print(f"  Architecture: {s['architecture']}")

    # Decision framework
    print(f"\n{'='*70}")
    print(f"Decision Framework:")
    print(f"{'='*70}")

    criteria = [
        ("Linear data flow (A -> B -> C)", "LCEL", "Simple pipe operator |"),
        ("Parallel execution", "LCEL", "RunnableParallel"),
        ("Cycles / loops", "LangGraph", "Conditional edges back to nodes"),
        ("Complex state management", "LangGraph", "TypedDict state across nodes"),
        ("Conditional branching", "Either", "LCEL: RunnableBranch; LangGraph: edges"),
        ("Human-in-the-loop", "LangGraph", "Interrupt nodes, checkpoint/resume"),
    ]

    print(f"\n  {'Pattern':<40} {'Use':<12} {'Notes'}")
    print(f"  {'-'*40} {'-'*12} {'-'*35}")
    for pattern, use, notes in criteria:
        print(f"  {pattern:<40} {use:<12} {notes}")

    # Demonstrate LangGraph state structure
    print(f"\n--- LangGraph State Example (Scenario B) ---")
    print("""
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, List

    class SearchState(TypedDict):
        question: str
        search_results: List[str]
        answer: str
        iterations: int

    graph = StateGraph(SearchState)
    graph.add_node("search", search_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_edge("search", "evaluate")
    graph.add_conditional_edges(
        "evaluate", should_continue,
        {"search": "search", END: END}
    )
    graph.set_entry_point("search")
    app = graph.compile()
    """)

    print("Key takeaway: Use LCEL for DAGs (directed acyclic graphs),")
    print("LangGraph when you need cycles, complex state, or human-in-the-loop.")


if __name__ == "__main__":
    print("=== Exercise 1: LCEL Chain Composition ===")
    exercise_1()
    print("\n=== Exercise 2: Pydantic Output Parser ===")
    exercise_2()
    print("\n=== Exercise 3: RunnableWithMessageHistory ===")
    exercise_3()
    print("\n=== Exercise 4: LangChain vs LangGraph Selection ===")
    exercise_4()
    print("\nAll exercises completed!")
