# Exercise: Lesson 04 — Constitutional AI
# Complete the TODO items below.
#
# Run: python 04_constitutional_ai.py


def write_constitution(domain: str) -> list[dict]:
    """Write a set of constitutional principles for an AI system.

    Args:
        domain: The application domain (e.g., "medical", "education",
                "customer_service", "general_assistant").

    Returns:
        List of principle dicts with keys:
            - "id": principle identifier (e.g., "P1")
            - "category": one of "harmlessness", "helpfulness", "honesty"
            - "text": the principle statement
            - "priority": int (1 = highest priority)
    """
    # TODO: Define at least 5 principles covering harmlessness,
    # helpfulness, and honesty categories.

    # TODO: Tailor principles to the given domain (e.g., medical domain
    # should emphasize "do not provide diagnoses" and "defer to doctors").

    # TODO: Assign priority rankings and return.
    pass


def critique_response(response: str, principles: list[dict]) -> list[dict]:
    """Critique an AI response against constitutional principles.

    Args:
        response: The AI-generated response text.
        principles: List of principle dicts (from write_constitution).

    Returns:
        List of critique dicts with:
            - "principle_id": which principle was evaluated
            - "violated": bool
            - "severity": one of "minor", "moderate", "severe"
            - "explanation": why the principle was or was not violated
    """
    # TODO: For each principle, check if the response violates it.
    # Use keyword matching or rule-based heuristics.

    # TODO: Assess severity of any violations.

    # TODO: Provide a clear explanation for each evaluation.
    pass


def revise_response(response: str, critiques: list[dict]) -> str:
    """Revise a response based on constitutional critiques.

    Args:
        response: The original AI response.
        critiques: List of critique dicts (from critique_response).

    Returns:
        A revised response string that addresses all violations.
    """
    # TODO: Identify all violated principles from the critiques.

    # TODO: For each violation, determine how to modify the response
    # (remove harmful content, add caveats, soften language, etc.).

    # TODO: Apply modifications and return the revised response.
    pass


def run_critique_loop(response: str, principles: list[dict],
                      max_iterations: int = 3) -> dict:
    """Run the full Constitutional AI critique-revision loop.

    Args:
        response: Initial AI response.
        principles: Constitutional principles.
        max_iterations: Maximum number of critique-revision rounds.

    Returns:
        dict with "final_response", "iterations_used" (int),
        "revision_history" (list of intermediate responses),
        "all_clear" (bool - True if no violations remain).
    """
    # TODO: Repeatedly critique and revise the response until either
    # no violations remain or max_iterations is reached.

    # TODO: Track revision history for transparency.

    # TODO: Return the final result with metadata.
    pass


if __name__ == "__main__":
    # Test write_constitution
    principles = write_constitution("medical")
    print(f"Constitution ({len(principles) if principles else 0} principles):")
    if principles:
        for p in principles:
            print(f"  {p}")

    # Test critique_response
    response = "You should take 500mg of ibuprofen immediately for your headache."
    if principles:
        critiques = critique_response(response, principles)
        print(f"\nCritiques: {critiques}")

        # Test revise_response
        revised = revise_response(response, critiques)
        print(f"\nRevised: {revised}")

        # Test full loop
        result = run_critique_loop(response, principles)
        print(f"\nLoop result: {result}")
    else:
        print("Implement write_constitution first.")
