# Exercise: Lesson 06 — Scalable Oversight
# Complete the TODO items below.
#
# Run: python 06_scalable_oversight.py


def build_debate_protocol(question: str, num_rounds: int = 3) -> dict:
    """Build an AI safety debate protocol for a given question.

    In the debate framework, two AI agents argue opposing positions
    while a human judge evaluates. This simulates scalable oversight.

    Args:
        question: The question or claim to debate.
        num_rounds: Number of argument rounds.

    Returns:
        dict with:
            - "question": the original question
            - "rounds": list of round dicts with "round_num",
              "pro_argument_template", "con_argument_template",
              "judge_criteria"
            - "final_judgment_rubric": list of evaluation criteria
    """
    # TODO: Create argument templates for each round that escalate
    # in depth (round 1: opening claims, round 2: rebuttals,
    # round 3: final summaries).

    # TODO: Define judge criteria for each round (relevance,
    # evidence quality, logical consistency).

    # TODO: Create the final judgment rubric.
    pass


def decompose_task(task: str, max_depth: int = 3) -> dict:
    """Recursively decompose a complex task into verifiable subtasks.

    This implements the recursive reward modeling / task decomposition
    approach to scalable oversight.

    Args:
        task: High-level task description.
        max_depth: Maximum decomposition depth.

    Returns:
        Nested dict with "task", "subtasks" (list of nested dicts),
        "verifiable" (bool - True if leaf node),
        "verification_method" (str for leaf nodes).
    """
    # TODO: Determine if the task is simple enough to verify directly.

    # TODO: If not, break it into 2-4 subtasks that are easier to verify.

    # TODO: Recursively decompose subtasks until max_depth or verifiable.

    # TODO: Assign verification methods to leaf tasks.
    pass


def evaluate_debate_round(pro_arg: str, con_arg: str,
                          criteria: list[str]) -> dict:
    """Evaluate a single debate round using specified criteria.

    Args:
        pro_arg: The argument in favor.
        con_arg: The argument against.
        criteria: List of evaluation criteria strings.

    Returns:
        dict with "scores" (dict mapping criterion to {"pro": float, "con": float}),
        "round_winner" ("pro" or "con"),
        "reasoning" (str).
    """
    # TODO: Score each argument against each criterion (0-10 scale).
    # Use heuristics like argument length, keyword presence, structure.

    # TODO: Determine the round winner based on aggregate scores.

    # TODO: Provide reasoning for the judgment.
    pass


def market_based_oversight(predictions: list[dict]) -> dict:
    """Simulate a prediction market for AI oversight decisions.

    Args:
        predictions: List of dicts with "agent_id", "prediction" (str),
                     "confidence" (float 0-1), "track_record" (float 0-1).

    Returns:
        dict with "consensus_prediction", "aggregated_confidence",
        "weights_used" (dict of agent_id -> weight),
        "disagreement_score" (float 0-1).
    """
    # TODO: Weight each agent's prediction by track_record and confidence.

    # TODO: Aggregate predictions (e.g., weighted majority vote).

    # TODO: Compute disagreement score as variance in confidence levels.

    # TODO: Return the consensus result.
    pass


if __name__ == "__main__":
    # Test debate protocol
    protocol = build_debate_protocol("Should AI systems be given internet access?")
    print(f"Debate protocol: {protocol}")

    # Test task decomposition
    tree = decompose_task("Verify that this research paper's claims are accurate")
    print(f"\nTask decomposition: {tree}")

    # Test debate evaluation
    result = evaluate_debate_round(
        pro_arg="Internet access enables real-time knowledge and better answers.",
        con_arg="Internet access creates security risks and misinformation vectors.",
        criteria=["logical_consistency", "evidence_quality", "safety_awareness"]
    )
    print(f"\nDebate evaluation: {result}")

    # Test market-based oversight
    predictions = [
        {"agent_id": "A1", "prediction": "safe", "confidence": 0.8, "track_record": 0.9},
        {"agent_id": "A2", "prediction": "unsafe", "confidence": 0.6, "track_record": 0.7},
        {"agent_id": "A3", "prediction": "safe", "confidence": 0.9, "track_record": 0.85},
    ]
    market = market_based_oversight(predictions)
    print(f"\nMarket oversight: {market}")
