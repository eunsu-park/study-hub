# Exercise: Lesson 02 — Alignment Problem
# Complete the TODO items below.
#
# Run: python 02_alignment_problem.py


def detect_reward_hacking(trajectory: list[dict]) -> dict:
    """Detect reward hacking in an agent's trajectory.

    Args:
        trajectory: List of dicts with keys "state", "action",
                    "reward", "intended_outcome", "actual_outcome".

    Returns:
        dict with keys:
            - "is_hacking": bool
            - "hacking_steps": list of step indices where hacking occurred
            - "pattern": description of the hacking pattern found
    """
    # TODO: Compare intended_outcome vs actual_outcome for each step.
    # If reward is high but actual_outcome diverges from intended_outcome,
    # flag it as reward hacking.

    # TODO: Identify the pattern (e.g., "reward gaming via shortcut",
    # "exploiting measurement flaw", "specification loophole").

    # TODO: Return the detection result.
    pass


def identify_specification_gaming(spec: dict, behaviors: list[str]) -> list[dict]:
    """Identify specification gaming in observed agent behaviors.

    Args:
        spec: dict with "objective" (str) and "constraints" (list of str).
        behaviors: List of observed behavior descriptions.

    Returns:
        List of dicts with "behavior", "is_gaming" (bool),
        "violated_spirit" (str or None), "technically_valid" (bool).
    """
    # TODO: For each behavior, check if it technically satisfies the
    # specification constraints while violating the spirit of the objective.

    # TODO: Determine whether the behavior is technically valid
    # (meets letter of spec) but gaming (violates intent).

    # TODO: Return analysis for each behavior.
    pass


def analyze_goodhart_failure(metric: str, proxy_values: list[float],
                              true_values: list[float]) -> dict:
    """Analyze Goodhart's Law failure where optimizing a proxy diverges
    from the true objective.

    Args:
        metric: Name of the proxy metric being optimized.
        proxy_values: Sequence of proxy metric values over time.
        true_values: Sequence of true objective values over time.

    Returns:
        dict with "divergence_point" (index where proxy and true diverge),
        "correlation_before" (float), "correlation_after" (float),
        "failure_type" (one of "regressional", "extremal", "causal", "adversarial").
    """
    # TODO: Calculate correlation between proxy and true values
    # for the full sequence.

    # TODO: Find the divergence point where correlation breaks down
    # (sliding window approach).

    # TODO: Classify the Goodhart failure type based on the divergence
    # pattern.

    # TODO: Return the analysis.
    pass


def propose_alignment_fix(problem: dict) -> dict:
    """Propose an alignment fix for a given misalignment problem.

    Args:
        problem: dict with "description", "type" (reward_hacking,
                 specification_gaming, goal_misgeneralization),
                 and "severity" (low, medium, high).

    Returns:
        dict with "fix_type", "description", "tradeoffs" (list of str),
        and "confidence" (float 0-1).
    """
    # TODO: Based on problem type, select appropriate fix strategy
    # (e.g., reward shaping, constrained optimization, human oversight).

    # TODO: Describe the fix and its expected impact.

    # TODO: Identify tradeoffs (e.g., performance cost, scalability).

    # TODO: Return the proposal.
    pass


if __name__ == "__main__":
    # Test detect_reward_hacking
    trajectory = [
        {"state": "s0", "action": "normal", "reward": 1.0,
         "intended_outcome": "clean room", "actual_outcome": "clean room"},
        {"state": "s1", "action": "cover_mess", "reward": 1.0,
         "intended_outcome": "clean room", "actual_outcome": "mess hidden under rug"},
    ]
    print(f"Reward hacking: {detect_reward_hacking(trajectory)}")

    # Test identify_specification_gaming
    spec = {"objective": "maximize user engagement",
            "constraints": ["no explicit content", "respond to all queries"]}
    behaviors = ["generates clickbait", "provides helpful answers",
                 "creates artificial urgency"]
    print(f"Spec gaming: {identify_specification_gaming(spec, behaviors)}")

    # Test analyze_goodhart_failure
    proxy = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
    true = [1.0, 1.9, 2.8, 3.0, 2.5, 1.0]
    print(f"Goodhart: {analyze_goodhart_failure('engagement_time', proxy, true)}")

    # Test propose_alignment_fix
    problem = {"description": "Agent finds shortcut that inflates metric",
               "type": "reward_hacking", "severity": "high"}
    print(f"Fix proposal: {propose_alignment_fix(problem)}")
