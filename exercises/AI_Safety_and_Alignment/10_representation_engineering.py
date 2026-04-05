# Exercise: Lesson 10 — Representation Engineering
# Complete the TODO items below.
#
# Run: python 10_representation_engineering.py

import math


def train_linear_probe(activations: list[list[float]],
                       labels: list[int],
                       learning_rate: float = 0.01,
                       epochs: int = 50) -> dict:
    """Train a linear probe to detect a safety-relevant concept in activations.

    Args:
        activations: List of activation vectors (list of float).
        labels: Binary labels (1 = concept present, 0 = absent).
        learning_rate: Gradient descent step size.
        epochs: Number of training passes.

    Returns:
        dict with "weights" (list of float), "bias" (float),
        "final_accuracy" (float), "loss_history" (list of float).
    """
    # TODO: Initialize weights to zeros and bias to zero.

    # TODO: For each epoch, compute predictions using sigmoid(w . x + b).

    # TODO: Compute binary cross-entropy loss and gradients.

    # TODO: Update weights and bias via gradient descent.

    # TODO: Return trained probe parameters and training history.
    pass


def find_safety_direction(safe_activations: list[list[float]],
                          unsafe_activations: list[list[float]]) -> list[float]:
    """Find the "safety direction" in activation space using mean difference.

    The safety direction is the vector pointing from the centroid of
    unsafe activations to the centroid of safe activations.

    Args:
        safe_activations: Activation vectors for safe examples.
        unsafe_activations: Activation vectors for unsafe examples.

    Returns:
        Normalized direction vector (list of float).
    """
    # TODO: Compute the centroid (mean) of safe activations.

    # TODO: Compute the centroid (mean) of unsafe activations.

    # TODO: Compute the difference vector (safe_mean - unsafe_mean).

    # TODO: Normalize to unit length and return.
    pass


def project_onto_direction(activation: list[float],
                           direction: list[float]) -> float:
    """Project an activation vector onto a direction to get a scalar score.

    Args:
        activation: The activation vector.
        direction: The direction vector (should be unit length).

    Returns:
        Scalar projection value (dot product).
    """
    # TODO: Compute and return the dot product of activation and direction.
    pass


def steer_activation(activation: list[float], direction: list[float],
                     strength: float = 1.0) -> list[float]:
    """Steer an activation vector along the safety direction.

    Args:
        activation: Original activation vector.
        direction: Safety direction vector (unit length).
        strength: How much to shift along the direction (positive = safer).

    Returns:
        Modified activation vector.
    """
    # TODO: Add strength * direction to the activation vector.

    # TODO: Return the steered activation.
    pass


def evaluate_probe(probe: dict, test_activations: list[list[float]],
                   test_labels: list[int]) -> dict:
    """Evaluate a trained linear probe on test data.

    Args:
        probe: dict with "weights" (list of float) and "bias" (float).
        test_activations: Test activation vectors.
        test_labels: Ground truth binary labels.

    Returns:
        dict with "accuracy" (float), "precision" (float),
        "recall" (float), "f1_score" (float).
    """
    # TODO: Compute predictions using sigmoid(w . x + b) > 0.5.

    # TODO: Calculate true positives, false positives, true negatives,
    # false negatives.

    # TODO: Compute accuracy, precision, recall, and F1 score.
    pass


if __name__ == "__main__":
    # Test data: 2D activations for simplicity
    safe_acts = [[1.0, 2.0], [1.5, 2.5], [0.8, 1.8]]
    unsafe_acts = [[-1.0, -0.5], [-1.5, -1.0], [-0.8, -0.3]]
    all_acts = safe_acts + unsafe_acts
    labels = [1, 1, 1, 0, 0, 0]

    # Test linear probe
    probe = train_linear_probe(all_acts, labels)
    print(f"Probe: {probe}")

    # Test safety direction
    direction = find_safety_direction(safe_acts, unsafe_acts)
    print(f"\nSafety direction: {direction}")

    # Test projection
    if direction:
        score = project_onto_direction([0.5, 1.0], direction)
        print(f"Projection score: {score}")

        # Test steering
        steered = steer_activation([-1.0, -0.5], direction, strength=2.0)
        print(f"Steered activation: {steered}")

    # Test probe evaluation
    if probe:
        eval_result = evaluate_probe(probe, all_acts, labels)
        print(f"\nProbe evaluation: {eval_result}")
