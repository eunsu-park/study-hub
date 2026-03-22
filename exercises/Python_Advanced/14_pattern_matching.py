"""
Exercises for Lesson 14: Pattern Matching
Topic: Python

Solutions to practice problems from the lesson.
Requires Python 3.10+ for match/case syntax.
"""

from dataclasses import dataclass
from typing import Optional
import math


# === Exercise 1: Shape Area Calculator ===
# Problem: Write a function that calculates areas of various shapes.

@dataclass
class Circle:
    radius: float


@dataclass
class Rectangle:
    width: float
    height: float


@dataclass
class Triangle:
    base: float
    height: float


@dataclass
class Trapezoid:
    top: float
    bottom: float
    height: float


Shape = Circle | Rectangle | Triangle | Trapezoid


def calculate_area(shape: Shape) -> float:
    """Calculate the area of a shape using structural pattern matching.

    Pattern matching on dataclass instances extracts fields by name,
    making the code both readable and exhaustive. Adding a new shape
    only requires adding one more case branch.
    """
    match shape:
        case Circle(radius=r):
            return math.pi * r ** 2
        case Rectangle(width=w, height=h):
            return w * h
        case Triangle(base=b, height=h):
            return 0.5 * b * h
        case Trapezoid(top=t, bottom=b, height=h):
            return 0.5 * (t + b) * h
        case _:
            raise ValueError(f"Unknown shape: {shape}")


def exercise_1():
    """Demonstrate shape area calculator."""
    shapes = [
        Circle(5),
        Rectangle(4, 5),
        Triangle(6, 4),
        Trapezoid(3, 7, 4),
    ]

    for shape in shapes:
        area = calculate_area(shape)
        print(f"  {shape.__class__.__name__}: area = {area:.2f}")


# === Exercise 2: HTTP Request Router ===
# Problem: Implement a simple HTTP request router.

def route_request(method: str, path: str) -> str:
    """Route HTTP requests using pattern matching on method and path segments.

    Splits the path by '/' and matches against known route patterns.
    Captures dynamic segments (like user_id) as variables directly
    in the pattern -- no need for regex or manual parsing.
    """
    match (method, path.split("/")):
        case ("GET", ["", ""]):
            return "200 OK: Home page"

        case ("GET", ["", "users"]):
            return "200 OK: List all users"

        case ("GET", ["", "users", user_id]):
            return f"200 OK: Get user {user_id}"

        case ("GET", ["", "users", user_id, "posts"]):
            return f"200 OK: List posts for user {user_id}"

        case ("POST", ["", "users"]):
            return "201 Created: New user"

        case ("PUT", ["", "users", user_id]):
            return f"200 OK: Updated user {user_id}"

        case ("DELETE", ["", "users", user_id]):
            return f"200 OK: Deleted user {user_id}"

        case ("GET", ["", "health"]):
            return "200 OK: Healthy"

        case (method, _):
            return f"404 Not Found: {method} {path}"


def exercise_2():
    """Demonstrate HTTP request router."""
    requests = [
        ("GET", "/"),
        ("GET", "/users"),
        ("GET", "/users/123"),
        ("GET", "/users/123/posts"),
        ("POST", "/users"),
        ("PUT", "/users/456"),
        ("DELETE", "/users/789"),
        ("GET", "/health"),
        ("GET", "/nonexistent"),
        ("PATCH", "/users/1"),
    ]

    for method, path in requests:
        response = route_request(method, path)
        print(f"  {method:6s} {path:20s} -> {response}")


# === Exercise 3: Recursive Tree Traversal ===
# Problem: Traverse a tree structure using pattern matching.

@dataclass
class TreeNode:
    value: int
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


def sum_tree(node: Optional[TreeNode]) -> int:
    """Sum all values in a binary tree using pattern matching.

    The match/case on None vs TreeNode(...) makes the base case and
    recursive case visually distinct. Pattern matching destructures
    the node in one step, avoiding separate attribute access.
    """
    match node:
        case None:
            return 0
        case TreeNode(value=v, left=l, right=r):
            return v + sum_tree(l) + sum_tree(r)


def find_max(node: Optional[TreeNode]) -> Optional[int]:
    """Find the maximum value in a binary tree using pattern matching."""
    match node:
        case None:
            return None
        case TreeNode(value=v, left=None, right=None):
            # Leaf node -- its value is the only candidate
            return v
        case TreeNode(value=v, left=l, right=r):
            left_max = find_max(l)
            right_max = find_max(r)
            candidates = [v]
            if left_max is not None:
                candidates.append(left_max)
            if right_max is not None:
                candidates.append(right_max)
            return max(candidates)


def tree_depth(node: Optional[TreeNode]) -> int:
    """Calculate the depth of a binary tree using pattern matching."""
    match node:
        case None:
            return 0
        case TreeNode(left=l, right=r):
            return 1 + max(tree_depth(l), tree_depth(r))


def exercise_3():
    """Demonstrate recursive tree traversal with pattern matching."""
    #       1
    #      / \
    #     2   3
    #    / \   \
    #   4   5   6
    tree = TreeNode(
        1,
        TreeNode(2, TreeNode(4), TreeNode(5)),
        TreeNode(3, None, TreeNode(6)),
    )

    print(f"  Tree sum: {sum_tree(tree)}")       # 1+2+3+4+5+6 = 21
    print(f"  Tree max: {find_max(tree)}")       # 6
    print(f"  Tree depth: {tree_depth(tree)}")   # 3

    # Empty tree
    print(f"  Empty tree sum: {sum_tree(None)}")    # 0
    print(f"  Empty tree max: {find_max(None)}")    # None
    print(f"  Empty tree depth: {tree_depth(None)}")  # 0

    # Single node
    single = TreeNode(42)
    print(f"  Single node sum: {sum_tree(single)}")    # 42
    print(f"  Single node max: {find_max(single)}")    # 42
    print(f"  Single node depth: {tree_depth(single)}")  # 1


if __name__ == "__main__":
    print("=== Exercise 1: Shape Area Calculator ===")
    exercise_1()

    print("\n=== Exercise 2: HTTP Request Router ===")
    exercise_2()

    print("\n=== Exercise 3: Recursive Tree Traversal ===")
    exercise_3()

    print("\nAll exercises completed!")
