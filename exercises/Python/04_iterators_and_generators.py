"""
Exercises for Lesson 04: Iterators and Generators
Topic: Python

Solutions to practice problems from the lesson.
"""

from typing import Generator, Iterator, Optional, Any
from collections.abc import Iterable


# === Exercise 1: Chunk Division ===
# Problem: Create a generator that divides a list into chunks of specified size.
# chunk([1,2,3,4,5], 2) -> [1,2], [3,4], [5]

def chunk(data: list, size: int) -> Generator[list, None, None]:
    """Yield successive chunks of `size` from `data`.

    Uses range stepping to slice the list. The last chunk may be
    shorter than `size` if the list length is not evenly divisible.
    """
    for i in range(0, len(data), size):
        yield data[i:i + size]


def exercise_1():
    """Demonstrate chunk generator."""
    data = [1, 2, 3, 4, 5]
    print(f"chunk({data}, 2):")
    for c in chunk(data, 2):
        print(f"  {c}")

    print(f"\nchunk({list(range(10))}, 3):")
    for c in chunk(list(range(10)), 3):
        print(f"  {c}")

    # Edge case: chunk size larger than the list
    print(f"\nchunk([1, 2], 5):")
    for c in chunk([1, 2], 5):
        print(f"  {c}")


# === Exercise 2: Sliding Window ===
# Problem: Create a generator that produces sliding windows.
# sliding_window([1,2,3,4,5], 3) -> (1,2,3), (2,3,4), (3,4,5)

def sliding_window(data: list, window_size: int) -> Generator[tuple, None, None]:
    """Yield sliding windows of `window_size` over `data`.

    Each window is a tuple containing `window_size` consecutive elements.
    The number of windows is len(data) - window_size + 1.
    """
    if window_size > len(data):
        return  # No valid windows possible
    for i in range(len(data) - window_size + 1):
        yield tuple(data[i:i + window_size])


def exercise_2():
    """Demonstrate sliding window generator."""
    data = [1, 2, 3, 4, 5]
    print(f"sliding_window({data}, 3):")
    for window in sliding_window(data, 3):
        print(f"  {window}")

    print(f"\nsliding_window({data}, 1):")
    for window in sliding_window(data, 1):
        print(f"  {window}")

    # Edge case: window size equals list length
    print(f"\nsliding_window({data}, 5):")
    for window in sliding_window(data, 5):
        print(f"  {window}")


# === Exercise 3: Tree Traversal ===
# Problem: Create a generator that traverses a binary tree.

class TreeNode:
    """Simple binary tree node."""

    def __init__(self, value: Any, left: Optional["TreeNode"] = None,
                 right: Optional["TreeNode"] = None):
        self.value = value
        self.left = left
        self.right = right


def inorder(node: Optional[TreeNode]) -> Generator[Any, None, None]:
    """In-order traversal: left -> root -> right.

    Using 'yield from' delegates to recursive calls, keeping the
    generator protocol clean. For a BST, this produces sorted output.
    """
    if node is None:
        return
    yield from inorder(node.left)
    yield node.value
    yield from inorder(node.right)


def preorder(node: Optional[TreeNode]) -> Generator[Any, None, None]:
    """Pre-order traversal: root -> left -> right."""
    if node is None:
        return
    yield node.value
    yield from preorder(node.left)
    yield from preorder(node.right)


def postorder(node: Optional[TreeNode]) -> Generator[Any, None, None]:
    """Post-order traversal: left -> right -> root."""
    if node is None:
        return
    yield from postorder(node.left)
    yield from postorder(node.right)
    yield node.value


def exercise_3():
    """Demonstrate tree traversal generators."""
    #       4
    #      / \
    #     2   6
    #    / \ / \
    #   1  3 5  7
    tree = TreeNode(
        4,
        TreeNode(2, TreeNode(1), TreeNode(3)),
        TreeNode(6, TreeNode(5), TreeNode(7)),
    )

    print(f"In-order:   {list(inorder(tree))}")    # [1, 2, 3, 4, 5, 6, 7]
    print(f"Pre-order:  {list(preorder(tree))}")   # [4, 2, 1, 3, 6, 5, 7]
    print(f"Post-order: {list(postorder(tree))}")  # [1, 3, 2, 5, 7, 6, 4]


if __name__ == "__main__":
    print("=== Exercise 1: Chunk Division ===")
    exercise_1()

    print("\n=== Exercise 2: Sliding Window ===")
    exercise_2()

    print("\n=== Exercise 3: Tree Traversal ===")
    exercise_3()

    print("\nAll exercises completed!")
