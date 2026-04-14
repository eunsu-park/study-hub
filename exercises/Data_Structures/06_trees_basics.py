"""
Exercise 06: Trees Basics

Practice tree traversals and property computation.
"""

from collections import deque


class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def inorder_iterative(root):
    """Iterative inorder traversal using a stack.

    >>> tree = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
    >>> inorder_iterative(tree)
    [4, 2, 5, 1, 3]
    """
    # TODO: Implement this
    pass


def level_order_by_level(root):
    """Return values grouped by level.

    >>> tree = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3, None, TreeNode(6)))
    >>> level_order_by_level(tree)
    [[1], [2, 3], [4, 5, 6]]
    """
    # TODO: Implement this
    pass


def max_depth(root):
    """Return the maximum depth (height) of a binary tree.

    >>> tree = TreeNode(1, TreeNode(2, TreeNode(4)), TreeNode(3))
    >>> max_depth(tree)
    2
    >>> max_depth(None)
    -1
    """
    # TODO: Implement this
    pass


def is_symmetric(root):
    """Check if a binary tree is symmetric (mirror of itself).

    >>> is_symmetric(TreeNode(1, TreeNode(2, TreeNode(3), TreeNode(4)), TreeNode(2, TreeNode(4), TreeNode(3))))
    True
    >>> is_symmetric(TreeNode(1, TreeNode(2, None, TreeNode(3)), TreeNode(2, None, TreeNode(3))))
    False
    """
    # TODO: Implement this
    pass


def count_nodes(root):
    """Count total number of nodes in a binary tree.

    >>> count_nodes(TreeNode(1, TreeNode(2), TreeNode(3)))
    3
    >>> count_nodes(None)
    0
    """
    # TODO: Implement this
    pass


if __name__ == "__main__":
    tree = TreeNode(1,
        TreeNode(2, TreeNode(4), TreeNode(5)),
        TreeNode(3, None, TreeNode(6)))

    assert inorder_iterative(tree) == [4, 2, 5, 1, 3, 6]
    assert inorder_iterative(None) == []
    print("inorder_iterative: PASSED")

    assert level_order_by_level(tree) == [[1], [2, 3], [4, 5, 6]]
    assert level_order_by_level(None) == []
    print("level_order_by_level: PASSED")

    assert max_depth(tree) == 2
    assert max_depth(None) == -1
    print("max_depth: PASSED")

    sym = TreeNode(1, TreeNode(2, TreeNode(3), TreeNode(4)),
                      TreeNode(2, TreeNode(4), TreeNode(3)))
    assert is_symmetric(sym) is True
    assert is_symmetric(tree) is False
    print("is_symmetric: PASSED")

    assert count_nodes(tree) == 6
    assert count_nodes(None) == 0
    print("count_nodes: PASSED")

    print("\nAll tests passed!")
