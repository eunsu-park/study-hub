"""
Exercise 07: Binary Search Trees

Practice BST operations and validation.
"""


class BSTNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def insert(root, val):
    """Insert a value into a BST. Return the root.

    >>> root = None
    >>> for v in [5, 3, 7, 1, 4]: root = insert(root, v)
    >>> inorder(root)
    [1, 3, 4, 5, 7]
    """
    # TODO: Implement this
    pass


def inorder(root):
    """Return inorder traversal as a list."""
    if root is None: return []
    return inorder(root.left) + [root.val] + inorder(root.right)


def search(root, val):
    """Search for a value in BST. Return True/False.

    >>> root = None
    >>> for v in [5, 3, 7]: root = insert(root, v)
    >>> search(root, 3)
    True
    >>> search(root, 4)
    False
    """
    # TODO: Implement this
    pass


def find_min(root):
    """Find the minimum value in BST.

    >>> root = None
    >>> for v in [5, 3, 7, 1, 4]: root = insert(root, v)
    >>> find_min(root)
    1
    """
    # TODO: Implement this
    pass


def delete(root, val):
    """Delete a value from BST. Return the root.

    >>> root = None
    >>> for v in [5, 3, 7, 1, 4]: root = insert(root, v)
    >>> root = delete(root, 3)
    >>> inorder(root)
    [1, 4, 5, 7]
    """
    # TODO: Implement this
    pass


def is_valid_bst(root):
    """Check if a binary tree is a valid BST.

    >>> root = BSTNode(2, BSTNode(1), BSTNode(3))
    >>> is_valid_bst(root)
    True
    >>> root = BSTNode(2, BSTNode(3), BSTNode(1))
    >>> is_valid_bst(root)
    False
    """
    # TODO: Implement this
    pass


def sorted_array_to_bst(nums):
    """Convert a sorted array to a balanced BST.

    >>> root = sorted_array_to_bst([1, 2, 3, 4, 5, 6, 7])
    >>> is_valid_bst(root)
    True
    >>> inorder(root)
    [1, 2, 3, 4, 5, 6, 7]
    """
    # TODO: Implement this
    pass


if __name__ == "__main__":
    root = None
    for v in [5, 3, 7, 1, 4]:
        root = insert(root, v)
    assert inorder(root) == [1, 3, 4, 5, 7]
    print("insert: PASSED")

    assert search(root, 3) is True
    assert search(root, 6) is False
    print("search: PASSED")

    assert find_min(root) == 1
    print("find_min: PASSED")

    root = delete(root, 3)
    assert inorder(root) == [1, 4, 5, 7]
    root = delete(root, 5)
    assert inorder(root) == [1, 4, 7]
    print("delete: PASSED")

    assert is_valid_bst(BSTNode(2, BSTNode(1), BSTNode(3))) is True
    assert is_valid_bst(BSTNode(2, BSTNode(3), BSTNode(1))) is False
    print("is_valid_bst: PASSED")

    bst = sorted_array_to_bst([1, 2, 3, 4, 5, 6, 7])
    assert is_valid_bst(bst)
    assert inorder(bst) == [1, 2, 3, 4, 5, 6, 7]
    print("sorted_array_to_bst: PASSED")

    print("\nAll tests passed!")
