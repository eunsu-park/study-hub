"""
Exercises for Lesson 09: Trees and BST
Topic: Algorithm

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Lowest Common Ancestor in BST ===
# Problem: Find the lowest common ancestor of two nodes in a Binary Search Tree.
# Approach: In a BST, if both target values are smaller than current node, go left;
#   if both are larger, go right; otherwise the current node is the LCA.

def exercise_1():
    """Solution: BST LCA in O(h) time."""
    class TreeNode:
        def __init__(self, val, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def lowest_common_ancestor(root, p_val, q_val):
        """Find LCA of nodes with values p_val and q_val in BST."""
        node = root
        while node:
            if p_val < node.val and q_val < node.val:
                # Both values in left subtree
                node = node.left
            elif p_val > node.val and q_val > node.val:
                # Both values in right subtree
                node = node.right
            else:
                # Values split here (or one equals current node)
                # This is the LCA
                return node
        return None

    # Build BST:
    #         6
    #        / \
    #       2   8
    #      / \ / \
    #     0  4 7  9
    #       / \
    #      3   5
    root = TreeNode(6,
        TreeNode(2,
            TreeNode(0),
            TreeNode(4,
                TreeNode(3),
                TreeNode(5)
            )
        ),
        TreeNode(8,
            TreeNode(7),
            TreeNode(9)
        )
    )

    # Test cases
    tests = [
        (2, 8, 6),    # LCA of 2 and 8 is 6
        (2, 4, 2),    # LCA of 2 and 4 is 2 (ancestor is one of the nodes)
        (3, 5, 4),    # LCA of 3 and 5 is 4
        (0, 5, 2),    # LCA of 0 and 5 is 2
        (7, 9, 8),    # LCA of 7 and 9 is 8
        (0, 9, 6),    # LCA of 0 and 9 is 6 (root)
    ]

    for p_val, q_val, expected_val in tests:
        result = lowest_common_ancestor(root, p_val, q_val)
        print(f"LCA({p_val}, {q_val}) = {result.val}")
        assert result.val == expected_val, f"Expected {expected_val}, got {result.val}"

    print("All BST LCA tests passed!")


if __name__ == "__main__":
    print("=== Exercise 1: Lowest Common Ancestor in BST ===")
    exercise_1()
    print("\nAll exercises completed!")
