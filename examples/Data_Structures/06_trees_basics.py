"""
06 Trees Basics
===============
Demonstrates binary tree construction, traversals,
and tree property computation.
"""

from collections import deque


class TreeNode:
    """A node in a binary tree."""
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def preorder(node):
    if node is None: return []
    return [node.val] + preorder(node.left) + preorder(node.right)

def inorder(node):
    if node is None: return []
    return inorder(node.left) + [node.val] + inorder(node.right)

def postorder(node):
    if node is None: return []
    return postorder(node.left) + postorder(node.right) + [node.val]

def level_order(root):
    if root is None: return []
    result, queue = [], deque([root])
    while queue:
        node = queue.popleft()
        result.append(node.val)
        if node.left: queue.append(node.left)
        if node.right: queue.append(node.right)
    return result

def height(node):
    if node is None: return -1
    return 1 + max(height(node.left), height(node.right))

def size(node):
    if node is None: return 0
    return 1 + size(node.left) + size(node.right)

def count_leaves(node):
    if node is None: return 0
    if not node.left and not node.right: return 1
    return count_leaves(node.left) + count_leaves(node.right)

def print_tree(node, level=0, prefix="Root: "):
    if node is not None:
        print(" " * (level * 4) + prefix + str(node.val))
        if node.left or node.right:
            print_tree(node.left, level + 1, "L--- ")
            print_tree(node.right, level + 1, "R--- ")


if __name__ == "__main__":
    root = TreeNode(1,
        TreeNode(2, TreeNode(4), TreeNode(5, TreeNode(7))),
        TreeNode(3, None, TreeNode(6)))

    print("Tree structure:")
    print_tree(root)
    print(f"\nPreorder:    {preorder(root)}")
    print(f"Inorder:     {inorder(root)}")
    print(f"Postorder:   {postorder(root)}")
    print(f"Level-order: {level_order(root)}")
    print(f"\nHeight: {height(root)}")
    print(f"Size:   {size(root)}")
    print(f"Leaves: {count_leaves(root)}")
