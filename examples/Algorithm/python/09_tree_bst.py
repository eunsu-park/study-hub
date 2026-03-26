"""
Tree and Binary Search Tree (Tree & BST)
Tree and Binary Search Tree

Implements tree structures and BST operations.
"""

from typing import List, Optional, Generator
from collections import deque


# =============================================================================
# 1. Binary Tree Node
# =============================================================================

class TreeNode:
    """Binary tree node"""

    def __init__(self, val: int, left: 'TreeNode' = None, right: 'TreeNode' = None):
        self.val = val
        self.left = left
        self.right = right

    def __repr__(self):
        return f"TreeNode({self.val})"


# =============================================================================
# 2. Tree Traversal
# =============================================================================

def preorder_recursive(root: TreeNode) -> List[int]:
    """Preorder traversal (recursive) - O(n)"""
    result = []

    def traverse(node):
        if not node:
            return
        result.append(node.val)
        traverse(node.left)
        traverse(node.right)

    traverse(root)
    return result


def preorder_iterative(root: TreeNode) -> List[int]:
    """Preorder traversal (iterative) - O(n)"""
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        # Push right first (so left is processed first)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result


def inorder_recursive(root: TreeNode) -> List[int]:
    """Inorder traversal (recursive) - O(n)"""
    result = []

    def traverse(node):
        if not node:
            return
        traverse(node.left)
        result.append(node.val)
        traverse(node.right)

    traverse(root)
    return result


def inorder_iterative(root: TreeNode) -> List[int]:
    """Inorder traversal (iterative) - O(n)"""
    result = []
    stack = []
    current = root

    while stack or current:
        # Go to the leftmost node
        while current:
            stack.append(current)
            current = current.left

        current = stack.pop()
        result.append(current.val)
        current = current.right

    return result


def postorder_recursive(root: TreeNode) -> List[int]:
    """Postorder traversal (recursive) - O(n)"""
    result = []

    def traverse(node):
        if not node:
            return
        traverse(node.left)
        traverse(node.right)
        result.append(node.val)

    traverse(root)
    return result


def postorder_iterative(root: TreeNode) -> List[int]:
    """Postorder traversal (iterative) - O(n)"""
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

    return result[::-1]  # Reverse


def level_order(root: TreeNode) -> List[List[int]]:
    """Level-order traversal (BFS) - O(n)"""
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result


# =============================================================================
# 3. Binary Search Tree (BST)
# =============================================================================

class BST:
    """Binary Search Tree"""

    def __init__(self):
        self.root: Optional[TreeNode] = None

    def insert(self, val: int) -> None:
        """Insert node - Average O(log n), Worst O(n)"""
        if not self.root:
            self.root = TreeNode(val)
            return

        current = self.root
        while True:
            if val < current.val:
                if current.left is None:
                    current.left = TreeNode(val)
                    return
                current = current.left
            else:
                if current.right is None:
                    current.right = TreeNode(val)
                    return
                current = current.right

    def search(self, val: int) -> Optional[TreeNode]:
        """Search node - Average O(log n), Worst O(n)"""
        current = self.root

        while current:
            if val == current.val:
                return current
            elif val < current.val:
                current = current.left
            else:
                current = current.right

        return None

    def delete(self, val: int) -> bool:
        """Delete node - Average O(log n), Worst O(n)"""

        def find_min(node: TreeNode) -> TreeNode:
            while node.left:
                node = node.left
            return node

        def delete_recursive(node: TreeNode, val: int) -> Optional[TreeNode]:
            if not node:
                return None

            if val < node.val:
                node.left = delete_recursive(node.left, val)
            elif val > node.val:
                node.right = delete_recursive(node.right, val)
            else:
                # Found the node to delete

                # Case 1: Leaf node
                if not node.left and not node.right:
                    return None

                # Case 2: One child
                if not node.left:
                    return node.right
                if not node.right:
                    return node.left

                # Case 3: Two children - replace with successor (min of right subtree)
                successor = find_min(node.right)
                node.val = successor.val
                node.right = delete_recursive(node.right, successor.val)

            return node

        old_root = self.root
        self.root = delete_recursive(self.root, val)
        return old_root != self.root or (self.root and old_root.val != val if old_root else False)

    def inorder(self) -> List[int]:
        """Inorder traversal (sorted order)"""
        return inorder_recursive(self.root)

    def find_min(self) -> Optional[int]:
        """Find minimum value - O(h)"""
        if not self.root:
            return None

        current = self.root
        while current.left:
            current = current.left
        return current.val

    def find_max(self) -> Optional[int]:
        """Find maximum value - O(h)"""
        if not self.root:
            return None

        current = self.root
        while current.right:
            current = current.right
        return current.val


# =============================================================================
# 4. Tree Property Checks
# =============================================================================

def tree_height(root: TreeNode) -> int:
    """Calculate tree height - O(n)"""
    if not root:
        return -1  # Empty tree has height -1, single node has height 0

    return 1 + max(tree_height(root.left), tree_height(root.right))


def is_balanced(root: TreeNode) -> bool:
    """Check if tree is balanced - O(n)"""

    def check(node: TreeNode) -> int:
        if not node:
            return 0

        left_height = check(node.left)
        if left_height == -1:
            return -1

        right_height = check(node.right)
        if right_height == -1:
            return -1

        if abs(left_height - right_height) > 1:
            return -1

        return 1 + max(left_height, right_height)

    return check(root) != -1


def is_valid_bst(root: TreeNode) -> bool:
    """Check if tree is a valid BST - O(n)"""

    def validate(node: TreeNode, min_val: float, max_val: float) -> bool:
        if not node:
            return True

        if node.val <= min_val or node.val >= max_val:
            return False

        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))

    return validate(root, float('-inf'), float('inf'))


def count_nodes(root: TreeNode) -> int:
    """Count nodes - O(n)"""
    if not root:
        return 0
    return 1 + count_nodes(root.left) + count_nodes(root.right)


# =============================================================================
# 5. Tree Transformation/Construction
# =============================================================================

def build_tree_from_list(values: List[Optional[int]]) -> Optional[TreeNode]:
    """Build tree from level-order list - O(n)"""
    if not values or values[0] is None:
        return None

    root = TreeNode(values[0])
    queue = deque([root])
    i = 1

    while queue and i < len(values):
        node = queue.popleft()

        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1

        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1

    return root


def sorted_array_to_bst(nums: List[int]) -> Optional[TreeNode]:
    """Build balanced BST from sorted array - O(n)"""
    if not nums:
        return None

    def build(left: int, right: int) -> Optional[TreeNode]:
        if left > right:
            return None

        mid = (left + right) // 2
        node = TreeNode(nums[mid])
        node.left = build(left, mid - 1)
        node.right = build(mid + 1, right)
        return node

    return build(0, len(nums) - 1)


def invert_tree(root: TreeNode) -> TreeNode:
    """Invert tree (mirror) - O(n)"""
    if not root:
        return None

    root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root


# =============================================================================
# 6. Practical Problems
# =============================================================================

def lowest_common_ancestor(root: TreeNode, p: int, q: int) -> Optional[TreeNode]:
    """Lowest Common Ancestor (LCA) in BST - O(h)"""
    current = root

    while current:
        if p < current.val and q < current.val:
            current = current.left
        elif p > current.val and q > current.val:
            current = current.right
        else:
            return current

    return None


def kth_smallest(root: TreeNode, k: int) -> int:
    """Kth smallest value in BST - O(h + k)"""
    stack = []
    current = root
    count = 0

    while stack or current:
        while current:
            stack.append(current)
            current = current.left

        current = stack.pop()
        count += 1

        if count == k:
            return current.val

        current = current.right

    return -1


def path_sum(root: TreeNode, target: int) -> bool:
    """Check root-to-leaf path sum - O(n)"""
    if not root:
        return False

    if not root.left and not root.right:
        return root.val == target

    remaining = target - root.val
    return path_sum(root.left, remaining) or path_sum(root.right, remaining)


def serialize(root: TreeNode) -> str:
    """Serialize tree - O(n)"""
    if not root:
        return "[]"

    result = []
    queue = deque([root])

    while queue:
        node = queue.popleft()
        if node:
            result.append(str(node.val))
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append("null")

    # Remove trailing nulls
    while result and result[-1] == "null":
        result.pop()

    return "[" + ",".join(result) + "]"


# =============================================================================
# Utility: Tree Visualization
# =============================================================================

def print_tree(root: TreeNode, prefix: str = "", is_left: bool = True) -> None:
    """Print tree in ASCII"""
    if not root:
        return

    print(prefix + ("├── " if is_left else "└── ") + str(root.val))

    children = []
    if root.left:
        children.append((root.left, True))
    if root.right:
        children.append((root.right, False))

    for i, (child, is_left_child) in enumerate(children):
        extension = "│   " if is_left and i < len(children) - 1 else "    "
        print_tree(child, prefix + extension, is_left_child)


# =============================================================================
# Tests
# =============================================================================

def main():
    print("=" * 60)
    print("Tree and Binary Search Tree (Tree & BST) Examples")
    print("=" * 60)

    # 1. Tree Construction
    print("\n[1] Tree Construction")
    #       4
    #      / \
    #     2   6
    #    / \ / \
    #   1  3 5  7
    root = build_tree_from_list([4, 2, 6, 1, 3, 5, 7])
    print("    Level order: [4, 2, 6, 1, 3, 5, 7]")
    print("    Tree structure:")
    print_tree(root, "    ")

    # 2. Tree Traversal
    print("\n[2] Tree Traversal")
    print(f"    Preorder:  {preorder_recursive(root)}")
    print(f"    Inorder:   {inorder_recursive(root)}")
    print(f"    Postorder: {postorder_recursive(root)}")
    print(f"    Level:     {level_order(root)}")

    # 3. Tree Properties
    print("\n[3] Tree Properties")
    print(f"    Height: {tree_height(root)}")
    print(f"    Node count: {count_nodes(root)}")
    print(f"    Balanced: {is_balanced(root)}")
    print(f"    Valid BST: {is_valid_bst(root)}")

    # 4. BST Operations
    print("\n[4] BST Operations")
    bst = BST()
    for val in [5, 3, 7, 1, 4, 6, 8]:
        bst.insert(val)
    print(f"    Insert: [5, 3, 7, 1, 4, 6, 8]")
    print(f"    Inorder traversal: {bst.inorder()}")
    print(f"    Search 4: {bst.search(4)}")
    print(f"    Min: {bst.find_min()}, Max: {bst.find_max()}")

    bst.delete(3)
    print(f"    After delete 3: {bst.inorder()}")

    # 5. Sorted Array -> Balanced BST
    print("\n[5] Sorted Array -> Balanced BST")
    arr = [1, 2, 3, 4, 5, 6, 7]
    balanced_bst = sorted_array_to_bst(arr)
    print(f"    Input: {arr}")
    print(f"    Level order: {level_order(balanced_bst)}")

    # 6. LCA
    print("\n[6] Lowest Common Ancestor (LCA)")
    lca = lowest_common_ancestor(root, 1, 3)
    print(f"    LCA of nodes 1, 3: {lca.val if lca else None}")
    lca = lowest_common_ancestor(root, 1, 6)
    print(f"    LCA of nodes 1, 6: {lca.val if lca else None}")

    # 7. Kth Smallest Value
    print("\n[7] Kth Smallest Value")
    for k in [1, 3, 5]:
        print(f"    {k}th smallest: {kth_smallest(root, k)}")

    # 8. Path Sum
    print("\n[8] Root-to-Leaf Path Sum")
    print(f"    Sum 7 (4->2->1): {path_sum(root, 7)}")
    print(f"    Sum 10 (4->6): {path_sum(root, 10)}")

    # 9. Serialization
    print("\n[9] Tree Serialization")
    print(f"    {serialize(root)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
