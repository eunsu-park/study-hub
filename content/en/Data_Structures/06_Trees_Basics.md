# Trees Basics

**Previous**: [Hash Tables](./05_Hash_Tables.md) | **Next**: [Binary Search Trees](./07_Binary_Search_Trees.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define tree terminology: root, leaf, parent, child, height, depth, subtree
2. Explain the difference between general trees and binary trees
3. Implement a binary tree using linked nodes
4. Perform all four traversal orders: inorder, preorder, postorder, level-order
5. Convert between recursive and iterative traversal implementations
6. Calculate tree properties: height, size, leaf count
7. Understand how trees model hierarchical data in real systems

---

A **tree** is a hierarchical data structure consisting of **nodes** connected by **edges**. Unlike linear structures (arrays, linked lists, stacks, queues), trees represent relationships where each element can have multiple children, forming a branching structure.

## Tree Terminology

```
                    [A]  <-- Root (depth 0, level 0)
                   / | \
                 /   |   \
              [B]   [C]   [D]  <-- depth 1
             / \          |
           [E] [F]       [G]   <-- depth 2
               / \
             [H] [I]          <-- depth 3 (leaves)

Height of tree = 3 (longest path from root to leaf)
```

| Term | Definition |
|------|-----------|
| **Root** | The topmost node (no parent) |
| **Leaf** | A node with no children |
| **Internal node** | A node with at least one child |
| **Parent** | The node directly above |
| **Child** | A node directly below |
| **Sibling** | Nodes sharing the same parent |
| **Depth** | Number of edges from root to node |
| **Height** | Number of edges on the longest path from node to a leaf |
| **Level** | Same as depth (root is level 0) |
| **Subtree** | A node and all its descendants |
| **Degree** | Number of children a node has |

### Formal Definition

A tree is a connected, acyclic graph. Equivalently:
- A tree with n nodes has exactly n-1 edges
- There is exactly one path between any two nodes
- Removing any edge disconnects the tree

## Binary Trees

A **binary tree** is a tree where each node has at most **two children**: left and right.

```
          [1]
         /   \
       [2]   [3]
      /   \     \
    [4]   [5]   [6]
         /
       [7]
```

### Node Implementation

```python
class TreeNode:
    """A node in a binary tree."""
    
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"
```

### Building a Tree

```python
# Build the tree shown above
root = TreeNode(1,
    TreeNode(2,
        TreeNode(4),
        TreeNode(5,
            TreeNode(7),
            None
        )
    ),
    TreeNode(3,
        None,
        TreeNode(6)
    )
)
```

## Types of Binary Trees

```
Full Binary Tree:        Complete Binary Tree:    Perfect Binary Tree:
Every node has 0         All levels filled        All leaves at same
or 2 children            except possibly last     depth, all internal
                         (filled left to right)    nodes have 2 children

      [1]                      [1]                      [1]
     /   \                    /   \                    /   \
   [2]   [3]               [2]   [3]               [2]   [3]
  / \                     / \   /                  / \   / \
[4] [5]                 [4] [5][6]               [4] [5][6] [7]

Degenerate (Skewed):    Balanced:
Every node has 1 child  Height = O(log n)

[1]                          [4]
  \                         /   \
  [2]                     [2]   [6]
    \                    / \   / \
    [3]               [1] [3][5] [7]
      \
      [4]
```

### Binary Tree Properties

| Property | Formula |
|----------|---------|
| Max nodes at level k | 2^k |
| Max nodes in tree of height h | 2^(h+1) - 1 |
| Min height for n nodes | floor(log2(n)) |
| Max height for n nodes | n - 1 (degenerate) |
| Leaves in a full binary tree | (n + 1) / 2 |

## Tree Traversals

Traversal means visiting every node exactly once. There are four standard orders:

### Preorder (Root, Left, Right) -- NLR

```
          [1]
         /   \         Visit order: 1, 2, 4, 5, 3, 6
       [2]   [3]
      /   \     \
    [4]   [5]   [6]
```

```python
def preorder(node):
    """Preorder traversal: Root -> Left -> Right."""
    if node is None:
        return []
    return [node.val] + preorder(node.left) + preorder(node.right)

def preorder_iterative(root):
    """Iterative preorder using an explicit stack."""
    if root is None:
        return []
    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        result.append(node.val)
        if node.right:  # Push right first (LIFO)
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result
```

### Inorder (Left, Root, Right) -- LNR

```
          [1]
         /   \         Visit order: 4, 2, 5, 1, 3, 6
       [2]   [3]
      /   \     \
    [4]   [5]   [6]
```

```python
def inorder(node):
    """Inorder traversal: Left -> Root -> Right."""
    if node is None:
        return []
    return inorder(node.left) + [node.val] + inorder(node.right)

def inorder_iterative(root):
    """Iterative inorder using a stack."""
    result = []
    stack = []
    current = root
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.val)
        current = current.right
    return result
```

### Postorder (Left, Right, Root) -- LRN

```
          [1]
         /   \         Visit order: 4, 5, 2, 6, 3, 1
       [2]   [3]
      /   \     \
    [4]   [5]   [6]
```

```python
def postorder(node):
    """Postorder traversal: Left -> Right -> Root."""
    if node is None:
        return []
    return postorder(node.left) + postorder(node.right) + [node.val]

def postorder_iterative(root):
    """Iterative postorder using two stacks."""
    if root is None:
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
    return result[::-1]  # Reverse of modified preorder
```

### Level-Order (Breadth-First)

```
          [1]
         /   \         Visit order: 1, 2, 3, 4, 5, 6
       [2]   [3]       Level 0: [1]
      /   \     \       Level 1: [2, 3]
    [4]   [5]   [6]    Level 2: [4, 5, 6]
```

```python
from collections import deque

def level_order(root):
    """Level-order (BFS) traversal."""
    if root is None:
        return []
    result = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        result.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result

def level_order_by_level(root):
    """Return values grouped by level."""
    if root is None:
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
```

### Traversal Summary

```
Given tree:        Preorder:   1 2 4 5 3 6  (Root first -- for copying)
      [1]          Inorder:    4 2 5 1 3 6  (Sorted order for BST)
     /   \         Postorder:  4 5 2 6 3 1  (Root last -- for deletion)
   [2]   [3]       Level-order: 1 2 3 4 5 6 (Top to bottom, left to right)
  /   \     \
[4]   [5]   [6]
```

## Tree Properties -- Recursive Computation

```python
def height(node):
    """Compute the height of a binary tree."""
    if node is None:
        return -1  # Height of empty tree is -1
    return 1 + max(height(node.left), height(node.right))

def size(node):
    """Count the total number of nodes."""
    if node is None:
        return 0
    return 1 + size(node.left) + size(node.right)

def count_leaves(node):
    """Count the number of leaf nodes."""
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)

def is_balanced(node):
    """Check if the tree is height-balanced (heights differ by at most 1)."""
    def check(node):
        if node is None:
            return 0
        left_h = check(node.left)
        right_h = check(node.right)
        if left_h == -1 or right_h == -1 or abs(left_h - right_h) > 1:
            return -1
        return 1 + max(left_h, right_h)
    return check(node) != -1
```

## Tree Serialization

Converting a tree to a string (and back) for storage or transmission:

```python
def serialize(root):
    """Serialize a binary tree to a string."""
    if root is None:
        return "null"
    return f"{root.val},{serialize(root.left)},{serialize(root.right)}"

def deserialize(data):
    """Deserialize a string to a binary tree."""
    tokens = iter(data.split(","))
    
    def build():
        val = next(tokens)
        if val == "null":
            return None
        node = TreeNode(int(val))
        node.left = build()
        node.right = build()
        return node
    
    return build()
```

## Real-World Trees

| Application | Tree Type |
|------------|-----------|
| File systems | General tree (directories/files) |
| HTML/XML DOM | General tree |
| Database indexes | B-trees, B+ trees |
| Compilers (AST) | Binary/general trees |
| Decision making | Decision trees |
| Huffman coding | Binary tree |
| Network routing | Spanning trees |

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Tree | Hierarchical, connected, acyclic graph |
| Binary tree | Each node has at most 2 children |
| Full vs Complete vs Perfect | Structural properties with different guarantees |
| Preorder | Root-Left-Right (copying/serialization) |
| Inorder | Left-Root-Right (sorted order in BST) |
| Postorder | Left-Right-Root (deletion/evaluation) |
| Level-order | BFS using a queue |
| Height | Longest path to leaf, computed recursively |

---

**Next**: [Binary Search Trees](./07_Binary_Search_Trees.md) -- Add the ordering property for efficient search.
