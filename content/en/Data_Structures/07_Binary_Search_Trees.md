# Binary Search Trees

**Previous**: [Trees Basics](./06_Trees_Basics.md) | **Next**: [Heaps](./08_Heaps.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define the BST property and explain why it enables efficient search
2. Implement insert, search, delete, and traversal operations
3. Find the minimum, maximum, successor, and predecessor in a BST
4. Handle all three deletion cases (leaf, one child, two children)
5. Analyze BST performance for balanced and degenerate cases
6. Understand why self-balancing trees (AVL, Red-Black) exist
7. Use Python's `bisect` module for sorted-sequence operations

---

A **Binary Search Tree (BST)** is a binary tree with an ordering property: for every node, all values in its left subtree are less than the node's value, and all values in its right subtree are greater.

## The BST Property

```
          [8]
         /   \
       [3]   [10]
      /   \      \
    [1]   [6]   [14]
         / \    /
       [4] [7][13]

For node [8]: left subtree {1,3,4,6,7} < 8 < right subtree {10,13,14}
For node [3]: left {1} < 3 < right {4,6,7}
For node [10]: no left < 10 < right {13,14}
```

**Key insight**: An inorder traversal of a BST produces values in **sorted order**: 1, 3, 4, 6, 7, 8, 10, 13, 14.

## BST Implementation

```python
class BSTNode:
    """A node in a Binary Search Tree."""
    
    __slots__ = ('val', 'left', 'right')
    
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class BST:
    """Binary Search Tree implementation."""
    
    def __init__(self):
        self.root = None
        self._size = 0
    
    def __len__(self):
        return self._size
```

## Search

Binary search in a tree: compare with current node, go left or right:

```python
def search(self, val):
    """Search for a value -- O(h) where h is height."""
    return self._search(self.root, val)

def _search(self, node, val):
    if node is None:
        return None
    if val == node.val:
        return node
    elif val < node.val:
        return self._search(node.left, val)
    else:
        return self._search(node.right, val)

# Iterative version (avoids stack overhead)
def search_iterative(self, val):
    node = self.root
    while node:
        if val == node.val:
            return node
        elif val < node.val:
            node = node.left
        else:
            node = node.right
    return None
```

```
Search for 6:
          [8]    6 < 8, go left
         /
       [3]      6 > 3, go right
          \
          [6]    Found!

Search for 5:
          [8]    5 < 8, go left
         /
       [3]      5 > 3, go right
          \
          [6]    5 < 6, go left
         /
       [4]      5 > 4, go right -> None (not found)
```

## Insertion

New values are always inserted as leaves:

```python
def insert(self, val):
    """Insert a value -- O(h)."""
    self.root = self._insert(self.root, val)
    self._size += 1

def _insert(self, node, val):
    if node is None:
        return BSTNode(val)
    if val < node.val:
        node.left = self._insert(node.left, val)
    elif val > node.val:
        node.right = self._insert(node.right, val)
    # val == node.val: duplicate, do nothing (or handle as needed)
    return node
```

```
Insert 5 into the BST:
          [8]           [8]
         /   \         /   \
       [3]   [10]    [3]   [10]
      /   \     \   /   \     \
    [1]   [6]  [14][1]  [6]  [14]
         / \   /        / \   /
       [4] [7][13]    [4] [7][13]
                      /
                    [5]  <-- new leaf
```

## Deletion

The most complex BST operation. Three cases:

### Case 1: Deleting a Leaf

Simply remove it:

```
Delete 4:
       [6]         [6]
      / \    -->   / \
    [4] [7]      [?] [7]  -->  [6] - [7]
                                \
```

### Case 2: Node with One Child

Replace the node with its child:

```
Delete 10:
       [8]              [8]
      /   \             /   \
    [3]   [10]   -->  [3]   [14]
              \              /
             [14]          [13]
             /
           [13]
```

### Case 3: Node with Two Children

Replace with **inorder successor** (smallest in right subtree) or **inorder predecessor** (largest in left subtree):

```
Delete 3 (has two children):
          [8]                    [8]
         /   \                  /   \
       [3]   [10]     -->    [4]   [10]
      /   \                 /   \
    [1]   [6]             [1]   [6]
         / \                   / \
       [4] [7]              [5] [7]
        \
        [5]

Inorder successor of 3 is 4 (smallest in right subtree).
Replace 3 with 4, then delete 4 from its original position.
```

```python
def delete(self, val):
    """Delete a value -- O(h)."""
    self.root = self._delete(self.root, val)
    self._size -= 1

def _delete(self, node, val):
    if node is None:
        raise ValueError(f"{val} not found")
    
    if val < node.val:
        node.left = self._delete(node.left, val)
    elif val > node.val:
        node.right = self._delete(node.right, val)
    else:
        # Found the node to delete
        if node.left is None:       # Case 1 & 2: no left child
            return node.right
        elif node.right is None:    # Case 2: no right child
            return node.left
        else:                       # Case 3: two children
            # Find inorder successor (min of right subtree)
            successor = self._find_min(node.right)
            node.val = successor.val
            node.right = self._delete(node.right, successor.val)
    
    return node
```

## Min, Max, Successor, Predecessor

```python
def find_min(self):
    """Find the minimum value -- O(h)."""
    if self.root is None:
        raise ValueError("Tree is empty")
    return self._find_min(self.root).val

def _find_min(self, node):
    """Go left until no more left children."""
    while node.left:
        node = node.left
    return node

def find_max(self):
    """Find the maximum value -- O(h)."""
    if self.root is None:
        raise ValueError("Tree is empty")
    node = self.root
    while node.right:
        node = node.right
    return node.val
```

```
Minimum: follow left pointers
          [8]
         /
       [3]
      /
    [1]  <-- minimum

Maximum: follow right pointers
          [8]
             \
             [10]
                \
                [14]  <-- maximum
```

## Inorder Successor and Predecessor

```python
def inorder_successor(self, val):
    """Find the next value in sorted order."""
    successor = None
    node = self.root
    while node:
        if val < node.val:
            successor = node
            node = node.left
        elif val > node.val:
            node = node.right
        else:
            # Found the node
            if node.right:
                return self._find_min(node.right).val
            return successor.val if successor else None
    return None
```

## BST Validation

```python
def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    """Check if a binary tree satisfies the BST property."""
    if root is None:
        return True
    if root.val <= min_val or root.val >= max_val:
        return False
    return (is_valid_bst(root.left, min_val, root.val) and
            is_valid_bst(root.right, root.val, max_val))
```

## Performance Analysis

| Operation | Average (balanced) | Worst (degenerate) |
|-----------|-------------------|-------------------|
| Search | O(log n) | O(n) |
| Insert | O(log n) | O(n) |
| Delete | O(log n) | O(n) |
| Min/Max | O(log n) | O(n) |
| Inorder traversal | O(n) | O(n) |

### The Degenerate Case

If you insert sorted data, the BST becomes a linked list:

```
Insert: 1, 2, 3, 4, 5
[1]
  \
  [2]
    \
    [3]
      \
      [4]
        \
        [5]    Height = 4 = n-1 (worst case!)
```

This is why **self-balancing BSTs** (AVL, Red-Black) exist -- they maintain O(log n) height through rotations.

## Self-Balancing Trees: A Preview

| Tree | Balance Guarantee | Rotation Cost | Best For |
|------|------------------|---------------|----------|
| AVL | Height difference <= 1 | O(log n) per op | Read-heavy workloads |
| Red-Black | Height <= 2 * log(n+1) | O(1) amortized | General-purpose (C++ STL, Java TreeMap) |
| B-tree | All leaves at same depth | O(log n) | Disk-based storage (databases) |

## Python's `bisect` Module

For sorted lists, Python's `bisect` provides BST-like operations:

```python
import bisect

sorted_list = [1, 3, 4, 6, 7, 8, 10, 13, 14]

# Binary search for insertion point
bisect.bisect_left(sorted_list, 6)   # 3 (index where 6 is)
bisect.bisect_right(sorted_list, 6)  # 4 (index after 6)

# Insert while maintaining sorted order
bisect.insort(sorted_list, 5)
# [1, 3, 4, 5, 6, 7, 8, 10, 13, 14]
```

## Building a BST from Sorted Array

To get a balanced BST, pick the middle element as root:

```python
def sorted_array_to_bst(nums):
    """Convert a sorted array to a balanced BST -- O(n)."""
    if not nums:
        return None
    mid = len(nums) // 2
    root = BSTNode(nums[mid])
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid + 1:])
    return root
```

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| BST property | Left < Node < Right |
| Inorder traversal | Yields sorted order |
| Search | O(h) -- binary search in a tree |
| Insert | Always creates a new leaf |
| Delete | Three cases: leaf, one child, two children |
| Balanced BST | O(log n) operations |
| Degenerate BST | O(n) operations (linked list) |
| Self-balancing | AVL, Red-Black maintain O(log n) height |

---

**Next**: [Heaps](./08_Heaps.md) -- Learn about partially ordered trees for priority queue operations.
