# Trees and Binary Search Trees (Tree and BST)

## Learning Objectives

After completing this lesson, you will be able to:

1. Define core tree terminology (root, leaf, depth, height, complete vs. full binary tree) and represent a binary tree using linked nodes
2. Implement in-order, pre-order, and post-order tree traversals both recursively and iteratively, and predict the output for a given tree
3. Implement Binary Search Tree (BST) insert, search, and delete operations, explaining the BST invariant maintained at each step
4. Analyze BST operation time complexity as O(h) and contrast performance between balanced and degenerate (skewed) trees
5. Explain how AVL trees and Red-Black trees maintain O(log n) height through rotations and rebalancing
6. Apply tree traversal techniques to solve problems such as level-order traversal, lowest common ancestor, and path sum

---

## Overview

A tree is a non-linear data structure that represents hierarchical structures. Binary Search Trees (BST) support efficient search, insertion, and deletion operations.

---

## Table of Contents

1. [Tree Basic Concepts](#1-tree-basic-concepts)
2. [Tree Traversal](#2-tree-traversal)
3. [Binary Search Tree](#3-binary-search-tree)
4. [BST Operations](#4-bst-operations)
5. [Balanced Tree Concepts](#5-balanced-tree-concepts)
6. [AVL and Red-Black Tree Visualization](#6-avl-and-red-black-tree-visualization)
7. [Practice Problems](#7-practice-problems)

---

## 1. Tree Basic Concepts

### Terminology

```
        (A) ← Root
       / | \
     (B)(C)(D) ← Internal Nodes
     / \     \
   (E)(F)    (G) ← Leaf Nodes

- Root: Topmost node (A)
- Leaf: Nodes with no children (E, F, C, G)
- Edge: Connection between nodes
- Parent/Child: A is B's parent, B is A's child
- Sibling: Nodes with the same parent (B, C, D)
- Depth: Distance from root (A=0, B=1, E=2)
- Height: Distance to the deepest leaf
```

### Binary Tree

```
Each node has at most 2 children

      (1)
     /   \
   (2)   (3)
   / \   /
 (4)(5)(6)

Special Binary Trees:
- Complete Binary Tree: All levels filled except last
- Full Binary Tree: All levels completely filled
- Skewed Tree: Children only on one side
```

### Node Structure

```c
// C
typedef struct Node {
    int data;
    struct Node* left;
    struct Node* right;
} Node;

Node* createNode(int data) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->data = data;
    node->left = node->right = NULL;
    return node;
}
```

```cpp
// C++
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;

    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};
```

```python
# Python
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None
```

---

## 2. Tree Traversal

### Traversal Methods

```
      (1)
     /   \
   (2)   (3)
   / \
 (4)(5)

Preorder: Root → Left → Right
  1 → 2 → 4 → 5 → 3

Inorder: Left → Root → Right
  4 → 2 → 5 → 1 → 3

Postorder: Left → Right → Root
  4 → 5 → 2 → 3 → 1

Level-order: Level by level, left to right
  1 → 2 → 3 → 4 → 5
```

### Recursive Implementation

```cpp
// C++
void preorder(TreeNode* root) {
    if (!root) return;
    cout << root->val << " ";  // Visit
    preorder(root->left);
    preorder(root->right);
}

void inorder(TreeNode* root) {
    if (!root) return;
    inorder(root->left);
    cout << root->val << " ";  // Visit
    inorder(root->right);
}

void postorder(TreeNode* root) {
    if (!root) return;
    postorder(root->left);
    postorder(root->right);
    cout << root->val << " ";  // Visit
}
```

```python
def preorder(root):
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def inorder(root):
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def postorder(root):
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```

### Iterative Implementation (Stack)

```cpp
// C++ - Inorder Traversal
vector<int> inorderIterative(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> st;
    TreeNode* curr = root;

    while (curr || !st.empty()) {
        while (curr) {
            st.push(curr);
            curr = curr->left;
        }

        curr = st.top();
        st.pop();
        result.push_back(curr->val);
        curr = curr->right;
    }

    return result;
}
```

```python
def inorder_iterative(root):
    result = []
    stack = []
    curr = root

    # Loop continues as long as there is an unvisited node (curr) or
    # nodes waiting to be processed (stack) — both conditions are needed
    # because the stack empties between subtrees
    while curr or stack:
        # Descend to the leftmost node, pushing ancestors onto the stack
        # so we can return to them after their left subtree is finished
        while curr:
            stack.append(curr)
            curr = curr.left

        # The top of the stack is the next node in sorted order —
        # its left subtree is fully processed
        curr = stack.pop()
        result.append(curr.val)
        # Move to the right subtree; if it is None the outer while-loop
        # will pop the next ancestor instead
        curr = curr.right

    return result
```

### Level-order Traversal (BFS)

```cpp
// C++
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> result;
    if (!root) return result;

    queue<TreeNode*> q;
    q.push(root);

    while (!q.empty()) {
        int size = q.size();
        vector<int> level;

        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front();
            q.pop();
            level.push_back(node->val);

            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }

        result.push_back(level);
    }

    return result;
}
```

```python
from collections import deque

def level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        # Snapshot len(queue) at the start of each iteration so we only
        # process nodes from the current level, not the children we enqueue
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)

            # Enqueue children for the next level; None children are skipped
            # to avoid storing sentinel values in the queue
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        # Each inner loop produces exactly one level's worth of values
        result.append(level)

    return result
```

---

## 3. Binary Search Tree (BST)

### BST Property

```
For every node:
- All values in left subtree < node value
- All values in right subtree > node value

       (8)
      /   \
    (3)   (10)
    / \      \
  (1)(6)    (14)
     / \    /
   (4)(7)(13)

Inorder traversal: 1, 3, 4, 6, 7, 8, 10, 13, 14 (sorted!)
```

### BST Operation Complexity

```
┌──────────┬─────────────┬─────────────┐
│ Operation│ Average     │ Worst       │
├──────────┼─────────────┼─────────────┤
│ Search   │ O(log n)    │ O(n)        │
│ Insert   │ O(log n)    │ O(n)        │
│ Delete   │ O(log n)    │ O(n)        │
└──────────┴─────────────┴─────────────┘

Worst case: Skewed tree (becomes like a linked list)
```

---

## 4. BST Operations

### 4.1 Search

```cpp
// C++
TreeNode* search(TreeNode* root, int key) {
    if (!root || root->val == key)
        return root;

    if (key < root->val)
        return search(root->left, key);

    return search(root->right, key);
}

// Iterative
TreeNode* searchIterative(TreeNode* root, int key) {
    while (root && root->val != key) {
        if (key < root->val)
            root = root->left;
        else
            root = root->right;
    }
    return root;
}
```

```python
def search(root, key):
    if not root or root.val == key:
        return root

    if key < root.val:
        return search(root.left, key)

    return search(root.right, key)
```

### 4.2 Insertion

```
Insert 5:
       (8)                 (8)
      /   \               /   \
    (3)   (10)    →     (3)   (10)
    / \                 / \
  (1)(6)              (1)(6)
                         /
                       (5)
```

```cpp
// C++
TreeNode* insert(TreeNode* root, int key) {
    if (!root) return new TreeNode(key);

    if (key < root->val)
        root->left = insert(root->left, key);
    else if (key > root->val)
        root->right = insert(root->right, key);

    return root;
}
```

```python
def insert(root, key):
    if not root:
        return TreeNode(key)

    if key < root.val:
        root.left = insert(root.left, key)
    elif key > root.val:
        root.right = insert(root.right, key)

    return root
```

### 4.3 Deletion

```
Three cases:

1. Leaf node: Just delete

2. One child: Replace with child

3. Two children: Replace with inorder successor
   - Minimum value in right subtree
   - Or maximum value in left subtree

       (8)                 (8)
      /   \               /   \
    (3)   (10)    →     (4)   (10)
    / \                 / \
  (1)(6)              (1)(6)
     /
   (4)

Delete 3: Replace with successor 4
```

```cpp
// C++
TreeNode* findMin(TreeNode* root) {
    while (root->left)
        root = root->left;
    return root;
}

TreeNode* deleteNode(TreeNode* root, int key) {
    if (!root) return nullptr;

    if (key < root->val) {
        root->left = deleteNode(root->left, key);
    } else if (key > root->val) {
        root->right = deleteNode(root->right, key);
    } else {
        // Found!
        if (!root->left) {
            TreeNode* temp = root->right;
            delete root;
            return temp;
        }
        if (!root->right) {
            TreeNode* temp = root->left;
            delete root;
            return temp;
        }

        // Two children: Replace with successor
        TreeNode* successor = findMin(root->right);
        root->val = successor->val;
        root->right = deleteNode(root->right, successor->val);
    }

    return root;
}
```

```python
def delete_node(root, key):
    if not root:
        return None

    # BST property guides us toward the target node in O(h) time
    if key < root.val:
        root.left = delete_node(root.left, key)
    elif key > root.val:
        root.right = delete_node(root.right, key)
    else:
        # Node found — handle the three deletion cases:
        # Case 1 & 2: zero or one child — simply splice the node out
        if not root.left:
            return root.right
        if not root.right:
            return root.left

        # Case 3: two children — replace value with in-order successor
        # (minimum of right subtree) so the BST property is preserved;
        # then delete the successor from the right subtree
        successor = root.right
        while successor.left:
            successor = successor.left  # Leftmost node = smallest in right subtree

        root.val = successor.val
        root.right = delete_node(root.right, successor.val)

    return root
```

---

## 5. Balanced Tree Concepts

### Skewed Tree Problem

```
1 → 2 → 3 → 4 → 5

All operations become O(n)!
```

### Types of Balanced Trees

```
1. AVL Tree
   - Height difference between left/right <= 1 for all nodes
   - Maintains balance through rotations on insert/delete

2. Red-Black Tree
   - Each node is red or black
   - Maintains balance through specific rules
   - Foundation for C++ map, set

3. B-Tree
   - Multi-way search tree
   - Used in databases
```

### Tree Height Calculation

```python
def height(root):
    if not root:
        return -1  # or 0
    return 1 + max(height(root.left), height(root.right))
```

### BST Validation

```python
def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    if not root:
        return True

    if root.val <= min_val or root.val >= max_val:
        return False

    return (is_valid_bst(root.left, min_val, root.val) and
            is_valid_bst(root.right, root.val, max_val))
```

---

## 6. AVL and Red-Black Tree Visualization

Understanding balanced tree rotations is much easier with visual diagrams. This section walks through AVL rotations, Red-Black tree insertion cases, and a step-by-step insertion trace.

### 6.1 AVL Tree Rotations

An AVL tree maintains the invariant that for every node, the height difference between its left and right subtrees (the **balance factor**) is at most 1. When an insertion or deletion violates this, we restore balance through **rotations**.

#### Single Right Rotation (LL Case)

When a node becomes left-heavy because of an insertion into the left child's left subtree:

```
Before (insert 5):           After right rotation at 30:

        30  (bf=+2)                  20
       /                            /  \
      20  (bf=+1)                 10    30
     /
    10

The left child (20) becomes the new root of this subtree.
30 adopts 20's right child (if any) as its left child.
```

#### Single Left Rotation (RR Case)

Mirror of the LL case — a node becomes right-heavy due to an insertion into the right child's right subtree:

```
Before (insert 30):          After left rotation at 10:

    10  (bf=-2)                      20
      \                             /  \
      20  (bf=-1)                 10    30
        \
        30

The right child (20) becomes the new root.
10 adopts 20's left child (if any) as its right child.
```

#### Left-Right Rotation (LR Case)

The node is left-heavy, but the imbalance is in the left child's **right** subtree. We need two rotations: first left-rotate the left child, then right-rotate the node.

```
Before (insert 25):

        30  (bf=+2)
       /
      20  (bf=-1)
        \
        25

Step 1 — Left rotate at 20:     Step 2 — Right rotate at 30:

        30                               25
       /                                /  \
      25                              20    30
     /
    20
```

#### Right-Left Rotation (RL Case)

Mirror of the LR case — right-heavy, with imbalance in the right child's left subtree:

```
Before (insert 25):

    20  (bf=-2)
      \
      30  (bf=+1)
      /
    25

Step 1 — Right rotate at 30:    Step 2 — Left rotate at 20:

    20                                   25
      \                                 /  \
      25                              20    30
        \
        30
```

### 6.2 Red-Black Tree Insert Cases

A Red-Black tree enforces these rules:
1. Every node is red or black
2. The root is black
3. No two consecutive red nodes (red node's children must be black)
4. Every path from root to a NULL leaf has the same number of black nodes

After inserting a new node (always colored **red**), violations are fixed through recoloring and rotations.

#### Case 1: Uncle is Red (Recolor)

When both the parent and uncle are red, we simply recolor:

```
Before:                       After recolor:

      G(B)                         G(R) ← may violate at grandparent
     /   \                        /   \
   P(R)  U(R)                  P(B)  U(B)
   /                           /
  N(R) ← new                 N(R)

Recolor parent and uncle to black, grandparent to red.
Then recurse upward from grandparent.
```

#### Case 2: Uncle is Black, Triangle (Rotate to Line)

The new node, its parent, and grandparent form a "triangle" (e.g., left child of right child):

```
Before (triangle):            After rotate at P:

      G(B)                         G(B)
     /   \                        /   \
   P(R)  U(B)                  N(R)  U(B)
     \                         /
     N(R)                    P(R)

Left-rotate at P to convert triangle into a line.
Then apply Case 3.
```

#### Case 3: Uncle is Black, Line (Rotate + Recolor)

The new node, parent, and grandparent form a "line" (e.g., left-left):

```
Before (line):                After rotate at G + recolor:

      G(B)                         P(B)
     /   \                        /   \
   P(R)  U(B)                  N(R)  G(R)
   /                                   \
  N(R)                                 U(B)

Right-rotate at G, swap colors of P and G.
```

#### All Four Rotation Patterns Summary

```
┌────────────────────┬──────────────────────────────────────────────┐
│ Configuration      │ Action                                       │
├────────────────────┼──────────────────────────────────────────────┤
│ Uncle Red          │ Recolor P, U → black, G → red, recurse      │
│ Uncle Black, LL    │ Right-rotate G, swap P/G colors              │
│ Uncle Black, RR    │ Left-rotate G, swap P/G colors               │
│ Uncle Black, LR    │ Left-rotate P → becomes LL, then fix LL     │
│ Uncle Black, RL    │ Right-rotate P → becomes RR, then fix RR    │
└────────────────────┴──────────────────────────────────────────────┘
```

### 6.3 Step-by-Step AVL Insertion Trace

Insert the sequence **[10, 20, 30, 25, 28]** into an empty AVL tree:

```
Step 1: Insert 10              Step 2: Insert 20

    10                             10
                                     \
                                     20

Step 3: Insert 30 → RR violation at 10 (bf=-2), left rotate:

    10  (bf=-2)                    20
      \                           /  \
      20           →            10    30
        \
        30

Step 4: Insert 25 → RL violation at 30 (bf=+1→ok), tree is balanced:

        20
       /  \
     10    30
           /
         25

Step 5: Insert 28 → RL at 30: right-rotate 25/28, then left-rotate 30/28:

        20                         20                         20
       /  \                       /  \                       /  \
     10    30  (bf=+2)  →      10    30         →         10    28
           /                         /                         /  \
         25 (bf=-1)               28                         25    30
           \                     /
           28                  25
```

### 6.4 Python Tree Visualization

```python
class AVLNode:
    """AVL tree node with height tracking."""
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1  # New nodes start at height 1


def print_tree(node, prefix="", is_left=True):
    """
    Print a binary tree in a readable ASCII format.

    Why this approach: Recursive prefix-building produces clean,
    indented output that clearly shows parent-child relationships.
    Each node displays its key and balance factor (for AVL trees).
    """
    if node is None:
        return

    # Right subtree first (appears on top in the output)
    print_tree(node.right, prefix + ("│   " if is_left else "    "), False)

    # Current node
    connector = "└── " if is_left else "┌── "
    bf = get_balance(node)
    print(f"{prefix}{connector}{node.key} (bf={bf:+d})")

    # Left subtree
    print_tree(node.left, prefix + ("    " if is_left else "│   "), True)


def height(node):
    return node.height if node else 0


def get_balance(node):
    """Balance factor = left height - right height."""
    if not node:
        return 0
    return height(node.left) - height(node.right)


def right_rotate(y):
    """
    Right rotation (LL case fix).

         y              x
        / \            / \
       x   C   →     A   y
      / \                / \
     A   B              B   C
    """
    x = y.left
    B = x.right

    x.right = y
    y.left = B

    y.height = 1 + max(height(y.left), height(y.right))
    x.height = 1 + max(height(x.left), height(x.right))

    return x


def left_rotate(x):
    """
    Left rotation (RR case fix).

       x                y
      / \              / \
     A   y     →      x   C
        / \          / \
       B   C        A   B
    """
    y = x.right
    B = y.left

    y.left = x
    x.right = B

    x.height = 1 + max(height(x.left), height(x.right))
    y.height = 1 + max(height(y.left), height(y.right))

    return y


def avl_insert(node, key):
    """
    Insert a key into an AVL tree and rebalance.

    The function returns the new root of the subtree after
    insertion and any necessary rotations.
    """
    # Standard BST insert
    if not node:
        return AVLNode(key)

    if key < node.key:
        node.left = avl_insert(node.left, key)
    elif key > node.key:
        node.right = avl_insert(node.right, key)
    else:
        return node  # Duplicate keys not allowed

    # Update height
    node.height = 1 + max(height(node.left), height(node.right))

    # Check balance
    bf = get_balance(node)

    # LL Case: left-heavy, inserted into left-left
    if bf > 1 and key < node.left.key:
        return right_rotate(node)

    # RR Case: right-heavy, inserted into right-right
    if bf < -1 and key > node.right.key:
        return left_rotate(node)

    # LR Case: left-heavy, inserted into left-right
    if bf > 1 and key > node.left.key:
        node.left = left_rotate(node.left)
        return right_rotate(node)

    # RL Case: right-heavy, inserted into right-left
    if bf < -1 and key < node.right.key:
        node.right = right_rotate(node.right)
        return left_rotate(node)

    return node


# Demo: build AVL tree and visualize
if __name__ == "__main__":
    root = None
    for key in [10, 20, 30, 25, 28, 5, 3]:
        root = avl_insert(root, key)
        print(f"\n--- After inserting {key} ---")
        print_tree(root)
```

### 6.5 AVL vs Red-Black Tree — When to Use Which

```
┌────────────────────┬──────────────────────┬──────────────────────┐
│ Criterion          │ AVL Tree             │ Red-Black Tree       │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Balance strictness │ Strict (bf ≤ 1)      │ Relaxed (≤ 2× depth) │
│ Search speed       │ Faster (shorter)     │ Slightly slower      │
│ Insert/Delete      │ Slower (more rotate) │ Faster (fewer rotate)│
│ Rotations/insert   │ Up to O(log n)       │ At most 2            │
│ Rotations/delete   │ Up to O(log n)       │ At most 3            │
│ Memory per node    │ Height (int)         │ Color (1 bit)        │
│ Best for           │ Read-heavy workloads │ Write-heavy workloads│
│ Real-world usage   │ Databases, lookups   │ std::map, TreeMap    │
└────────────────────┴──────────────────────┴──────────────────────┘

Rule of thumb:
- If your workload is mostly searches → AVL tree
  (stricter balance = shorter tree = faster lookups)
- If your workload has frequent inserts/deletes → Red-Black tree
  (fewer rotations per modification = faster writes)
- Most standard library implementations (C++ std::map, Java TreeMap)
  use Red-Black trees because general-purpose usage involves
  a mix of reads and writes, and fewer rotations simplify the code.
```

---

## 7. Practice Problems

### Problem 1: Lowest Common Ancestor (LCA)

Find the lowest common ancestor of two nodes in a BST

<details>
<summary>Solution Code</summary>

```python
def lowest_common_ancestor(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root

    return None
```

</details>

### Recommended Problems

| Difficulty | Problem | Platform | Type |
|--------|------|--------|------|
| ⭐ | [Tree Traversal](https://www.acmicpc.net/problem/1991) | BOJ | Traversal |
| ⭐⭐ | [Validate BST](https://leetcode.com/problems/validate-binary-search-tree/) | LeetCode | BST |
| ⭐⭐ | [Binary Tree Inorder](https://leetcode.com/problems/binary-tree-inorder-traversal/) | LeetCode | Traversal |
| ⭐⭐ | [Binary Search Tree](https://www.acmicpc.net/problem/5639) | BOJ | BST |
| ⭐⭐⭐ | [LCA](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) | LeetCode | LCA |

---

## Next Steps

- [10_Heaps_and_Priority_Queues.md](./10_Heaps_and_Priority_Queues.md) - Heaps, Priority Queues

---

## References

- [Tree Visualization](https://visualgo.net/en/bst)
- Introduction to Algorithms (CLRS) - Chapter 12, 13
