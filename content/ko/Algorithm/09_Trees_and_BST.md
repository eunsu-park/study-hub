# 트리와 이진 탐색 트리 (Tree and BST)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 트리의 핵심 용어(루트, 리프, 깊이, 높이, 완전 이진 트리 vs 포화 이진 트리)를 정의하고 연결 노드로 이진 트리(binary tree)를 표현할 수 있다
2. 중위(in-order), 전위(pre-order), 후위(post-order) 트리 순회를 재귀 및 반복 방식으로 구현하고, 주어진 트리에 대한 출력을 예측할 수 있다
3. 이진 탐색 트리(BST, Binary Search Tree)의 삽입, 탐색, 삭제 연산을 구현하고 각 단계에서 유지되는 BST 불변 조건을 설명할 수 있다
4. BST 연산 시간 복잡도를 O(h)로 분석하고, 균형 트리(balanced tree)와 편향 트리(degenerate/skewed tree) 간의 성능 차이를 비교할 수 있다
5. AVL 트리와 레드-블랙 트리(Red-Black tree)가 회전(rotation)과 재균형(rebalancing)을 통해 O(log n) 높이를 유지하는 방법을 설명할 수 있다
6. 레벨 순서 순회(level-order traversal), 최소 공통 조상(LCA, Lowest Common Ancestor), 경로 합(path sum) 등의 문제를 해결하기 위해 트리 순회 기법을 적용할 수 있다

---

## 개요

트리는 계층적 구조를 나타내는 비선형 자료구조입니다. 이진 탐색 트리(BST)는 효율적인 탐색, 삽입, 삭제를 지원합니다.

---

## 목차

1. [트리 기본 개념](#1-트리-기본-개념)
2. [트리 순회](#2-트리-순회)
3. [이진 탐색 트리](#3-이진-탐색-트리)
4. [BST 연산](#4-bst-연산)
5. [균형 트리 개념](#5-균형-트리-개념)
6. [AVL 트리와 Red-Black 트리 시각화](#6-avl-트리와-red-black-트리-시각화)
7. [연습 문제](#7-연습-문제)

---

## 1. 트리 기본 개념

### 용어 정리

```
        (A) ← 루트 (Root)
       / | \
     (B)(C)(D) ← 내부 노드 (Internal)
     / \     \
   (E)(F)    (G) ← 리프 (Leaf)

- 루트: 최상위 노드 (A)
- 리프: 자식이 없는 노드 (E, F, C, G)
- 간선: 노드 연결선
- 부모/자식: A는 B의 부모, B는 A의 자식
- 형제: 같은 부모를 가진 노드 (B, C, D)
- 깊이: 루트에서의 거리 (A=0, B=1, E=2)
- 높이: 가장 깊은 리프까지 거리
```

### 이진 트리 (Binary Tree)

```
각 노드가 최대 2개의 자식을 가짐

      (1)
     /   \
   (2)   (3)
   / \   /
 (4)(5)(6)

특별한 이진 트리:
- 완전 이진 트리: 마지막 레벨 제외 모두 채워짐
- 포화 이진 트리: 모든 레벨이 완전히 채워짐
- 편향 트리: 한쪽으로만 자식이 있음
```

### 노드 구조

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

## 2. 트리 순회

### 순회 방법

```
      (1)
     /   \
   (2)   (3)
   / \
 (4)(5)

전위 (Preorder): 루트 → 왼쪽 → 오른쪽
  1 → 2 → 4 → 5 → 3

중위 (Inorder): 왼쪽 → 루트 → 오른쪽
  4 → 2 → 5 → 1 → 3

후위 (Postorder): 왼쪽 → 오른쪽 → 루트
  4 → 5 → 2 → 3 → 1

레벨 (Level-order): 레벨별로 왼쪽에서 오른쪽
  1 → 2 → 3 → 4 → 5
```

### 재귀 구현

```cpp
// C++
void preorder(TreeNode* root) {
    if (!root) return;
    cout << root->val << " ";  // 방문
    preorder(root->left);
    preorder(root->right);
}

void inorder(TreeNode* root) {
    if (!root) return;
    inorder(root->left);
    cout << root->val << " ";  // 방문
    inorder(root->right);
}

void postorder(TreeNode* root) {
    if (!root) return;
    postorder(root->left);
    postorder(root->right);
    cout << root->val << " ";  // 방문
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

### 반복 구현 (스택)

```cpp
// C++ - 중위 순회
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

    # 미방문 노드(curr)가 있거나 처리 대기 중인 노드(stack)가 있는 한 계속;
    # 서브트리 사이에서 스택이 비워지므로 두 조건 모두 필요
    while curr or stack:
        # 가장 왼쪽 노드까지 내려가며 조상을 스택에 push하여
        # 왼쪽 서브트리 처리 후 돌아올 수 있게 함
        while curr:
            stack.append(curr)
            curr = curr.left

        # 스택의 최상단이 정렬 순서상 다음 노드 —
        # 왼쪽 서브트리는 완전히 처리됨
        curr = stack.pop()
        result.append(curr.val)
        # 오른쪽 서브트리로 이동; None이면 외부 while 루프가
        # 대신 다음 조상을 pop함
        curr = curr.right

    return result
```

### 레벨 순회 (BFS)

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
        # 각 반복 시작 시 len(queue)를 스냅샷하여 현재 레벨의
        # 노드만 처리하고, 큐에 추가되는 자식은 처리하지 않음
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)

            # 다음 레벨을 위해 자식을 큐에 추가; None 자식은 건너뛰어
            # 큐에 센티널 값을 저장하지 않음
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        # 각 내부 루프는 정확히 한 레벨 분량의 값을 생성
        result.append(level)

    return result
```

---

## 3. 이진 탐색 트리 (BST)

### BST 속성

```
모든 노드에 대해:
- 왼쪽 서브트리의 모든 값 < 노드 값
- 오른쪽 서브트리의 모든 값 > 노드 값

       (8)
      /   \
    (3)   (10)
    / \      \
  (1)(6)    (14)
     / \    /
   (4)(7)(13)

중위 순회: 1, 3, 4, 6, 7, 8, 10, 13, 14 (정렬됨!)
```

### BST 연산 복잡도

```
┌──────────┬─────────────┬─────────────┐
│ 연산     │ 평균        │ 최악        │
├──────────┼─────────────┼─────────────┤
│ 탐색     │ O(log n)    │ O(n)        │
│ 삽입     │ O(log n)    │ O(n)        │
│ 삭제     │ O(log n)    │ O(n)        │
└──────────┴─────────────┴─────────────┘

최악: 편향 트리 (연결 리스트처럼 됨)
```

---

## 4. BST 연산

### 4.1 탐색

```cpp
// C++
TreeNode* search(TreeNode* root, int key) {
    if (!root || root->val == key)
        return root;

    if (key < root->val)
        return search(root->left, key);

    return search(root->right, key);
}

// 반복
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

### 4.2 삽입

```
5 삽입:
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

### 4.3 삭제

```
3가지 경우:

1. 리프 노드: 그냥 삭제

2. 자식 1개: 자식으로 대체

3. 자식 2개: 후속자(inorder successor)로 대체
   - 오른쪽 서브트리의 최솟값
   - 또는 왼쪽 서브트리의 최댓값

       (8)                 (8)
      /   \               /   \
    (3)   (10)    →     (4)   (10)
    / \                 / \
  (1)(6)              (1)(6)
     /
   (4)

3 삭제: 후속자 4로 대체
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
        // 찾음!
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

        // 자식 2개: 후속자로 대체
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

    # BST 속성이 O(h) 시간에 대상 노드로 안내
    if key < root.val:
        root.left = delete_node(root.left, key)
    elif key > root.val:
        root.right = delete_node(root.right, key)
    else:
        # 노드 발견 — 세 가지 삭제 케이스를 처리:
        # 케이스 1 & 2: 자식이 0개 또는 1개 — 단순히 노드를 잘라냄
        if not root.left:
            return root.right
        if not root.right:
            return root.left

        # 케이스 3: 자식 2개 — 중위 후속자(오른쪽 서브트리의 최솟값)로
        # 값을 대체하여 BST 속성을 유지;
        # 그 다음 오른쪽 서브트리에서 후속자를 삭제
        successor = root.right
        while successor.left:
            successor = successor.left  # 가장 왼쪽 노드 = 오른쪽 서브트리에서 가장 작은 값

        root.val = successor.val
        root.right = delete_node(root.right, successor.val)

    return root
```

---

## 5. 균형 트리 개념

### 편향 트리 문제

```
1 → 2 → 3 → 4 → 5

모든 연산이 O(n)!
```

### 균형 트리 종류

```
1. AVL 트리
   - 모든 노드에서 왼쪽/오른쪽 높이 차 ≤ 1
   - 삽입/삭제 시 회전으로 균형 유지

2. Red-Black 트리
   - 각 노드가 빨강/검정
   - 특정 규칙으로 균형 유지
   - C++ map, set의 기반

3. B-트리
   - 다진 탐색 트리
   - 데이터베이스에서 사용
```

### 트리 높이 계산

```python
def height(root):
    if not root:
        return -1  # 또는 0
    return 1 + max(height(root.left), height(root.right))
```

### BST 검증

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

## 6. AVL 트리와 Red-Black 트리 시각화

균형 트리의 회전(Rotation)을 시각적 다이어그램과 함께 이해하면 훨씬 쉽습니다. 이 섹션에서는 AVL 회전, Red-Black 트리 삽입 케이스, 단계별 삽입 추적을 다룹니다.

### 6.1 AVL 트리 회전(Rotation)

AVL 트리는 모든 노드에서 왼쪽과 오른쪽 서브트리의 높이 차이(**균형 인수**, Balance Factor)가 최대 1인 불변식을 유지합니다. 삽입이나 삭제로 이 조건이 깨지면 **회전**을 통해 균형을 복원합니다.

#### 단일 우회전 (LL Case)

왼쪽 자식의 왼쪽 서브트리에 삽입하여 노드가 왼쪽으로 무거워진 경우:

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

#### 단일 좌회전 (RR Case)

LL 케이스의 거울상 — 오른쪽 자식의 오른쪽 서브트리에 삽입하여 노드가 오른쪽으로 무거워진 경우:

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

#### 좌-우 이중 회전 (LR Case)

노드가 왼쪽으로 무거운데, 불균형이 왼쪽 자식의 **오른쪽** 서브트리에 있는 경우입니다. 두 번의 회전이 필요합니다: 먼저 왼쪽 자식을 좌회전, 그 다음 노드를 우회전합니다.

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

#### 우-좌 이중 회전 (RL Case)

LR 케이스의 거울상 — 오른쪽으로 무거운데, 불균형이 오른쪽 자식의 왼쪽 서브트리에 있는 경우:

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

### 6.2 Red-Black 트리 삽입 케이스

Red-Black 트리는 다음 규칙을 강제합니다:
1. 모든 노드는 빨강(Red) 또는 검정(Black)
2. 루트(Root)는 검정
3. 연속된 두 빨강 노드 불가 (빨강 노드의 자식은 반드시 검정)
4. 루트에서 모든 NULL 리프까지의 경로에 있는 검정 노드 수가 동일

새 노드를 삽입하면 (항상 **빨강**으로 색칠) 위반이 발생할 수 있으며, 재색칠(Recoloring)과 회전(Rotation)으로 수정합니다.

#### Case 1: 삼촌(Uncle)이 빨강 (재색칠)

부모와 삼촌이 모두 빨강이면 단순히 재색칠합니다:

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

#### Case 2: 삼촌이 검정, 삼각형(Triangle) 구조 (회전하여 직선으로 변환)

새 노드, 부모, 조부모가 "삼각형"을 이루는 경우 (예: 오른쪽 자식의 왼쪽 자식):

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

#### Case 3: 삼촌이 검정, 직선(Line) 구조 (회전 + 재색칠)

새 노드, 부모, 조부모가 "직선"을 이루는 경우 (예: 왼쪽-왼쪽):

```
Before (line):                After rotate at G + recolor:

      G(B)                         P(B)
     /   \                        /   \
   P(R)  U(B)                  N(R)  G(R)
   /                                   \
  N(R)                                 U(B)

Right-rotate at G, swap colors of P and G.
```

#### 4가지 회전 패턴 요약

```
┌────────────────────┬──────────────────────────────────────────────┐
│ 구성               │ 동작                                         │
├────────────────────┼──────────────────────────────────────────────┤
│ Uncle Red          │ P, U → 검정, G → 빨강으로 재색칠, 위로 재귀  │
│ Uncle Black, LL    │ G 우회전, P/G 색상 교환                      │
│ Uncle Black, RR    │ G 좌회전, P/G 색상 교환                      │
│ Uncle Black, LR    │ P 좌회전 → LL 변환, LL 수정                  │
│ Uncle Black, RL    │ P 우회전 → RR 변환, RR 수정                  │
└────────────────────┴──────────────────────────────────────────────┘
```

### 6.3 단계별 AVL 삽입 추적

빈 AVL 트리에 **[10, 20, 30, 25, 28]** 순서로 삽입:

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

### 6.4 Python 트리 시각화 코드

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

### 6.5 AVL vs Red-Black 트리 — 언제 어떤 것을 사용할까

```
┌────────────────────┬──────────────────────┬──────────────────────┐
│ 기준               │ AVL 트리             │ Red-Black 트리       │
├────────────────────┼──────────────────────┼──────────────────────┤
│ 균형 엄격도        │ 엄격 (bf ≤ 1)        │ 느슨 (≤ 2× 깊이)    │
│ 탐색 속도          │ 더 빠름 (높이 낮음)   │ 약간 느림            │
│ 삽입/삭제          │ 느림 (회전 많음)      │ 빠름 (회전 적음)     │
│ 삽입당 회전 수     │ 최대 O(log n)        │ 최대 2               │
│ 삭제당 회전 수     │ 최대 O(log n)        │ 최대 3               │
│ 노드당 메모리      │ 높이(Height, int)    │ 색상(Color, 1 bit)   │
│ 적합한 용도        │ 읽기 위주 워크로드    │ 쓰기 위주 워크로드   │
│ 실제 사용          │ 데이터베이스, 검색    │ std::map, TreeMap    │
└────────────────────┴──────────────────────┴──────────────────────┘

경험 법칙:
- 검색이 주된 작업이라면 → AVL 트리
  (엄격한 균형 = 낮은 트리 = 빠른 검색)
- 삽입/삭제가 빈번하다면 → Red-Black 트리
  (수정당 회전이 적어 쓰기 성능이 우수)
- 대부분의 표준 라이브러리(C++ std::map, Java TreeMap)는
  Red-Black 트리를 사용합니다. 범용적으로 읽기와 쓰기가
  혼합되며, 회전이 적어 구현이 간단하기 때문입니다.
```

---

## 7. 연습 문제

### 문제 1: 최소 공통 조상 (LCA)

BST에서 두 노드의 최소 공통 조상 찾기

<details>
<summary>정답 코드</summary>

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

### 추천 문제

| 난이도 | 문제 | 플랫폼 | 유형 |
|--------|------|--------|------|
| ⭐ | [트리 순회](https://www.acmicpc.net/problem/1991) | 백준 | 순회 |
| ⭐⭐ | [Validate BST](https://leetcode.com/problems/validate-binary-search-tree/) | LeetCode | BST |
| ⭐⭐ | [Binary Tree Inorder](https://leetcode.com/problems/binary-tree-inorder-traversal/) | LeetCode | 순회 |
| ⭐⭐ | [이진 검색 트리](https://www.acmicpc.net/problem/5639) | 백준 | BST |
| ⭐⭐⭐ | [LCA](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) | LeetCode | LCA |

---

## 다음 단계

- [10_Heaps_and_Priority_Queues.md](./10_Heaps_and_Priority_Queues.md) - 힙, 우선순위 큐

---

## 참고 자료

- [Tree Visualization](https://visualgo.net/en/bst)
- Introduction to Algorithms (CLRS) - Chapter 12, 13
