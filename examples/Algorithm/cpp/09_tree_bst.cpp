/*
 * Tree and Binary Search Tree (BST)
 * Tree Traversal, BST Operations, LCA
 *
 * Implementation and applications of hierarchical data structures.
 */

#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <algorithm>
#include <climits>

using namespace std;

// =============================================================================
// 1. Binary Tree Node
// =============================================================================

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

// =============================================================================
// 2. Tree Traversal
// =============================================================================

// Preorder Traversal
void preorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    result.push_back(root->val);
    preorder(root->left, result);
    preorder(root->right, result);
}

// Inorder Traversal
void inorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    inorder(root->left, result);
    result.push_back(root->val);
    inorder(root->right, result);
}

// Postorder Traversal
void postorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    postorder(root->left, result);
    postorder(root->right, result);
    result.push_back(root->val);
}

// Level Order Traversal
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

// Iterative Inorder Traversal
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

// =============================================================================
// 3. BST Operations
// =============================================================================

class BST {
public:
    TreeNode* root;

    BST() : root(nullptr) {}

    // Insert
    TreeNode* insert(TreeNode* node, int val) {
        if (!node) return new TreeNode(val);

        if (val < node->val)
            node->left = insert(node->left, val);
        else if (val > node->val)
            node->right = insert(node->right, val);

        return node;
    }

    void insert(int val) {
        root = insert(root, val);
    }

    // Search
    TreeNode* search(TreeNode* node, int val) {
        if (!node || node->val == val) return node;

        if (val < node->val)
            return search(node->left, val);
        return search(node->right, val);
    }

    bool search(int val) {
        return search(root, val) != nullptr;
    }

    // Find minimum
    TreeNode* findMin(TreeNode* node) {
        while (node && node->left)
            node = node->left;
        return node;
    }

    // Delete
    TreeNode* remove(TreeNode* node, int val) {
        if (!node) return nullptr;

        if (val < node->val) {
            node->left = remove(node->left, val);
        } else if (val > node->val) {
            node->right = remove(node->right, val);
        } else {
            // No child or one child
            if (!node->left) {
                TreeNode* temp = node->right;
                delete node;
                return temp;
            }
            if (!node->right) {
                TreeNode* temp = node->left;
                delete node;
                return temp;
            }

            // Two children
            TreeNode* successor = findMin(node->right);
            node->val = successor->val;
            node->right = remove(node->right, successor->val);
        }
        return node;
    }

    void remove(int val) {
        root = remove(root, val);
    }
};

// =============================================================================
// 4. Tree Properties
// =============================================================================

// Height
int height(TreeNode* root) {
    if (!root) return 0;
    return 1 + max(height(root->left), height(root->right));
}

// Node count
int countNodes(TreeNode* root) {
    if (!root) return 0;
    return 1 + countNodes(root->left) + countNodes(root->right);
}

// Balance check
bool isBalanced(TreeNode* root) {
    if (!root) return true;

    int leftH = height(root->left);
    int rightH = height(root->right);

    return abs(leftH - rightH) <= 1 &&
           isBalanced(root->left) &&
           isBalanced(root->right);
}

// BST validity check
bool isValidBST(TreeNode* root, long long minVal = LLONG_MIN, long long maxVal = LLONG_MAX) {
    if (!root) return true;

    if (root->val <= minVal || root->val >= maxVal)
        return false;

    return isValidBST(root->left, minVal, root->val) &&
           isValidBST(root->right, root->val, maxVal);
}

// =============================================================================
// 5. LCA (Lowest Common Ancestor)
// =============================================================================

// LCA for general binary tree
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) return root;

    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);

    if (left && right) return root;
    return left ? left : right;
}

// LCA for BST
TreeNode* lcaBST(TreeNode* root, TreeNode* p, TreeNode* q) {
    while (root) {
        if (p->val < root->val && q->val < root->val)
            root = root->left;
        else if (p->val > root->val && q->val > root->val)
            root = root->right;
        else
            return root;
    }
    return nullptr;
}

// =============================================================================
// 6. Path Sum
// =============================================================================

// Check if a root-to-leaf path with given target sum exists
bool hasPathSum(TreeNode* root, int targetSum) {
    if (!root) return false;

    if (!root->left && !root->right)
        return targetSum == root->val;

    return hasPathSum(root->left, targetSum - root->val) ||
           hasPathSum(root->right, targetSum - root->val);
}

// Find all paths
void pathSumHelper(TreeNode* root, int sum, vector<int>& path, vector<vector<int>>& result) {
    if (!root) return;

    path.push_back(root->val);

    if (!root->left && !root->right && sum == root->val) {
        result.push_back(path);
    } else {
        pathSumHelper(root->left, sum - root->val, path, result);
        pathSumHelper(root->right, sum - root->val, path, result);
    }

    path.pop_back();
}

vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
    vector<vector<int>> result;
    vector<int> path;
    pathSumHelper(root, targetSum, path, result);
    return result;
}

// =============================================================================
// Test
// =============================================================================

void printVector(const vector<int>& v) {
    cout << "[";
    for (size_t i = 0; i < v.size(); i++) {
        cout << v[i];
        if (i < v.size() - 1) cout << ", ";
    }
    cout << "]";
}

int main() {
    cout << "============================================================" << endl;
    cout << "Tree and BST Examples" << endl;
    cout << "============================================================" << endl;

    // Create test tree
    //       4
    //      / \
    //     2   6
    //    / \ / \
    //   1  3 5  7
    TreeNode* root = new TreeNode(4);
    root->left = new TreeNode(2);
    root->right = new TreeNode(6);
    root->left->left = new TreeNode(1);
    root->left->right = new TreeNode(3);
    root->right->left = new TreeNode(5);
    root->right->right = new TreeNode(7);

    // 1. Tree Traversal
    cout << "\n[1] Tree Traversal" << endl;
    vector<int> pre, in, post;
    preorder(root, pre);
    inorder(root, in);
    postorder(root, post);

    cout << "    Preorder: ";
    printVector(pre);
    cout << endl;
    cout << "    Inorder: ";
    printVector(in);
    cout << endl;
    cout << "    Postorder: ";
    printVector(post);
    cout << endl;

    // 2. Level Order Traversal
    cout << "\n[2] Level Order Traversal" << endl;
    auto levels = levelOrder(root);
    for (size_t i = 0; i < levels.size(); i++) {
        cout << "    Level " << i << ": ";
        printVector(levels[i]);
        cout << endl;
    }

    // 3. BST Operations
    cout << "\n[3] BST Operations" << endl;
    BST bst;
    bst.insert(50);
    bst.insert(30);
    bst.insert(70);
    bst.insert(20);
    bst.insert(40);

    cout << "    Inserted: 50, 30, 70, 20, 40" << endl;
    cout << "    Search 30: " << (bst.search(30) ? "found" : "not found") << endl;
    cout << "    Search 60: " << (bst.search(60) ? "found" : "not found") << endl;

    // 4. Tree Properties
    cout << "\n[4] Tree Properties" << endl;
    cout << "    Height: " << height(root) << endl;
    cout << "    Node count: " << countNodes(root) << endl;
    cout << "    Balanced: " << (isBalanced(root) ? "yes" : "no") << endl;
    cout << "    Valid BST: " << (isValidBST(root) ? "yes" : "no") << endl;

    // 5. LCA
    cout << "\n[5] Lowest Common Ancestor (LCA)" << endl;
    TreeNode* lca = lowestCommonAncestor(root, root->left->left, root->left->right);
    cout << "    LCA(1, 3) = " << lca->val << endl;
    lca = lowestCommonAncestor(root, root->left, root->right);
    cout << "    LCA(2, 6) = " << lca->val << endl;

    // 6. Path Sum
    cout << "\n[6] Path Sum" << endl;
    cout << "    Path with sum 7 exists: " << (hasPathSum(root, 7) ? "yes" : "no") << endl;
    cout << "    Path with sum 15 exists: " << (hasPathSum(root, 15) ? "yes" : "no") << endl;

    // 7. Complexity Summary
    cout << "\n[7] Complexity Summary" << endl;
    cout << "    | Operation  | Average   | Worst     |" << endl;
    cout << "    |------------|-----------|-----------|" << endl;
    cout << "    | Search     | O(log n)  | O(n)      |" << endl;
    cout << "    | Insert     | O(log n)  | O(n)      |" << endl;
    cout << "    | Delete     | O(log n)  | O(n)      |" << endl;
    cout << "    | Traversal  | O(n)      | O(n)      |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
