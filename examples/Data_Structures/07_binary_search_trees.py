"""
07 Binary Search Trees
======================
Demonstrates BST insert, search, delete, min/max, and validation.
"""


class BSTNode:
    __slots__ = ('val', 'left', 'right')
    def __init__(self, val, left=None, right=None):
        self.val = val; self.left = left; self.right = right


class BST:
    def __init__(self):
        self.root = None

    def insert(self, val):
        self.root = self._insert(self.root, val)

    def _insert(self, node, val):
        if node is None: return BSTNode(val)
        if val < node.val: node.left = self._insert(node.left, val)
        elif val > node.val: node.right = self._insert(node.right, val)
        return node

    def search(self, val):
        node = self.root
        while node:
            if val == node.val: return True
            node = node.left if val < node.val else node.right
        return False

    def find_min(self):
        node = self.root
        while node.left: node = node.left
        return node.val

    def find_max(self):
        node = self.root
        while node.right: node = node.right
        return node.val

    def delete(self, val):
        self.root = self._delete(self.root, val)

    def _delete(self, node, val):
        if node is None: return None
        if val < node.val: node.left = self._delete(node.left, val)
        elif val > node.val: node.right = self._delete(node.right, val)
        else:
            if node.left is None: return node.right
            if node.right is None: return node.left
            succ = node.right
            while succ.left: succ = succ.left
            node.val = succ.val
            node.right = self._delete(node.right, succ.val)
        return node

    def inorder(self):
        result = []
        def walk(n):
            if n: walk(n.left); result.append(n.val); walk(n.right)
        walk(self.root)
        return result


def is_valid_bst(root, lo=float('-inf'), hi=float('inf')):
    if root is None: return True
    if root.val <= lo or root.val >= hi: return False
    return is_valid_bst(root.left, lo, root.val) and is_valid_bst(root.right, root.val, hi)


if __name__ == "__main__":
    bst = BST()
    values = [8, 3, 10, 1, 6, 14, 4, 7, 13]
    for v in values: bst.insert(v)
    print(f"Inserted: {values}")
    print(f"Inorder:  {bst.inorder()}")
    print(f"Min: {bst.find_min()}, Max: {bst.find_max()}")
    print(f"Search 6: {bst.search(6)}, Search 5: {bst.search(5)}")
    print(f"Valid BST: {is_valid_bst(bst.root)}")
    bst.delete(3)
    print(f"After delete(3): {bst.inorder()}")
    bst.delete(8)
    print(f"After delete(8): {bst.inorder()}")
