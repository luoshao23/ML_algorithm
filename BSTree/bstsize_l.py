import bst_l

def size(node):
    if node is None:
        return 0
    else:
        return node.size

def update_size(node):
    node.size = size(node.left) + size(node.right) + 1


class BST(bst_l.BST):

    def __init__(self):
        self.root = None

    def insert(self, t):
        """Insert key t into this BST, modifying it in-place."""
        node = bst_l.BST.insert(self, t)
        while node is not None:
            update_size(node)
            node = node.parent

    def delete_min(self):
        """Delete the minimum key (and return the old node containing it)."""
        _, parent = bst.BST.delete_min(self)
        node = parent
        while node is not None:
            update_size(node)
            node = node.parent
        return deleted, parent

def test(args=None):
    t = bst_l.test(args, BSTtype=BST)
    return t

if __name__ == '__main__': test()
