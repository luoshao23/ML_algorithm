class BSTnode(object):
    """docstring for BSTnode"""

    def __init__(self, parent, t):

        self.parent = parent
        self.key = t
        self.left = None
        self.right = None
        self.size = 1

    def update_stats(self):
        self.size = (0 if self.left is None else self.left.size) + \
            (0 if self.right is None else self.right.size)

    def insert(self, t, NodeType):
        self.size += 1
        if t < self.key:
            if self.left is None:
                self.left = NodeType(self, t)
                return self.left
            else:
                return self.left.insert(t, NodeType)
        else:
            if self.right is None:
                self.right = NodeType(self, t)
                return self.right
            else:
                return self.right.insert(t, NodeType)

    def find(self, t):
        if t == self.key:
            return self
        elif t < self.key:
            if self.left is None:
                return None
            else:
                return self.left.find(t)

        else:
            if self.right is None:
                return None
            else:
                return self.right.find(t)

    def rank(self, t):
        """Return the number of keys <= t in the subtree rooted at this node."""
        left_size = 0 if self.left is None else self.left.size
        if t == self.key:
            return left_size + 1
        elif t < self.key:
            if self.left is None:
                return 0
            else:
                return self.left.rank(t)
        else:
            if self.right is None:
                return left_size + 1
            else:
                return self.right.rank(t) + left_size + 1

    def minimum(self):
        current = self
        while current.left is not None:
            current = current.left
        return current

    def successor(self):
        if self.right is not None:
            return self.right.minimum()
        current = self
        while current.parent is not None and current.parent.right is current:
            current = current.parent
        return current.parent

    def delete(self):
        if self.left is None or self.right is None:
            if self is self.parent.left:
                self.parent.left = self.left or self.right
                if self.parent.left is not None:
                    self.parent.left.parent = self.parent
            else:
                self.parent.right = self.left or self.right
                if self.parent.right is not None:
                    self.parent.right.parent = self.parent
            current = self.parent
            while current is not None:
                current.update_stats()
                current = current.parent
            return self, self.parent
        else:
            s = self.successor()
            self.key, s.key = s.key, self.key
            return s.delete()

        def check(self, lo, hi):
            if lo is not None and self.key <= lo:
                raise "BST RI violation"
            if hi is not None and self.key >= hi:
                raise "BST RI violation"
            if self.left is not None:
                if self.left.parent is not self:
                    raise "BST RI violation"
                self.left.check(lo, self.key)
            if self.right is not None:
                if self.right.parent is not self:
                    raise "BST RI violation"
                self.right.check(self.key, hi)
            if self.size != 1 + (0 if self.left is None else self.left.size) + (0 if self.right is None else self.right.size):
                raise "BST RI violation"

        def __repr__(self):
            return "BST Node, key: %s" % str(self.key)

class BST(object):
    """docstring for BST"""
    def __init__(self, NodeType=BSTnode):
        self.root = None
        self.NodeType = NodeType
        # self.psroot = self.NodeType(None, None)

    # def reroot(self):
    #     self.root = self.psroot.left

    def insert(self, t):
        if self.root is None:
            # self.psroot.left = self.NodeType(self.psroot, t)
            # self.reroot()
            self.root = self.NodeType(None, t)
            return self.root
        else:
            return self.root.insert(t, self.NodeType)

    def find(self, t):
        if self.root is None:
            return None
        else:
            return self.root.find(t)

    def rank(self, t):
        if self.root is None:
            return 0
        else:
            self.root.rank(t)

    def delete(self, t):
        """Delete the node for key t if it is in the tree."""
        node = self.find(t)
        deleted = node.delete()
        # self.reroot()
        return deleted

    def check(self):
        if self.root is not None:
            self.root.check(None, None)

    def __str__(self):
        node = self.root
        if node is None:
            return '<empty tree>'
        def recurse(node):
            if node is None: return [], 0, 0
            label = str(node.key)
            left_lines, left_pos, left_width = recurse(node.left)
            right_lines, right_pos, right_width = recurse(node.right)
            middle = max(right_pos + left_width - left_pos + 1, len(label),2)
            pos = left_pos + middle // 2
            width = left_pos + middle + right_width - right_pos
            label = label.center(middle, '=')
            if len(left_lines) < len(right_lines):
                left_lines.extend([' ' * left_width]*(len(right_lines) - len(left_lines)))
            if len(left_lines) > len(right_lines):
                right_lines.extend([' ' * right_width]*(len(left_lines) - len(right_lines)))
            lines = [' ' * left_pos + label + ' ' * (right_width - right_pos)]+\
                    [' ' * (left_pos)+'/'+' '*(middle-2)+'\\'+' '*(right_width - right_pos)]+\
                    [left_line+' '*(width - left_width - right_width)+right_line for left_line, right_line in zip(left_lines, right_lines)]
            return lines, pos, width
        return '\n'.join(recurse(self.root) [0])

def test():
    test1 = range(0, 100, 10)
    test2 = [31, 41, 59, 26, 53, 58, 97, 93, 23]

    mytree = BST()
    print mytree
    for x in test2:
        mytree.insert(x)
        print
        print mytree
    return mytree

