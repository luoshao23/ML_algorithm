
# http://courses.csail.mit.edu/6.006/spring11/notes.shtml


class BSNode(object):
    """docstring for BSTree"""

    def __init__(self, key=None, left=None, right=None, parent=None):
        self.key = key
        self.disconnect()
    def disconnect(self):
        self.left = None
        self.right = None
        self.parent = None


class BST(object):
    """docstring for BST"""

    def __init__(self):
        self.root = None

    def insert(self, a):
        new = BSNode(a)
        if self.root is None:
            self.root = new
        else:
            node = self.root
            while True:
                if a < node.key:
                    # left tree
                    if node.left is None:
                        node.left = new
                        new.parent = node
                        break
                    node = node.left
                else:
                    # right tree
                    if node.right is None:
                        node.right = new
                        new.parent = node
                        break
                    node = node.right
        return new

    def find(self, v):
        node = self.root
        while node is not None:
            if node.key == v:
                return node
            elif v < node.key:
                node = node.left
            else:
                node = node.right
        return None

    def del_min(self):
        if self.root is None:
            return None, None
        else:
            node = self.root
            while node.left is not None:
                node = node.left
            if node.parent is not None:
                node.parent.left = node.right
            else:
                self.root = node.right
            if node.right is not None:
                node.right.parent = node.parent
            parent = node.parent
            node.disconnect()
        return node, parent

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




def get_height(node):
    if node is None:
        return 0
    return max(get_height(node.left), get_height(node.right)) + 1


def get_width(node):
    if node is None:
        return 0
    return 1 + get_width(node.left) + get_width(node.right)


def test(args=None, BSTtype=BST):
    import random, sys
    if not args:
        args = sys.argv[1:]
    if not args:
        print 'usage: %s <number-of-random-items | item item item ...>' % \
              sys.argv[0]
        sys.exit()
    elif len(args) == 1:
        items = (random.randrange(100) for i in xrange(int(args[0])))
    else:
        items = [int(i) for i in args]

    tree = BSTtype()
    print tree
    for item in items:
        tree.insert(item)
        print
        print tree
    return tree

if __name__ == '__main__': test()
