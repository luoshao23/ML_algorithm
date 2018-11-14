class node(object):

    def __init__(self, val, pnext=None, pprev=None):
        self.val = val
        self._next = pnext
        # self._prev = pprev

    def __repr__(self):
        return str(self.val)


class Chain(object):

    def __init__(self):
        self.head = None
        self.length = 0

    def __repr__(self):
        context = '<'
        start = self.head
        if not start:
            return context + '>'
        else:
            context += str(self.head)

        while start._next:
            context += ', ' + str(start._next)
            start = start._next
        return context + '>'

    def append(self, input_item):
        item = None
        if isinstance(input_item, node):
            item = input_item
        else:
            item = node(input_item)

        if not self.head:
            self.head = item
            self.length += 1
        else:
            start = self.head
            while start._next:
                start = start._next
            start._next = item
            self.length += 1

    def getIndex(self, val):
        start = self.head
        index = 0
        if self.length == 0:
            raise ValueError('%s is not in chain' % str(val))
        while start:
            if start.val == val:
                return index

            start = start._next
            index += 1
        raise ValueError('%s is not in chain' % str(val))

    def getVal(self, index):
        if self.length <= index:
            raise IndexError('chain index out of range')

        start_index = 0
        start = self.head
        while start_index < index:
            start_index += 1
            start = start._next
        return start.val

    def remove(self, val):
        start = self.head
        prev_start = None
        if self.length == 0:
            raise ValueError('remove(%s): %s is not in chain' %
                             (str(val), str(val)))
        while start:
            if start.val == val:
                if prev_start:
                    prev_start._next = start._next
                else:
                    self.head = start._next
                self.length -= 1
                del start
                return
            prev_start = start
            start = start._next

        raise ValueError('%s is not in chain' % str(val))
