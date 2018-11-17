import copy

class Node(object):
    def __init__(self, value):
        self.value = value
        self.next = None

    def __repr__(self):
        return "(%s)" % self.value

    __str__ = __repr__


class SingleLinkedList(object):
    def __init__(self, item=None):
        self._head = item

    def is_empty(self):
        return self._head is None

    def __repr__(self):
        this = self._head
        strs = ""
        while this:
            strs += str(this) + "->"
            this = this.next
        strs += str(this)
        return strs

    __str__ = __repr__

    @property
    def length(self):
        count = 0
        this = self._head
        while this:
            count += 1
            this = this.next
        return count

    def travel(self):
        this = self._head
        slow_this = None
        while this:
            slow_this = this
            this = this.next

        return slow_this

    def add(self, item):
        """add to the top of the list"""
        item.next = self._head
        self._head = item

    def append(self, item):
        """append to the tail of the list"""
        this = self.travel()
        if hasattr(this, 'next'):
            this.next = item
        else:
            self._head = item

    def insert(self, pos, item):
        if pos > self.length:
            raise IndexError("index of bound!")
        count = 0
        this = self._head
        slow_this = None
        while count < pos:
            slow_this = this
            this = this.next
            count += 1

        if hasattr(slow_this, 'next'):
            slow_this.next = item
        else:
            self._head = item
        item.next = this


if __name__ == "__main__":
    ll = SingleLinkedList()
    for i in range(5):
        n = Node(i)
        ll.add(n)
    print(ll)
    print(ll.length)

    # ll = SingleLinkedList()
    # ll.append(Node(5))
    # print(ll)
    # print(ll.length)

    ll.insert(5, Node(999))
    print(ll)
    print(ll.length)


