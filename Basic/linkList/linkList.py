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
        if self.is_empty():
            return this
        else:
            while this.next:
                this = this.next
            return this

    def add(self, item):
        """add to the top of the list"""
        item.next = self._head
        self._head = item

    def append(self, item):
        """append to the tail of the list"""
        this = self.travel()
        if this is None:
            self._head = item
        else:
            this.next = item

    def insert(self, pos, item):
        if pos > self.length or pos < 0:
            raise IndexError("index of bound!")
        elif pos == 0:
            self.add(item)
        else:
            count = 0
            this = self._head
            while count < pos - 1:
                this = this.next
                count += 1
            item.next = this.next
            this.next = item

    def search(self, item):
        if not self.is_empty():
            this = self._head
            while this.next:
                if this.value == item.value:
                    return True
                this = this.next
        return False

    def remove(self, pos):
        if not self.is_empty():
            if pos > 0 and pos < self.length:
                count = 0
                this = self._head
                while count < pos - 1:
                    this = this.next
                    count += 1
                tmp = this.next
                this.next = tmp.next
                del tmp
            elif pos == 0:
                tmp = self._head
                self._head = tmp.next
                del tmp
            else:
                raise IndexError("Out of bound.")
        else:
            raise RuntimeError("Cannot remove item from an empty list.")



    # def insert(self, pos, item):
    #     if pos > self.length:
    #         raise IndexError("index of bound!")
    #     count = 0
    #     this = self._head
    #     slow_this = None
    #     while count < pos:
    #         slow_this = this
    #         this = this.next
    #         count += 1
    #     if self.is_empty():
    #         self._head = item
    #     else:
    #         slow_this.next = item
    #     item.next = this


if __name__ == "__main__":
    ll = SingleLinkedList()
    print("is empty?: ", ll.is_empty())
    for i in range(5):
        n = Node(i)
        ll.add(n)
    print(ll)
    print("is empty?: ", ll.is_empty())
    print(ll.length)

    # ll = SingleLinkedList()
    # ll.append(Node(5))
    # print(ll.length)
    ll.insert(0, Node(999))
    print(ll)
    ll.insert(5, Node(999))
    print(ll)
    ll.insert(ll.length, Node(999))
    print(ll)
    print(ll.length)
    print(ll.search(Node(0)))
    print(ll.search(Node(10)))
    print(SingleLinkedList().search(Node(10)))
    ll.remove(2)
    print(ll)
    ll.remove(0)
    print(ll)
    ll.remove(5)
    print(ll)
    ll.remove(5)


