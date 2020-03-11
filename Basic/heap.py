from copy import copy

class Heap(object):

    def __init__(self, lst=None):
        if lst:
            if isinstance(lst, list):
                self._heap = lst
                self.heapify()
            else:
                raise TypeError('wrong input type')
        else:
            self._heap = []

    def __repr__(self):
        return str(self._heap)

    __str__ = __repr__

    def is_empty(self):
        return len(self._heap) == 0

    def find_min(self):
        if self._heap:
            return self._heap[0]

    def insert(self, e):
        self._heap.append(e)
        self._percolateup(0, len(self._heap) - 1)

    def _percolateup(self, start, pos):

        last = self._heap[pos]
        while pos > start and last < self._heap[(pos - 1) // 2]:
            self._heap[pos] = self._heap[(pos - 1) // 2]
            pos = (pos - 1) // 2
        self._heap[pos] = last

    def delete_min(self):
        last = self._heap.pop()
        if self._heap:
            item = self._heap[0]
            self._heap[0] = last
            self._percolatedown(0)
            return item
        return last

    def _percolatedown(self, pos):
        e = self._heap[pos]
        startpos = pos
        lastpos = len(self._heap) - 1
        childpos = 2*pos + 1
        while childpos < lastpos:
            rightchild = childpos + 1
            if rightchild <= lastpos and self._heap[rightchild] < self._heap[childpos]:
                childpos = rightchild
            if self._heap[childpos] < e:
                self._heap[pos] = self._heap[childpos]
            else:
                break
            pos = childpos
            childpos = 2*pos + 1
        self._heap[pos] = e

    def heapify(self):
        for i in reversed(range(len(self._heap)//2)):
            self._percolatedown(i)

    def heapsort(self):
        self.heapify()

        res = []
        for i in reversed(range(len(self._heap))):
            res.append(self.delete_min())
        return res



if __name__ == "__main__":
    # h = Heap()
    # h.insert(5)
    # h.insert(7)
    # h.insert(3)
    # h.insert(1)
    # h.insert(6)
    # print(h)
    # for _ in range(5):
    #     h.delete_min()
    #     print(h)

    # k = Heap([4, 3, 2, 7, 1])
    k = Heap([5, 7, 3, 1, 6])
    print(k)
    res = k.heapsort()
    print(res)
