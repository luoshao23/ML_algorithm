import time

def insert_sort(lst):
    """
    best: O(n)
    worst: O(n^2)
    stable
    """
    n = len(lst)
    for i in range(1, n):
        for j in range(i - 1, -1, -1):
            if lst[j + 1] < lst[j]:
                lst[j + 1], lst[j] = lst[j], lst[j + 1]
            else:
                break


def insert_sort_better(lst):
    """
    best: O(n)
    worst: O(n^2)
    stable
    """
    for i in range(1, len(lst)):
        cur = lst[i]
        j = i
        while j > 0 and lst[j - 1] > cur:
            lst[j] = lst[j - 1]
            j -= 1
        lst[j] = cur


if __name__ == "__main__":
    st = time.time()
    for _ in range(10000):
        lst = [5, 4, 7, 1, 12, 13, 23, 0, 4]
        insert_sort_better(lst)
    print(time.time() - st)
