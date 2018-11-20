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

if __name__ == "__main__":
    # lst = [5, 4, 7, 1, 12, 13, 23, 0, 4]
    lst = [5, 4, 4]
    print(lst)
    insert_sort(lst)
    print(lst)
