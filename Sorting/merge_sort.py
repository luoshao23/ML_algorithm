def merge_sort(lst):
    n = len(lst)
    if n <= 1:
        return lst
    mid = n // 2

    left = merge_sort(lst[:mid])
    right = merge_sort(lst[mid:])

    li, ri = 0, 0
    res = []
    while li < len(left) and ri < len(right):
        if left[li] < right[ri]:
            res.append(left[li])
            li += 1
        else:
            res.append(right[ri])
            ri += 1
    res += left[li:]
    res += right[ri:]
    return res


if __name__ == "__main__":
    lst = [5, 4, 7, 1, 12, 13, 23, 0, 4]
    print(lst)
    lst = merge_sort(lst)
    print(lst)
