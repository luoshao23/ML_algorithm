def select_sort(lst):
    n = len(lst)
    for i in range(n - 1):
        ind = i
        for j in range(i + 1, n):
            if lst[j] < lst[ind]:
                ind = j
        lst[ind], lst[i] = lst[i], lst[ind]

if __name__ == "__main__":
    lst = [5, 4, 7, 1, 12, 13, 23, 0, 4]
    print(lst)
    select_sort(lst)
    print(lst)

