def shell_sort(lst):
    """
    shell sort
    """
    n = len(lst)
    gap = n // 2

    while gap:
        for i in range(gap, n):
            for j in range(i - gap, -1, -gap):
                if lst[j + gap] < lst[j]:
                    lst[j + gap], lst[j] = lst[j], lst[j + gap]
                else:
                    break
        gap //= 2

    # while gap:
    #     for i in range(gap, n):
    #         j = i
    #         while j > 0:
    #             if lst[j - gap] > lst[j]:
    #                 lst[j - gap], lst[j] = lst[j], lst[j - gap]
    #                 j -= gap
    #             else:
    #                 break
    #     gap //= 2




if __name__ == "__main__":
    lst = [5, 4, 7, 1, 12, 13, 23, 0, 4]
    print(lst)
    shell_sort(lst)
    print(lst)
