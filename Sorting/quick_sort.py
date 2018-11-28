def quick_sort(lst, start, end):

    if start >= end: return
    mid = lst[start]
    low, high = start, end

    while low < high:
        while low < high and lst[high] >= mid:
            high -= 1
        lst[low] = lst[high]
        while low < high and lst[low] < mid:
            low += 1
        lst[high] = lst[low]


    lst[high] = mid
    quick_sort(lst, start, low - 1)
    quick_sort(lst, low + 1, end)

if __name__ == "__main__":
    ll = [5, 4, 13, 27, 9, 7, 8]
    print(ll)
    quick_sort(ll, 0, len(ll) - 1)
    print(ll)
