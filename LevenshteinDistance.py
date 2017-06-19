def LevenshteinDistance(str_a, len_a, str_b,  len_b):
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    if str_a[len_a - 1] == str_b[len_b - 1]:
        cost = 0
    else:
        cost = 1
    return min(LevenshteinDistance(str_a, len_a - 1, str_b, len_b) + 1,
               LevenshteinDistance(str_a, len_a, str_b, len_b - 1) + 1,
               LevenshteinDistance(str_a, len_a - 1, str_b, len_b - 1) + cost
               )


def test():
    a = 'asa'
    b = 'aa'
    print LevenshteinDistance(a, len(a), b, len(b))

if __name__ == '__main__':
    test()
