from itertools import permutations


def check(lst, res):
    n = len(lst)
    if n == 1:
        if lst[0] == 24:
            print(res)
            return True
        else:
            return False
    else:
        for comb in permute(lst):
            a = comb.pop()
            b = comb.pop()
            # if len(res) > 0:
            #     res.append(b)
            # else:
            res += [a, b]

            if check(comb + [a+b], res + ['+']):
                # print(res)
                return True
            if check(comb + [a-b], res + ['-']):
                # print(res)
                return True
            if check(comb + [a*b], res + ['*']):
                # print(res)
                return True
            if b != 0:
                if check(comb + [a/b], res + ['/']):
                    # print(res)
                    return True
    return False


def check24(lst):
    return check(lst, [])


def permute(lst):
    n = len(lst)
    if n == 1:
        yield lst
    for i in range(n):
        res = [lst[i]]
        for comb in permute(lst[:i] + lst[i+1:]):
            yield res + comb



if __name__ == "__main__":
    print(check24([5, 5, 1, 5]))
    print(check24([2, 3, 4, 5]))

    # for k in permute([5, 3, 2]):
    #     print(k)
