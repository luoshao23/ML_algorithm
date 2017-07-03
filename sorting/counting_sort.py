def counting_sort(A, k):
    """
    Counting sort: No comparison between elements
    input: A[0:n] len(A)=n
    output: B[0:n] len(B)=n
    auxiliary: C[0:k] len(C)=k
    """

    c=[0]*k
    for a in A:
        c[a] += 1
    print c
    for i in xrange(1, k):
        c[i] = c[i] + c[i-1]
    print c
    B = [None for a in A]
    print B
    for j in xrange(len(A)-1, -1, -1):

        B[c[A[j]]-1] = A[j]
        c[A[j]] -= 1

    return B

def main():
    lst = [2,5,2,4,5]
    k = max(lst)+1
    print lst
    lst = counting_sort(lst, k)
    print lst

if __name__ == '__main__':
    main()