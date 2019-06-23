def dot(A, B):
    rowA = len(A)
    rowB = len(B)

    if rowA == 0 or rowB == 0:
        raise ValueError('Empty Matrix')

    colA = len(A[0])
    colB = len(B[0])

    if colA != rowB:
        raise ValueError('Matrix shape dismatch with shape (%d, %d) (%d, %d)' % (
            rowA, colA, rowB, colB))

    C = [[0 for _ in range(colB)] for _ in range(rowA)]

    for i in range(rowA):
        for j in range(colB):
            for k in range(colA):
                C[i][j] += A[i][k] * B[k][j]

    return C

if __name__ == "__main__":
    ma = [[2, 3, 4], [3, 1, 0]]
    mb = [[1, 1], [5, 7], [1, 9]]

    print(dot(ma, mb))
