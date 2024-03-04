A = [[2, 1], [3, 4]]
B = [[31], [1, 2]]

C = [[0, 0], [0, 0]]
for i in range(len(A)):
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[i][j] += A[i][k] * B[k][j]

print("C matrix:")
for row in C:
    print(row)
