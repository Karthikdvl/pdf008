def calculate_sums(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Initialize lists to store row and column sums
    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(matrix[i][j] for i in range(num_rows)) for j in range(num_cols)]

    # Calculate diagonal sum (main diagonal)
    diag_sum = sum(matrix[i][i] for i in range(min(num_rows, num_cols)))

    return row_sums, col_sums, diag_sum

# Given matrix
a = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Calculate sums
row_sums, col_sums, diag_sum = calculate_sums(a)

# Print results
for i, row_sum in enumerate(row_sums, start=1):
    print(f"Sum of {i} row: {row_sum}")

for j, col_sum in enumerate(col_sums, start=1):
    print(f"Sum of {j} column: {col_sum}")

print(f"Diagonal sum: {diag_sum}")
