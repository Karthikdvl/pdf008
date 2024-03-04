input_tuple = (3, 6, 8, 9, 8, 9, 12, 35, 8)

target_number = 8

count = 0

for num in input_tuple:
    if num == target_number:
        count += 1

print(f"Sample Output: {count}")


'''
def count_occurrences(tup, target):
    count = 0
    for num in tup:
        if num == target:
            count += 1
    return count

# Sample input
input_tuple = (3, 6, 8, 9, 8, 9, 12, 35, 8)
target_number = 8

# Calculate the count
result = count_occurrences(input_tuple, target_number)

# Print the output
print(f"Sample Output: {result}")

'''
