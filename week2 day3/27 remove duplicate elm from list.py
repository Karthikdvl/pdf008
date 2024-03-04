def remove_duplicates(input_list):
    # Create an empty list to store unique elements
    unique_list = []

    # Iterate through the input list
    for item in input_list:
        # If the item is not already in the unique list, add it
        if item not in unique_list:
            unique_list.append(item)

    return unique_list

# Input: Number of elements in the list
num_elements = int(input("Enter the number of elements in the list: "))

# Initialize an empty list
my_list = []

# Input: Elements of the list
for i in range(num_elements):
    element = int(input(f"Enter element{i + 1}: "))
    my_list.append(element)

# Remove duplicates
result_list = remove_duplicates(my_list)

# Output: Non-duplicate items
print(f"Non-duplicate items: {result_list}")
