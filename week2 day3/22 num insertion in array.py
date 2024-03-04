def insert_element_at_index(lst, element, position):
    try:
        lst.insert(position, element)
        return lst
    except IndexError:
        return "Error: Invalid position. Index out of range."

# Input from the user
num_elements = int(input("Enter the number of elements: "))
user_elements = input("Enter the elements (comma-separated): ").split(",")
element_to_insert = int(input("Enter the element to be inserted: "))
insert_position = int(input("Position: "))

# Convert user input to integers
elements_list = [int(e) for e in user_elements]

# Insert the element
result_list = insert_element_at_index(elements_list, element_to_insert, insert_position)

# Print the modified list
print("Sample output:", result_list)
