def linear_search(arr, target):
    for item in arr:
        if item == target:
            return True
    return False

input_list = [2, 4, 6, 8, 9, 7, 9]
target_element_1 = 7
target_element_2 = 5

result_1 = "Exist" if linear_search(input_list, target_element_1) else "not exist"
result_2 = "Exist" if linear_search(input_list, target_element_2) else "not exist"

print(f"Target {target_element_1}: {result_1}")
print(f"Target {target_element_2}: {result_2}")
