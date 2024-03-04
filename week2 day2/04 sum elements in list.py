def sum_list_elements(my_list):
    return sum(my_list)

sample_list = [8, 2, 3, 0, 7]
result = sum_list_elements(sample_list)
print(f"Sum of all elements in the list is: {result}")

'''
method 2

def sum_list_elements(my_list):
    total = 0
    for element in my_list:
        total += element
    return total


sample_list = [8, 2, 3, 0, 7]
result = sum_list_elements(sample_list)
print(f"Sum of all elements in the list is: {result}")
'''

