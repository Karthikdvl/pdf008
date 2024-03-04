
list_of_elements = [16, -18, 27, -16, 23, -21, 19]

negative_count = 0

for element in list_of_elements:

    if element < 0:
        
        negative_count += 1


print("Number of negative numbers in List of elements =", negative_count)
