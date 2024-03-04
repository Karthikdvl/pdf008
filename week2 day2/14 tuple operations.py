
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)
element = 3

concatenated_tuple = tuple1 + tuple2
print("Concatenated tuple =", concatenated_tuple)

index = concatenated_tuple.index(element)
print("Index of element", element, ":", index)

count = concatenated_tuple.count(element)
print("Number of occurrences of element", element, "is :", count)
