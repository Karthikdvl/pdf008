def sumsquare(l):
    odd_sum = 0
    even_sum = 0

    for item in l:
        square = item ** 2
        if item % 2 == 0:
            even_sum += square
        else:
            odd_sum += square

    return [odd_sum, even_sum]

try:
    num_elements = int(input("Enter the number of elements: "))
    if num_elements <= 0:
        print("Number of elements must be a positive integer.")
    else:
        list1 = []
        for _ in range(num_elements):
            element = int(input("Enter the element: "))
            list1.append(element)

        result = sumsquare(list1)
        print("Output:")
        print(result)
except ValueError:
    print("Invalid input. Please enter valid integers.")
