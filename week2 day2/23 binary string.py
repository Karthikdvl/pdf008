def add_binary_strings(a, b):
    a_decimal = int(a, 2)
    b_decimal = int(b, 2)

    sum_decimal = a_decimal + b_decimal

    sum_binary = bin(sum_decimal)

    return sum_binary[2:]

a = "11"
b = "1"
output = "100"

result = add_binary_strings(a, b)
print("Result:", result)

if result == output:
    print("Correct!")
else:
    print("Incorrect!")
