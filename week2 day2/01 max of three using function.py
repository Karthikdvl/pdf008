def find_maximum(a, b, c):

    return max(a, b, c)

num1 = int(input("enter n1 "))
num2 = int(input("enter n2 "))
num3 = int(input("enter n3 "))

result = find_maximum(num1, num2, num3)

print(f"The maximum of {num1}, {num2}, and {num3} is: {result}")
