def find_maximum_binary(num1, num2, num3):
    max_binary = ""
    for i in range(len(num1)):
        max_bit = max(num1[i], num2[i], num3[i])
        max_binary += max_bit
    return max_binary

def main():
    try:
        num1 = input("Enter the first binary number: ")
        num2 = input("Enter the second binary number: ")
        num3 = input("Enter the third binary number: ")

        if len(num1) != len(num2) or len(num1) != len(num3):
            print("All binary numbers must have the same length.")
            return

        max_result = find_maximum_binary(num1, num2, num3)
        print(f"Maximum Number: {max_result}")
    except ValueError:
        print("Invalid input. Please enter valid binary numbers.")

if __name__ == "__main__":
    main()
