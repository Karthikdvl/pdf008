import math

def calculate_gcd(a, b):
    return math.gcd(a, b)

def calculate_lcm(a, b):

    return (a * b)

def main():
    try:
        N = int(input("Enter the number of elements: "))
        if N < 2:
            print("Please enter at least 2 numbers.")
            return

        numbers = []
        for i in range(1, N + 1):
            num = int(input(f"Enter Number {i}: "))
            numbers.append(num)

        gcd_result = calculate_gcd(numbers[0], numbers[1])
        lcm_result = calculate_lcm(numbers[0], numbers[1])

        print(f"LCM = {lcm_result}")
        print(f"GCD = {gcd_result}")
    except ValueError:
        print("Invalid input. Please enter valid integers.")

if __name__ == "__main__":
    main()
