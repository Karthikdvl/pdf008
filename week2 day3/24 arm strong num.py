def is_armstrong_number(number):
    # Convert the number to a string to find its length (number of digits)
    num_str = str(number)
    num_digits = len(num_str)

    # Calculate the sum of the cubes of each digit
    armstrong_sum = sum(int(digit) ** num_digits for digit in num_str)

    # Check if the sum is equal to the original number
    if armstrong_sum == number:
        return True
    else:
        return False

# Input from the user
input_number = 153

# Check if it's an Armstrong number
if is_armstrong_number(input_number):
    print(f"{input_number} is an ARMSTRONG NUMBER.")
else:
    print(f"{input_number} is NOT an Armstrong number.")
