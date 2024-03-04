def calculate_sum_of_digits(number):
    # Calculate the sum of digits
    sum_of_digits = 0
    while number > 0:
        digit = number % 10
        sum_of_digits += digit
        number //= 10
    return sum_of_digits

def check_for_harshad_number(number):
    sum_of_digits = calculate_sum_of_digits(number)
    if number % sum_of_digits == 0:
        return True
    else:
        return False

# Input from the user
input_number = 153

# Check if it's a Harshad number
if check_for_harshad_number(input_number):
    print(f"{input_number} is a HARSHAD NUMBER.")
else:
    print(f"{input_number} is NOT a Harshad number.")
