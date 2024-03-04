def check_is_perfect_number(number):
    divisors_sum = 0
    for i in range(1, number):
        if number % i == 0:
            divisors_sum += i
    return divisors_sum == number

input_number = 28
if check_is_perfect_number(input_number):
    print(f"{input_number} is a PERFECT NUMBER.")
else:
    print(f"{input_number} is NOT a perfect number.")
