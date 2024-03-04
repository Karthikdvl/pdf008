def count_digits(n):
    count = 0
    while n > 0:
        n //= 10
        count += 1
    return count

def is_tech_number(n):
    num_digits = count_digits(n)
    if num_digits % 2 != 0:
        return False

    half = num_digits // 2
    first_half = n // 10**half
    second_half = n % 10**half
    sum_of_halves = first_half + second_half

    return sum_of_halves**2 == n

def main():
    try:
        number = int(input("Enter a number: "))
        if is_tech_number(number):
            print("Given number is Tech number")
        else:
            print("Given number is not a Tech number")
    except ValueError:
        print("Invalid input. Please enter a valid integer.")

if __name__ == "__main__":
    main()
