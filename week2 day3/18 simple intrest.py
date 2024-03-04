def calculate_simple_interest(principal, years, senior_citizen):
    # Define the rate of interest
    if senior_citizen.lower() == 'y':
        rate = 15  # Senior citizen rate
    else:
        rate = 12  # Regular rate (for non-senior citizens)

    # Calculate simple interest
    interest = (principal * rate * years) / 100

    return interest

# Input from the user
principal_amount = float(input("Enter the principal amount: "))
no_of_years = int(input("Enter the number of years: "))
gender = input("Gender (m/f): ")
senior_citizen = input("Is the customer a senior citizen (y/n): ")

# Calculate interest
interest_amount = calculate_simple_interest(principal_amount, no_of_years, senior_citizen)

# Display the result
print(f"Interest: {interest_amount:.2f}")
