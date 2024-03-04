
age = int(input("Enter the passenger's age: "))


if age >= 0 and age <= 3:

    price = 0
elif age >= 4 and age <= 12:
 
    price = 10
elif age > 12:

    price = 20
else:
 
    print("Invalid age. Please enter a positive integer.")

print("The ticket price for the passenger is Rs", price)
