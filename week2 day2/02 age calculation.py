user_name= input("enter your name ")

user_age = int(input("enter your age "))

present_age = int(input("enter present year"))

if (user_age <=0):
    print("invalid input , give positive integer")

elif (user_age >100):
    print("enter your age under 100 years")

else:
    remaining_age = 100 - user_age
    print(f"hello {user_name} , you will turn 100 in {remaining_age} years")
    print(f"you will turn 100 in {present_age + remaining_age}")

