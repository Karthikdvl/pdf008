#  dictionary initialization
fruit_dict = {
    'Apple': 5,
    'Orange': 10,
    'Banana': 12
}

# Print the dictionary contents
for fruit, quantity in fruit_dict.items():
    print(f"{fruit} : {quantity}")

print(fruit_dict)
print(fruit_dict['Apple'])
print("Type -- ", type(fruit_dict))
