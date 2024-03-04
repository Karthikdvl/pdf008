
statement = "Raju Birthday @ September 24, #&$% is the wishes code for him."

special_count = 0

for char in statement:

    if not char.isalnum():
       
        special_count += 1

print("Number of special characters:", special_count)
