def count_vowels_and_consonants(statement):
    vowels = 0
    consonants = 0

    statement = statement.lower()

    for i in statement:
        if i.isalpha():
            if i in 'aeiou':
                vowels += 1
            else:
                consonants += 1

    return vowels, consonants

user_input = "Saveetha School of Engineering"
vowels_count, consonants_count = count_vowels_and_consonants(user_input)

print(f"Number of vowels = {vowels_count}")
print(f"Number of consonants = {consonants_count}")

if vowels_count > consonants_count:
    print("Vowels have the maximum count.")
elif consonants_count > vowels_count:
    print("Consonants have the maximum count.")
else:
    print("Both vowels and consonants have equal counts.")
