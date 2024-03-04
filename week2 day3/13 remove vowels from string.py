def remove_vowels(input_string):
    vowels = "AEIOUaeiou"
    result = "".join(char for char in input_string if char not in vowels)
    return result

def main():
    try:
        user_input = input("Enter a string: ")
        modified_string = remove_vowels(user_input)
        print(f"The string without vowels is: {modified_string}")
    except ValueError:
        print("Invalid input. Please enter a valid string.")

if __name__ == "__main__":
    main()
