def main():
    try:
        input_string = input("Enter the string: ")
        length = len(input_string)
        print(f"Length of the String: {length}")
    except ValueError:
        print("Invalid input. Please enter a valid string.")

if __name__ == "__main__":
    main()
