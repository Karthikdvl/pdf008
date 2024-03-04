def main():
    try:
        word = input("Enter the word: ")
        sorted_word = ''.join(sorted(word, reverse=True))
        print(f"Alphabetical Order: {sorted_word}")
    except ValueError:
        print("Invalid input. Please enter a valid word.")

if __name__ == "__main__":
    main()
