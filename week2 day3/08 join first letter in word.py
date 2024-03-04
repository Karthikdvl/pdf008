def capitalize_and_join(sentence):
    return '.'.join(word.capitalize() for word in sentence.split())

if __name__ == "__main__":
    try:
        input_sentence = input("Enter the sentence: ")
        output = capitalize_and_join(input_sentence)
        print(f"Output: {output}")
    except ValueError:
        print("Invalid input. Please enter a valid sentence.")
