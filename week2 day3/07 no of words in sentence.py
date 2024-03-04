def count_words(sentence):
    words = sentence.split()
    
    return len(words)

def main():
    try:
        input_sentence = input("Enter the sentence: ")
        num_words = count_words(input_sentence)
        print(f"Number of words in the sentence: {num_words}")
    except ValueError:
        print("Invalid input. Please enter a valid sentence.")

if __name__ == "__main__":
    main()
