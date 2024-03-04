def count_matching_characters(s1, s2):
    # Ensure both strings have the same length
    if len(s1) != len(s2):
        return 0

    # Initialize the count of matching characters
    match_count = 0

    # Compare characters at each index
    for i in range(len(s1)):
        if s1[i] == s2[i]:
            match_count += 1

    return match_count

def main():
    try:
        s1 = input("Enter the first string: ")
        s2 = input("Enter the second string: ")

        num_matches = count_matching_characters(s1, s2)
        print(f"Number of matching characters: {num_matches}")
    except ValueError:
        print("Invalid input. Please enter valid strings.")

if __name__ == "__main__":
    main()
