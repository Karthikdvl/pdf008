def decimal_to_binary(n):
    return bin(n)[2:]  # Remove the "0b" prefix

if __name__ == "__main__":
    decimal_num = 12
    binary_result = decimal_to_binary(decimal_num)
    print(f"Binary representation of {decimal_num}: {binary_result}")
