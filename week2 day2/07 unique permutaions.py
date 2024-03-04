from itertools import permutations

def unique_permutations(num):
    
    num_str = str(num)
    
    permute = set(permutations(num_str))
    
    print("Permutations are:")
    for i in permute:
        print("".join(i))

number = 143
unique_permutations(number)
