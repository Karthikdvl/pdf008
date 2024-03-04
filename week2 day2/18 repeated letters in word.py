
word = input("Enter the word: ")

counts = {}

for letter in word:
  
    if letter in counts:
        counts[letter] += 1
  
    else:
        counts[letter] = 1

repeated = 0


for letter, count in counts.items():
 
    if count > 1:
    
        repeated += 1
        print("Repeated letter =", letter)
        print("Count =", count)

print("Number of repeated letters =", repeated)
