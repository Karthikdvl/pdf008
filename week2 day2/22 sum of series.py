def factorial(n):
  
    if n == 0 or n == 1:
        return 1
 
    else:
        return n * factorial(n-1)

# Define a function to calculate the sum of the series 1!+2!+...+N!
def sum_of_series(N):

    if N == 0:
        return 0

    else:
        return factorial(N) + sum_of_series(N-1)

N = int(input("Enter the number of terms: "))

sum = sum_of_series(N)

print("Sum =", sum)
