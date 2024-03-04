
arr = [16, 18, 27, 16, 23, 21, 19]

mean = sum(arr) / len(arr)
print(f"Mean / Average is: {mean:.1f}")

arr.sort()
n = len(arr)
if n % 2 == 0:
    median1 = arr[n // 2]
    median2 = arr[n // 2 - 1]
    median = (median1 + median2) / 2
else:
    median = arr[n // 2]
print(f"Median is: {median}")

from collections import Counter
data = Counter(arr)
max_count = max(data.values())
mode = [k for k, v in data.items() if v == max_count]
if len(mode) == len(arr):
    print("No mode found")
else:
    print(f"Mode is / are: {', '.join(map(str, mode))}")
