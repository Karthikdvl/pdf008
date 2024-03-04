
array = [14, 16, 87, 36, 25, 89, 34]
M = 1
N = 3

array.sort()

Mth_max = array[-M]

Nth_min = array[N-1]

sum = Mth_max + Nth_min

print("Mth Maximum Number =", Mth_max)
print("Nth Minimum Number =", Nth_min)
print("Sum =", sum)
