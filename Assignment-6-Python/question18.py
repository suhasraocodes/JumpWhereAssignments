
evens = [x for x in range(1, 101) if x % 2 == 0]
div_by_4 = [x for x in evens if x % 4 == 0]
div_by_6 = [x for x in evens if x % 6 == 0]
div_by_8 = [x for x in evens if x % 8 == 0]
div_by_10 = [x for x in evens if x % 10 == 0]

print(f"Divisible by 4: {div_by_4}")
print(f"Divisible by 6: {div_by_6}")
print(f"Divisible by 8: {div_by_8}")
print(f"Divisible by 10: {div_by_10}")

