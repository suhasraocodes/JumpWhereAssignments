lst = [19, 65, 57, 39, 152, 639, 121, 44, 90, 190]
filtered = list(filter(lambda x: x % 19 == 0 or x % 13 == 0, lst))
print(filtered)
