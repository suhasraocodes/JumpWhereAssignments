
lst = [1, "hello", 2.5, "world", 3, 4.5]
ints = [x for x in lst if isinstance(x, int)]
strings = [x for x in lst if isinstance(x, str)]
floats = [x for x in lst if isinstance(x, float)]

print(f"Integers: {ints}")
print(f"Strings: {strings}")
print(f"Floats: {floats}")

