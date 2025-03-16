lst = [19, 'red', 12, 'green', 'blue', 10, 'white', 'green', 1]
lst.sort(key=lambda x: (isinstance(x, str), x))
print(lst)
