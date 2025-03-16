matrix = [[1, 2, 3], [2, 4, 5], [1, 1, 1]]
matrix.sort(key=lambda x: sum(x))
print(matrix)
