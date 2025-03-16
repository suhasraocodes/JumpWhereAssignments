lst = [[10, 20], [40], [30, 56, 25], [10, 20], [33], [40]]
res = []
[res.append(i) for i in lst if i not in res]
print(res)
