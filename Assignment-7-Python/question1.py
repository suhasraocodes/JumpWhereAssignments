d = {2: 30, 1: 20, 3: 10}
print(dict(sorted(d.items(), key=lambda x: x[1])))
print(dict(sorted(d.items(), key=lambda x: x[1], reverse=True)))
