d = {1: 10, 2: 20, 3: 10}
res = {k: v for k, v in dict.fromkeys(d).items()}
print(res)
