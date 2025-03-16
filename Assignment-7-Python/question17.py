d1 = {'key1': 1, 'key2': 3, 'key3': 2}
d2 = {'key1': 1, 'key2': 2}
for k, v in d1.items():
    if k in d2 and d2[k] == v:
        print(f'{k}: {v} is present in both')
