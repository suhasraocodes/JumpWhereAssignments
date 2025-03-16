from collections import Counter
s = "thequickbrownfoxjumpsoverthelazydog"
c = Counter(s)
for k, v in c.items():
    if v > 1:
        print(k, v)
