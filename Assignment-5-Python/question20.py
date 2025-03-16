from itertools import groupby
s = "aaabbcddd"
print("".join(k for k, _ in groupby(s)))
