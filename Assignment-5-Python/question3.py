def first_last_chars(s):
    return s[:2] + s[-2:] if len(s) >= 2 else ""
print(first_last_chars("thisisniceone"))
print(first_last_chars("ab"))
print(first_last_chars("f"))
