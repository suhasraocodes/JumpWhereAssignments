def add_ing(s):
    return s + "ing" if len(s) >= 3 and not s.endswith("ing") else (s + "ly" if s.endswith("ing") else s)
print(add_ing("abc"))
print(add_ing("string"))
