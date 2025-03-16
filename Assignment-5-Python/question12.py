def convert_upper(s):
    return s.upper() if sum(1 for c in s[:4] if c.isupper()) >= 2 else s
print(convert_upper("AbCDxyz"))
