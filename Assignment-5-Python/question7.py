def replace_not_poor(s):
    not_idx = s.find("not")
    poor_idx = s.find("poor")
    return s[:not_idx] + "good" + s[poor_idx+4:] if not_idx < poor_idx else s
print(replace_not_poor("The lyrics is not that poor!"))
