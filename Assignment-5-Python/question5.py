def swap_first_two(s1, s2):
    return s2[:2] + s1[2:] + " " + s1[:2] + s2[2:]
print(swap_first_two("abc", "xyz"))
