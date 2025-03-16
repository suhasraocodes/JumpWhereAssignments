def reverse_if_multiple_of_4(s):
    return s[::-1] if len(s) % 4 == 0 else s
print(reverse_if_multiple_of_4("abcd"))
