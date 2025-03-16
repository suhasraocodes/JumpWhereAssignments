def longest_word(words):
    return max(words, key=len)
print(longest_word(["apple", "banana", "strawberry"]))
