words = input("Enter a sentence: ").split()
print("Smallest:", min(words, key=len))
print("Largest:", max(words, key=len))
