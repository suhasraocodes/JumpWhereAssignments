def is_balanced(s):
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}  # Matching pairs

    for char in s:
        if char in "({[":
            stack.append(char)  # Push opening brackets
        elif char in ")}]":
            if not stack or stack.pop() != mapping[char]:
                return False  # Unmatched closing bracket
    
    return not stack  # True if stack is empty (all matched)

# Example Usage
s = input("Enter a string: ")
print(is_balanced(s))
