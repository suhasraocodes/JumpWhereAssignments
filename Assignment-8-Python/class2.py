class ValidParentheses:
    def is_valid(self, s):
        stack, pairs = [], {'(': ')', '{': '}', '[': ']'}
        for ch in s:
            if ch in pairs: stack.append(pairs[ch])
            elif not stack or stack.pop() != ch: return False
        return not stack

