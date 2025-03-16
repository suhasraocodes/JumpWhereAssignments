validate = lambda s: (any(c.isupper() for c in s) and any(c.islower() for c in s) and any(c.isdigit() for c in s) and len(s) >= 10)
print(validate('PaceWisd0m'))
