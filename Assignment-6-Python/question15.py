
nums = []
while True:
    inp = input("Enter number (press q to quit): ")
    if inp.lower() == q:
        break
    nums.append(int(inp))

product = 1
for n in nums:
    product *= n

print(f"Average: {sum(nums) / len(nums) if nums else 0}, Product: {product}")

