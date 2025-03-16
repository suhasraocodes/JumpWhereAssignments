
numbers = []
while True:
    num = int(input("Enter a number (0 to finish): "))
    if num == 0:
        break
    numbers.append(num)

total = sum(numbers)
average = total / len(numbers) if numbers else 0
print(f"Sum: {total}, Average: {average}")

