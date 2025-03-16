
salary = float(input("Enter salary: "))
years = int(input("Enter years of service: "))
bonus = salary * 0.05 if years > 5 else 0
print(f"Bonus: {bonus}")

