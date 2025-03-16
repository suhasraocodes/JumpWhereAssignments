
marks = int(input("Enter marks: "))
grade = "F" if marks < 25 else "E" if marks < 45 else "D" if marks < 50 else "C" if marks < 60 else "B" if marks < 80 else "A"
print(f"Grade: {grade}")

