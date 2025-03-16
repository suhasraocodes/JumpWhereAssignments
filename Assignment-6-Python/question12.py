
classes_held = int(input("Enter number of classes held: "))
classes_attended = int(input("Enter number of classes attended: "))

percentage = (classes_attended / classes_held) * 100
print(f"Attendance: {percentage:.2f}%")
print("Allowed" if percentage >= 75 else "Not Allowed")

