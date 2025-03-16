
lst = list(map(int, input("Enter space-separated numbers: ").split()))
to_remove = int(input("Enter number to remove: "))

if to_remove in lst:
    lst.remove(to_remove)

print(lst)

