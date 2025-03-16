lst = ['red', 'black', 'white', 'green', 'orange']
substring = 'ack'
filtered = list(filter(lambda x: substring in x, lst))
print(filtered)
