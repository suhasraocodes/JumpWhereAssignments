
evens = [x for x in range(1, 101) if x % 2 == 0]
odds = [x for x in range(1, 101) if x % 2 != 0]
primes = [x for x in range(2, 101) if all(x % d != 0 for d in range(2, int(x ** 0.5) + 1))]

print(f"Evens: {evens}")
print(f"Odds: {odds}")
print(f"Primes: {primes}")

