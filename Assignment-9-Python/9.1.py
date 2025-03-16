import random

def generate_number():
    """Generate a random 4-digit number with unique digits."""
    digits = list(range(10))
    random.shuffle(digits)
    return ''.join(map(str, digits[:4]))

def count_cows_and_bulls(secret, guess):
    """Count cows (correct digit, correct place) and bulls (correct digit, wrong place)."""
    cows = sum(1 for i in range(4) if secret[i] == guess[i])
    bulls = sum(1 for digit in guess if digit in secret) - cows
    return cows, bulls

def is_valid_guess(guess):
    """Check if the guess is a valid 4-digit number with unique digits."""
    return len(guess) == 4 and guess.isdigit() and len(set(guess)) == 4

def main():
    print("Welcome to the Cows and Bulls Game!")
    secret_number = generate_number()
    attempts = 0
    
    while True:
        guess = input("Enter a 4-digit number with unique digits: ")
        if not is_valid_guess(guess):
            print("Invalid input! Please enter a 4-digit number with unique digits.")
            continue
        
        attempts += 1
        cows, bulls = count_cows_and_bulls(secret_number, guess)
        
        print(f"{cows} cows, {bulls} bulls")
        
        if cows == 4:
            print(f"Congratulations! You guessed the number {secret_number} in {attempts} attempts.")
            break

if __name__ == "__main__":
    main()
