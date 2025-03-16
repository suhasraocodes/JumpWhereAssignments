import random

# Step 1: Read words from the file
def load_words(filename="words.txt"):
    with open(filename, "r") as file:
        return [line.strip().upper() for line in file]

# Step 2: Choose a random word
def choose_word(word_list):
    return random.choice(word_list)

# Step 3: Play Hangman
def play_hangman(word_list):
    word = choose_word(word_list)  # Random word
    guessed_letters = set()  # Keep track of guessed letters
    correct_letters = set(word)  # Unique letters in the word
    attempts = 6  # Incorrect guess limit
    display = ["_"] * len(word)  # Hidden word representation

    print("\n>>> Welcome to Hangman!")
    
    while attempts > 0:
        print("\n" + " ".join(display))  # Show current progress
        guess = input(">>> Guess your letter: ").upper()

        if guess in guessed_letters:
            print("You've already guessed this letter. Try again.")
            continue  # Don't penalize for repeated guesses

        guessed_letters.add(guess)  # Store guessed letter

        if guess in correct_letters:
            for i, letter in enumerate(word):
                if letter == guess:
                    display[i] = guess  # Reveal correct letters
        else:
            attempts -= 1
            print(f"Incorrect! You have {attempts} chances left.")

        if "_" not in display:  # All letters guessed
            print("\nCongratulations! You guessed the word:", word)
            break
    else:
        print("\nGame Over! The word was:", word)

    # Ask to play again
    if input("\nPlay again? (y/n): ").lower() == "y":
        play_hangman(word_list)

# Run the game
if __name__ == "__main__":
    words = load_words()
    play_hangman(words)
