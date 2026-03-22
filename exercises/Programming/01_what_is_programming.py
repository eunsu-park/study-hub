"""
Exercises for Lesson 01: What Is Programming
Topic: Programming

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Decomposition ===
# Problem: Break down a library management system into computational steps.

def exercise_1():
    """Solution: Decompose a library management system."""
    # Decomposition means breaking a large problem into smaller, manageable sub-problems.
    # A library system naturally divides into components, data, and operations.

    components = {
        "Main Components": [
            "Book Catalog - stores all book information",
            "Member Management - handles library member registration and records",
            "Borrowing System - manages checkout/return workflows",
            "Search Engine - allows finding books by various criteria",
            "Notification System - sends reminders for due dates",
        ],
        "Data to Store": [
            "Books: title, author, ISBN, genre, quantity, location",
            "Members: name, ID, contact info, membership status",
            "Transactions: borrow date, return date, book ID, member ID",
            "Fines: overdue days, amount, payment status",
        ],
        "Operations Needed": [
            "Add/remove/update book records",
            "Register/deactivate members",
            "Check out a book (decrement available count)",
            "Return a book (increment available count, calculate fines)",
            "Search books by title, author, ISBN, or genre",
            "Generate reports (popular books, overdue items, revenue)",
            "Send overdue notifications",
        ],
    }

    for category, items in components.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  - {item}")


# === Exercise 2: Abstraction ===
# Problem: Describe "Send an email" at different levels of abstraction.

def exercise_2():
    """Solution: Describe sending an email at multiple abstraction levels."""
    # Abstraction is about focusing on essential features while hiding irrelevant details.
    # Each level reveals more implementation detail.

    levels = {
        "High-level (User perspective)": [
            "1. Open email application",
            "2. Click 'Compose'",
            "3. Type recipient address, subject, and body",
            "4. Click 'Send'",
            "5. See confirmation that email was sent",
        ],
        "Mid-level (Application perspective)": [
            "1. Validate recipient email address format",
            "2. Encode message body (handle attachments, HTML formatting)",
            "3. Authenticate with SMTP server using credentials",
            "4. Construct email headers (From, To, Subject, Date, MIME type)",
            "5. Submit message to SMTP server's outgoing queue",
            "6. Handle server response (success, auth failure, rate limit)",
            "7. Update UI to reflect sent status",
        ],
        "Low-level (Network/Protocol perspective)": [
            "1. DNS lookup to resolve SMTP server IP address (MX records)",
            "2. Open TCP connection to server on port 587 (or 465 for SSL)",
            "3. Perform TLS handshake for encrypted communication",
            "4. SMTP EHLO/HELO handshake with server",
            "5. Authenticate (AUTH LOGIN with base64-encoded credentials)",
            "6. MAIL FROM: <sender> command",
            "7. RCPT TO: <recipient> command",
            "8. DATA command followed by RFC 5322 formatted message",
            "9. Server queues message, relays to recipient's mail server",
            "10. Close TCP connection (QUIT command)",
        ],
    }

    for level, steps in levels.items():
        print(f"\n{level}:")
        for step in steps:
            print(f"  {step}")


# === Exercise 3: Algorithmic Thinking ===
# Problem: Write an algorithm to determine if a word is a palindrome.

def exercise_3():
    """Solution: Palindrome checker algorithm and implementation."""
    # Algorithm in plain English:
    # 1. Convert the word to lowercase (case-insensitive comparison)
    # 2. Set two pointers: one at the start, one at the end
    # 3. Compare characters at both pointers
    # 4. If they differ, it's NOT a palindrome
    # 5. Move pointers inward (start forward, end backward)
    # 6. Repeat until pointers meet or cross
    # 7. If all comparisons matched, it IS a palindrome

    def is_palindrome(word):
        """Check if a word reads the same forwards and backwards."""
        word = word.lower()
        left = 0
        right = len(word) - 1

        while left < right:
            if word[left] != word[right]:
                return False
            left += 1
            right -= 1

        return True

    # Test cases
    test_words = ["racecar", "hello", "madam", "python", "kayak", "a", ""]
    for word in test_words:
        result = is_palindrome(word)
        print(f"  '{word}' -> {result}")


# === Exercise 4: Problem Solving ===
# Problem: Count how many times a specific word appears in a sentence.

def exercise_4():
    """Solution: Apply the 5-step problem-solving process."""

    # Step 1: Understand the problem
    # - Input: a sentence (string) and a target word (string)
    # - Output: integer count of occurrences
    # - Edge cases: empty string, word not found, case sensitivity,
    #   punctuation attached to words, multiple spaces

    # Step 2: Plan the solution
    # Approach: split the sentence into words, normalize case,
    # strip punctuation, then count matches.

    # Step 3: Implement (Python)
    import string

    def count_word(sentence, target):
        """Count occurrences of a word in a sentence (case-insensitive)."""
        if not sentence or not target:
            return 0

        # Normalize: lowercase and strip punctuation from each word
        target = target.lower().strip(string.punctuation)
        words = sentence.lower().split()
        cleaned_words = [w.strip(string.punctuation) for w in words]

        return cleaned_words.count(target)

    # Step 4: Test
    test_cases = [
        ("The cat sat on the mat", "the", 2),
        ("Hello hello HELLO", "hello", 3),
        ("Python is great", "java", 0),
        ("", "word", 0),
        ("One word", "", 0),
        ("Wait, wait! Don't wait.", "wait", 3),
    ]

    print("  Testing count_word:")
    all_passed = True
    for sentence, target, expected in test_cases:
        result = count_word(sentence, target)
        status = "PASS" if result == expected else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"    {status}: count_word('{sentence}', '{target}') = {result} (expected {expected})")

    # Step 5: Refine
    # The implementation above is already clean and handles edge cases.
    # A one-liner alternative (less readable but concise):
    def count_word_v2(sentence, target):
        """Refined version using generator expression."""
        if not sentence or not target:
            return 0
        t = target.lower()
        return sum(1 for w in sentence.lower().split()
                   if w.strip(string.punctuation) == t)

    print(f"\n  Refined version also works: {count_word_v2('The cat sat on the mat', 'the')}")


# === Exercise 5: Computational Thinking ===
# Problem: Apply four components of computational thinking to organize a music collection.

def exercise_5():
    """Solution: Apply computational thinking to organizing a music collection."""

    analysis = {
        "Decomposition (sub-problems)": [
            "Gather all music files from various sources/folders",
            "Extract metadata (artist, album, genre, year, track number)",
            "Handle missing or incorrect metadata",
            "Define an organizational structure (folder hierarchy)",
            "Move/rename files according to the structure",
            "Remove duplicates",
            "Create playlists based on categories",
        ],
        "Pattern Recognition": [
            "Music files follow common formats: MP3, FLAC, WAV, AAC",
            "Metadata fields are standardized (ID3 tags for MP3)",
            "Artists often have multiple albums, albums have track lists",
            "Genre classifications follow established categories",
            "Duplicate files often share same size + duration",
            "File naming often follows patterns like 'Artist - Title.mp3'",
        ],
        "Abstraction (essential vs ignorable)": [
            "Essential: artist, album, title, genre, year",
            "Useful but optional: track number, album art, lyrics",
            "Ignorable: file creation date, exact bitrate, file location",
            "Key decision: organize by Artist > Album > Track (most common)",
            "Abstract away file format differences (treat all audio the same)",
        ],
        "Algorithm (steps to organize)": [
            "1. Scan all directories recursively for audio files",
            "2. For each file, read metadata (ID3 tags or similar)",
            "3. If metadata is missing, attempt to infer from filename",
            "4. Normalize metadata (consistent capitalization, spelling)",
            "5. Check for duplicates (same artist + title + duration)",
            "6. Create folder structure: Music/{Artist}/{Album}/",
            "7. Rename files: {TrackNumber}_{Title}.{ext}",
            "8. Move files to their designated folder",
            "9. Generate summary report (total files, artists, albums)",
        ],
    }

    for category, items in analysis.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  - {item}")


# === Exercise 6: Code Communication ===
# Problem: Refactor poorly named code to be more communicative.

def exercise_6():
    """Solution: Refactor cryptic code with meaningful names and docs."""

    # Original code:
    # def p(l):
    #     r = 1
    #     for i in l:
    #         r *= i
    #     return r

    # Why this refactoring matters: The original function uses single-letter
    # names that give zero context about purpose. "p" could mean anything.
    # Someone reading this code would have to trace through the logic
    # to understand it computes a product.

    def calculate_product(numbers):
        """
        Calculate the product of all numbers in a list.

        Multiplies all elements together. Returns 1 for an empty list
        (the multiplicative identity), matching mathematical convention.

        Args:
            numbers: List of numeric values to multiply together.

        Returns:
            The product of all numbers, or 1 if the list is empty.

        Raises:
            TypeError: If any element is not numeric.
        """
        if not numbers:
            return 1

        product = 1
        for number in numbers:
            product *= number
        return product

    # Edge case handling that the original didn't consider
    test_cases = [
        ([1, 2, 3, 4, 5], 120),
        ([10], 10),
        ([], 1),           # Empty list returns identity element
        ([2, -3, 4], -24), # Negative numbers work correctly
        ([5, 0, 3], 0),    # Zero makes product zero
    ]

    print("  Testing calculate_product:")
    for numbers, expected in test_cases:
        result = calculate_product(numbers)
        status = "PASS" if result == expected else "FAIL"
        print(f"    {status}: calculate_product({numbers}) = {result}")


if __name__ == "__main__":
    print("=== Exercise 1: Decomposition ===")
    exercise_1()
    print("\n=== Exercise 2: Abstraction ===")
    exercise_2()
    print("\n=== Exercise 3: Algorithmic Thinking ===")
    exercise_3()
    print("\n=== Exercise 4: Problem Solving ===")
    exercise_4()
    print("\n=== Exercise 5: Computational Thinking ===")
    exercise_5()
    print("\n=== Exercise 6: Code Communication ===")
    exercise_6()
    print("\nAll exercises completed!")
