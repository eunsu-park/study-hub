"""
Exercises for Lesson 01: NLP Basics
Topic: LLM_and_NLP

Solutions to practice problems from the lesson.
"""

import re
import unicodedata
import numpy as np
from collections import Counter


# === Exercise 1: Tokenization Comparison ===
# Problem: Given the sentence "unhappiness is not the opposite of happiness",
# tokenize it using three different approaches: (1) simple whitespace splitting,
# (2) BERT WordPiece tokenizer, and (3) GPT-style BPE via HuggingFace.
# Compare the resulting tokens and explain why the subword tokenizers split
# certain words differently.

def exercise_1():
    """Tokenization comparison: whitespace vs WordPiece vs BPE."""
    sentence = "unhappiness is not the opposite of happiness"

    # 1. Whitespace splitting
    whitespace_tokens = sentence.split()
    print("Whitespace:", whitespace_tokens)
    # ['unhappiness', 'is', 'not', 'the', 'opposite', 'of', 'happiness']

    # 2. Simulated BERT WordPiece (without requiring transformers download)
    # WordPiece splits "unhappiness" into ["un", "##happiness"] because
    # "un" and "happiness" are in the vocabulary but "unhappiness" may not be.
    # The ## prefix indicates a continuation of the previous word.
    bert_wordpiece_simulation = {
        "unhappiness": ["un", "##happi", "##ness"],
        "is": ["is"],
        "not": ["not"],
        "the": ["the"],
        "opposite": ["opposite"],
        "of": ["of"],
        "happiness": ["happiness"],
    }
    bert_tokens = []
    for word in whitespace_tokens:
        bert_tokens.extend(bert_wordpiece_simulation.get(word, [word]))
    print("BERT WordPiece (simulated):", bert_tokens)

    # 3. Simulated GPT-2 BPE
    # GPT-2 uses a space marker (represented as G) to indicate word beginnings.
    # Words not at sentence start keep the space prefix.
    gpt2_bpe_simulation = ["un", "happiness", "\u0120is", "\u0120not", "\u0120the",
                           "\u0120opposite", "\u0120of", "\u0120happiness"]
    print("GPT-2 BPE (simulated):", gpt2_bpe_simulation)

    # Key observations:
    print("\n--- Key Observations ---")
    print("1. Whitespace: keeps compound words intact, causing OOV problems")
    print("   if 'unhappiness' wasn't in training data.")
    print("2. BERT uses ## prefix to indicate continuation of a word")
    print("   (e.g., ##happiness follows un).")
    print("3. GPT-2 uses a special space marker to indicate word beginnings.")
    print("4. Both subword methods handle 'unhappiness' by reusing known")
    print("   subwords 'un' and 'happiness'.")


# === Exercise 2: Attention Mask Construction ===
# Problem: Given a batch of three sentences with different lengths after
# tokenization, write a function that pads them to the same length and creates
# the corresponding attention mask.

def exercise_2():
    """Attention mask construction with padding."""
    # Simulated tokenized sequences (already encoded to IDs)
    sequences = [
        [101, 7592, 2088, 102],          # length 4
        [101, 1045, 2293, 3019, 102],    # length 5
        [101, 4937, 102],                 # length 3
    ]

    pad_token_id = 0

    # Find max length
    max_len = max(len(seq) for seq in sequences)

    # Pad sequences and create attention masks
    padded = []
    attention_masks = []

    for seq in sequences:
        pad_length = max_len - len(seq)
        padded_seq = seq + [pad_token_id] * pad_length
        mask = [1] * len(seq) + [0] * pad_length

        padded.append(padded_seq)
        attention_masks.append(mask)

    print("Padded input_ids:")
    for i, p in enumerate(padded):
        print(f"  Sequence {i}: {p}")

    print("\nAttention masks:")
    for i, m in enumerate(attention_masks):
        print(f"  Sequence {i}: {m}")

    # Verify correctness
    for i in range(len(sequences)):
        for j in range(max_len):
            if j < len(sequences[i]):
                assert attention_masks[i][j] == 1, "Real token should have mask=1"
                assert padded[i][j] == sequences[i][j], "Real token should be unchanged"
            else:
                assert attention_masks[i][j] == 0, "Padding should have mask=0"
                assert padded[i][j] == pad_token_id, "Padding should be pad_token_id"

    print("\nAll assertions passed! Attention mask correctly marks real tokens as 1")
    print("and padding tokens as 0.")


# === Exercise 3: Preprocessing Pipeline Design ===
# Problem: Design a complete text preprocessing pipeline for a sentiment analysis
# task on social media data (tweets). The pipeline should handle: URLs, hashtags,
# mentions, emojis, and repeated characters.

def exercise_3():
    """Tweet preprocessing pipeline for sentiment analysis."""

    def preprocess_tweet(text):
        """
        Preprocessing pipeline for social media text (tweets).
        Each step addresses a specific noise source in tweet data.
        """
        # Step 1: Unicode normalization - handles accented characters consistently
        text = unicodedata.normalize('NFC', text)

        # Step 2: Remove URLs - URLs carry little semantic value for sentiment
        text = re.sub(r'http\S+|www\S+', '', text)

        # Step 3: Replace mentions with a placeholder - preserves social signal
        text = re.sub(r'@\w+', '@user', text)

        # Step 4: Extract hashtag content (remove # but keep the word)
        text = re.sub(r'#(\w+)', r'\1', text)

        # Step 5: Remove emojis - alternatively, convert to text description
        text = text.encode('ascii', 'ignore').decode('ascii')

        # Step 6: Reduce repeated characters - "soooo good" -> "soo good"
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)

        # Step 7: Lowercase and strip
        text = text.lower().strip()

        # Step 8: Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        return text

    # Test cases
    tweets = [
        "OMG this is soooo amazing!! Check out https://example.com #NLP @anthropic",
        "Terrible service @company!!! Never again http://link.co #angry #bad",
        "Just okaaay... not great, not terrible @user1 @user2",
        "BEST. DAY. EVERRRR!!! #blessed #happy https://pic.me/abc",
    ]

    print("Tweet Preprocessing Pipeline Results:")
    print("=" * 60)
    for tweet in tweets:
        processed = preprocess_tweet(tweet)
        print(f"Original:  {tweet}")
        print(f"Processed: {processed}")
        print("-" * 60)

    # Design rationale
    print("\nDesign Rationale:")
    print("- URLs removed: they add noise without conveying sentiment.")
    print("- Mentions normalized: preserves social interaction signal")
    print("  without overfitting to specific usernames.")
    print("- Repeated characters reduced (not removed): 'sooo' likely")
    print("  means 'very', keeping 2 characters signals emphasis.")
    print("- Lowercase applied after URL removal to avoid breaking")
    print("  URL matching patterns.")


# === Exercise 4: Vocabulary Coverage Analysis ===
# Problem: Build a vocabulary from a training corpus and analyze OOV rates
# on a test set for different vocabulary sizes.

def exercise_4():
    """Vocabulary coverage analysis across different vocabulary sizes."""

    # Simulated corpus data
    train_texts = [
        "the cat sat on the mat the dog chased the cat",
        "machine learning is a subset of artificial intelligence",
        "natural language processing uses deep learning models",
        "the quick brown fox jumps over the lazy dog repeatedly",
        "deep learning neural networks transform language understanding",
        "cats and dogs are popular pets in many countries worldwide",
        "artificial neural networks learn patterns from training data",
        "the model processes text using attention mechanisms efficiently",
        "transformers revolutionized natural language processing tasks completely",
        "word embeddings represent semantic meaning in vector space",
    ]

    test_texts = [
        "the neural network learns language patterns",
        "convolutional networks process image data efficiently",
        "attention mechanisms improve translation quality significantly",
    ]

    def tokenizer(text):
        return text.lower().split()

    def analyze_vocabulary_coverage(train_texts, test_texts, tokenizer, vocab_sizes):
        """Analyze OOV rate for different vocabulary sizes."""
        # Count all word frequencies in training set
        train_counter = Counter()
        for text in train_texts:
            train_counter.update(tokenizer(text))

        # Count all tokens in test set
        test_tokens = []
        for text in test_texts:
            test_tokens.extend(tokenizer(text))
        total_test_tokens = len(test_tokens)

        print(f"Training vocabulary size: {len(train_counter)} unique words")
        print(f"Test set tokens: {total_test_tokens}")
        print(f"\n{'Vocab Size':>12}  {'OOV Rate':>10}  {'Coverage':>10}")
        print("-" * 38)

        results = {}
        for vocab_size in vocab_sizes:
            # Build vocabulary with top-k words
            top_words = set(w for w, _ in train_counter.most_common(vocab_size))

            # Count OOV tokens in test set
            oov_count = sum(1 for t in test_tokens if t not in top_words)
            oov_rate = oov_count / total_test_tokens * 100

            results[vocab_size] = {
                'oov_rate': oov_rate,
                'coverage': 100 - oov_rate,
                'oov_count': oov_count,
            }
            print(f"{vocab_size:>12d}  {oov_rate:>9.2f}%  {100 - oov_rate:>9.2f}%")

        return results

    vocab_sizes = [5, 10, 20, 30, 50]
    results = analyze_vocabulary_coverage(train_texts, test_texts, tokenizer, vocab_sizes)

    print("\nTrade-off Analysis:")
    print("- Larger vocabulary = lower OOV rate but larger embedding matrix")
    print("  (memory cost = vocab_size x embed_dim).")
    print("- Subword tokenizers (BPE, WordPiece) achieve near-0% OOV with")
    print("  compact vocabularies (~30k-50k tokens).")
    print("- For word-level models, 30k-50k vocabulary is a common practical")
    print("  choice balancing coverage and memory.")


# === Exercise 5: Tokenizer Special Token Roles ===
# Problem: Explain the purpose of the special tokens [CLS], [SEP], [PAD],
# [MASK], and [UNK] in BERT's tokenizer.

def exercise_5():
    """BERT special token roles and their purposes."""

    special_tokens = {
        "[CLS]": {
            "id": 101,
            "role": "Classification token, prepended to every input",
            "usage": "Its final hidden state is used as the sequence representation "
                     "for classification tasks.",
            "when": "Beginning of every input",
        },
        "[SEP]": {
            "id": 102,
            "role": "Separator token, appended at end of each segment",
            "usage": "Separates sentence A and sentence B in tasks like NLI or QA.",
            "when": "End of each sentence segment",
        },
        "[PAD]": {
            "id": 0,
            "role": "Padding token, fills shorter sequences to match batch length",
            "usage": "Always attended to with mask=0 so it doesn't affect computations.",
            "when": "Batch padding",
        },
        "[MASK]": {
            "id": 103,
            "role": "Masking token, replaces 15% of tokens during MLM pre-training",
            "usage": "The model must predict the original token from context.",
            "when": "15% of tokens during training",
        },
        "[UNK]": {
            "id": 100,
            "role": "Unknown token, replaces any word that cannot be tokenized",
            "usage": "With WordPiece this is rarely needed since most words can be "
                     "decomposed into subwords.",
            "when": "Rare; subwords handle most cases",
        },
    }

    print("BERT Special Tokens:")
    print("=" * 70)
    for token, info in special_tokens.items():
        print(f"\n{token} (ID: {info['id']})")
        print(f"  Role:  {info['role']}")
        print(f"  Usage: {info['usage']}")
        print(f"  When:  {info['when']}")

    # Demonstrate special token usage in a sentence
    print("\n\nExample: Encoding 'Hello [MASK] world'")
    print("  Tokens: ['[CLS]', 'hello', '[MASK]', 'world', '[SEP]']")
    print("  IDs:    [101, 7592, 103, 2088, 102]")

    print("\n  Summary Table:")
    print(f"  {'Token':<10} {'Role':<40} {'When Used'}")
    print(f"  {'-'*10} {'-'*40} {'-'*30}")
    for token, info in special_tokens.items():
        print(f"  {token:<10} {info['role'][:40]:<40} {info['when']}")


if __name__ == "__main__":
    print("=== Exercise 1: Tokenization Comparison ===")
    exercise_1()
    print("\n=== Exercise 2: Attention Mask Construction ===")
    exercise_2()
    print("\n=== Exercise 3: Preprocessing Pipeline Design ===")
    exercise_3()
    print("\n=== Exercise 4: Vocabulary Coverage Analysis ===")
    exercise_4()
    print("\n=== Exercise 5: Tokenizer Special Token Roles ===")
    exercise_5()
    print("\nAll exercises completed!")
