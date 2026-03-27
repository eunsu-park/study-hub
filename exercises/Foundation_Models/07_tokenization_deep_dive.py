"""
Exercises for Lesson 07: Tokenization Deep Dive
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""

from collections import Counter


# === Exercise 1: BPE Algorithm Trace ===
# Problem: Trace 3 BPE merges on a toy corpus.

def get_pair_counts(corpus: list) -> Counter:
    """Count adjacent symbol pairs across all words."""
    counts = Counter()
    for word, freq in corpus:
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            counts[pair] += freq
    return counts


def merge_pair(corpus: list, pair: tuple) -> list:
    """Merge a symbol pair in all words of the corpus."""
    merged = pair[0] + pair[1]
    new_corpus = []
    for word, freq in corpus:
        symbols = word.split()
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                new_symbols.append(merged)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_corpus.append((" ".join(new_symbols), freq))
    return new_corpus


def exercise_1():
    """Solution: BPE algorithm trace"""
    # Initial character-level corpus
    corpus = [
        ("l o w </w>", 5),
        ("l o w e r </w>", 2),
        ("n e w e s t </w>", 6),
        ("w i d e s t </w>", 3),
    ]

    vocab = set()
    for word, _ in corpus:
        for sym in word.split():
            vocab.add(sym)

    print("  Initial corpus:")
    for word, freq in corpus:
        print(f"    {word}  (x{freq})")
    print(f"  Initial vocab: {sorted(vocab)}")
    print()

    for merge_num in range(1, 4):
        pair_counts = get_pair_counts(corpus)
        best_pair = pair_counts.most_common(1)[0]
        pair, count = best_pair

        print(f"  Merge {merge_num}: '{pair[0]}' + '{pair[1]}' -> '{pair[0]+pair[1]}' (count {count})")
        corpus = merge_pair(corpus, pair)
        vocab.add(pair[0] + pair[1])

        for word, freq in corpus:
            print(f"    {word}  (x{freq})")
        print()

    print(f"  Final vocab: {sorted(vocab)}")


# === Exercise 2: Tokenization Fertility Analysis ===
# Problem: Calculate fertility for different text types.

def exercise_2():
    """Solution: Tokenization fertility analysis"""
    # Simulated tokenizations
    texts = [
        {
            "text": "Hello",
            "tokens": ["Hello"],
            "chars": 5,
        },
        {
            "text": "Anthropic",
            "tokens": ["Anthrop", "ic"],
            "chars": 9,
        },
        {
            "text": "나는 학교에 갑니다",
            "tokens": ["나", "는", " ", "학", "교", "에", " ", "갑", "니", "다"],
            "chars": 10,
        },
        {
            "text": "x = y ** 2 + z",
            "tokens": ["x", " =", " y", " **", " ", "2", " +", " z"],
            "chars": 15,
        },
    ]

    print(f"  {'Text':<25} {'Chars':<8} {'Tokens':<8} {'Fertility':<10}")
    print("  " + "-" * 55)

    for t in texts:
        fertility = len(t["tokens"]) / t["chars"]
        print(f"  {t['text']:<25} {t['chars']:<8} {len(t['tokens']):<8} {fertility:<10.2f}")

    print()
    print("  Korean text has the highest fertility (1.0) -- every character")
    print("  becomes a separate token, indicating the tokenizer is severely")
    print("  disadvantaged for Korean.")
    print()
    print("  Practical consequences of high fertility:")
    print("  - Sequence length explosion (Korean uses context window faster)")
    print("  - Cost disadvantage (APIs charge per token)")
    print("  - Performance disadvantage (harder to learn language patterns)")
    print("  - Context window waste (fewer Korean words fit in context)")


# === Exercise 3: WordPiece vs BPE Selection Criterion ===
# Problem: Compare BPE and WordPiece selection criteria with calculation.

def exercise_3():
    """Solution: WordPiece vs BPE selection criterion"""
    print("  1. BPE selection criterion:")
    print("     Selects the pair with the HIGHEST raw co-occurrence frequency.")
    print("     Purely frequency-based: merge the most common adjacent pair.")
    print()

    print("  2. WordPiece selection criterion:")
    print("     Selects the pair that maximizes language model likelihood increase:")
    print("     score(A, B) = freq(AB) / (freq(A) * freq(B))")
    print("     Essentially pointwise mutual information (PMI).")
    print()

    # Calculation
    freq_un = 100
    freq_u = 500
    freq_n = 400
    score_un = freq_un / (freq_u * freq_n)

    freq_ion = 50
    freq_i_prefix = 150
    freq_n_suffix = 80
    score_ion = freq_ion / (freq_i_prefix * freq_n_suffix)

    print("  3. WordPiece calculation:")
    print(f"     Pair A = 'un' (freq={freq_un}):")
    print(f"       freq('u')={freq_u}, freq('n')={freq_n}")
    print(f"       score = {freq_un} / ({freq_u} * {freq_n}) = {score_un:.6f}")
    print()
    print(f"     Pair B = '##ion' (freq={freq_ion}):")
    print(f"       freq('##i')={freq_i_prefix}, freq('##n')={freq_n_suffix}")
    print(f"       score = {freq_ion} / ({freq_i_prefix} * {freq_n_suffix}) = {score_ion:.5f}")
    print()
    print(f"     WordPiece prefers '##ion' (score {score_ion:.5f} >> {score_un:.6f})")
    print(f"     even though 'un' has {freq_un/freq_ion:.0f}x higher raw frequency.")
    print(f"     WordPiece captures that '##ion' parts co-occur in a very specific")
    print(f"     pattern (suffix), while 'u' and 'n' co-occur for many unrelated reasons.")


# === Exercise 4: Tokenization Failure Cases ===
# Problem: Identify tokenization problems and explain why they occur.

def exercise_4():
    """Solution: Tokenization failure cases"""
    cases = [
        {
            "case": "Arithmetic: 12345 + 67890 works but complex fails",
            "problem": (
                "BPE tokenizes large numbers inconsistently: '12345' may become "
                "['12', '34', '5'] or ['123', '45'] -- varying splits that don't "
                "align with digit positional values. Small numbers like '12' and '67' "
                "are likely single tokens."
            ),
            "fix": (
                "Digit-splitting: treat each digit as a separate token "
                "for code/math models."
            ),
        },
        {
            "case": "Prompting: 'Translate: Hello' works but 'Translate:Hello' fails",
            "problem": (
                "BPE treats word boundaries as significant. Without the space, "
                "'Hello' may be tokenized differently (merged with the colon or "
                "split differently). The model was trained on natural spacing, "
                "so unusual spacing creates out-of-distribution tokenizations."
            ),
            "fix": "Normalize whitespace in preprocessing, or train with varied spacing.",
        },
        {
            "case": "Multilingual mixing: EN+KO performs worse than each individually",
            "problem": (
                "Tokens at language switch boundaries create tokenizations never "
                "seen in training. BPE vocabularies are English-dominant. "
                "Mid-sentence language switching causes unusual byte-level fallbacks "
                "or single-character decompositions at the switching point."
            ),
            "fix": (
                "Use multilingual-aware tokenizers (SentencePiece with shared vocab), "
                "or add language boundary tokens."
            ),
        },
    ]

    for i, c in enumerate(cases, 1):
        print(f"  Case {i}: {c['case']}")
        print(f"    Problem: {c['problem']}")
        print(f"    Fix: {c['fix']}")
        print()


if __name__ == "__main__":
    print("=== Exercise 1: BPE Algorithm Trace ===")
    exercise_1()
    print("\n=== Exercise 2: Tokenization Fertility Analysis ===")
    exercise_2()
    print("\n=== Exercise 3: WordPiece vs BPE Selection Criterion ===")
    exercise_3()
    print("\n=== Exercise 4: Tokenization Failure Cases ===")
    exercise_4()
    print("\nAll exercises completed!")
