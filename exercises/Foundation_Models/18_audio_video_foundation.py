"""
Exercises for Lesson 18: Audio & Video Foundation Models
Topic: Foundation_Models

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Whisper Architecture and 30-Second Segments ===
# Problem: Explain 30-second chunk constraint and boundary handling.

def exercise_1():
    """Solution: Whisper 30-second segment analysis"""
    print("  Why 30-second chunks:")
    print("    Whisper's audio encoder uses fixed-size log-Mel spectrogram:")
    print("    80 frequency bins x 3000 time frames (each 10ms) = exactly 30s.")
    print("    Absolute sinusoidal positional embeddings cannot generalize")
    print("    to variable-length inputs without architectural changes.")
    print()

    print("  Handling shorter audio (5 seconds):")
    print("    Zero-padded to fill 30-second buffer.")
    print("    Decoder produces <endoftext> after actual speech content,")
    print("    not generating content for silent padding.")
    print()

    print("  Handling longer audio (3 minutes):")
    print("    Split into overlapping 30-second chunks.")
    print("    Each chunk transcribed independently.")
    print("    Stitched together using timestamp alignment.")
    print()

    print("  Boundary splitting problem:")
    print("    A word like 'extraordinary' split: 'extraord-' ends chunk 1,")
    print("    '-inary' starts chunk 2. First chunk may transcribe incorrectly,")
    print("    second chunk lacks beginning context.")
    print("    Mitigation: overlapping windows (25s chunks, 5s overlap)")
    print("    and use VAD to split at silences.")


# === Exercise 2: AudioLM Hierarchical Tokenization ===
# Problem: Explain why 3-level token hierarchy is necessary.

def exercise_2():
    """Solution: AudioLM hierarchical tokenization"""
    print("  Why single resolution fails:")
    print()
    print("  High-resolution only (fine acoustic, ~100/sec):")
    print("    10 seconds = 1000 tokens. Transformer must maintain semantic")
    print("    coherence (words, grammar) over 1000 tokens while tracking")
    print("    acoustic details. Long-range dependencies too hard at this scale.")
    print("    Wastes capacity learning acoustic copies rather than semantics.")
    print()

    print("  Low-resolution only (semantic, ~25/sec):")
    print("    Captures content (what is said) but discards speaker identity,")
    print("    prosody, fine acoustic details. Cannot reconstruct high-quality")
    print("    audio -- result sounds robotic.")
    print()

    print("  Why hierarchy solves this:")
    levels = [
        ("Semantic (~25/sec)", "What is said", "Long-context content planning"),
        ("Coarse Acoustic (~50/sec)", "How it sounds (speaker, prosody)", "Conditioned on semantic"),
        ("Fine Acoustic (~100/sec)", "Perceptual details (mic, timing)", "Conditioned on coarse"),
    ]
    for name, captures, role in levels:
        print(f"    {name}: {captures}. {role}")

    print()
    print("  Each level focuses on its own complexity: semantic planning")
    print("  over long context, then progressive acoustic refinement.")
    print("  Divide-and-conquer makes each sub-problem tractable.")


# === Exercise 3: Video Frame Sampling Strategy ===
# Problem: Analyze uniform sampling limitations and propose alternatives.

def exercise_3():
    """Solution: Video frame sampling strategies"""
    print("  Limitations of uniform sampling:")
    print("    1. Redundancy for static videos (talking head -> 99% identical frames)")
    print("    2. Missing key events (3s action + 27s replay -> undersampled)")
    print("    3. Ignores motion (all frames treated equally)")
    print("    4. Resolution-length mismatch (2-min and 10-min get same 8 frames)")
    print()

    print("  Alternative 1: Scene-change / keyframe-based sampling")
    print("    Algorithm:")
    print("      1. Compute inter-frame difference for all frames")
    print("      2. Score each frame by how much it differs from previous")
    print("      3. Select frames at highest scene-change scores")
    print("    Best for: action videos, movies, tutorials with distinct steps")
    print()

    # Simulate with synthetic motion data
    import random
    random.seed(42)

    # Simulated frame differences (high = scene change)
    num_frames = 100
    diffs = [random.random() * 0.1 for _ in range(num_frames)]
    # Add scene changes at specific frames
    for frame in [15, 35, 52, 78, 90]:
        diffs[frame] = random.random() * 0.5 + 0.5

    # Keyframe sampling
    num_sample = 8
    indexed_diffs = [(i, d) for i, d in enumerate(diffs)]
    indexed_diffs.sort(key=lambda x: -x[1])
    keyframes = sorted([x[0] for x in indexed_diffs[:num_sample]])

    # Uniform sampling
    uniform_frames = [int(i * (num_frames - 1) / (num_sample - 1)) for i in range(num_sample)]

    print(f"    Simulation ({num_frames} frames, scene changes at [15,35,52,78,90]):")
    print(f"      Uniform frames:   {uniform_frames}")
    print(f"      Keyframe frames:  {keyframes}")
    print(f"      Keyframes better capture actual scene transitions!")
    print()

    print("  Alternative 2: Activity-density-proportional sampling")
    print("    Algorithm:")
    print("      1. Compute optical flow/frame difference per second")
    print("      2. Compute 'activity score' per segment")
    print("      3. Allocate frames proportionally to activity")
    print("    Best for: sports, instructional videos, surveillance")


# === Exercise 4: MusicGen Token Budget Calculation ===
# Problem: Calculate tokens and maximum duration for MusicGen.

def exercise_4():
    """Solution: MusicGen token budget calculation"""
    tokens_per_second = 50  # EnCodec compression rate
    max_context = 4096

    # Part A
    duration_60 = 60
    total_tokens_60 = duration_60 * tokens_per_second
    print(f"  A) Total tokens for {duration_60} seconds:")
    print(f"     {duration_60} * {tokens_per_second} = {total_tokens_60} tokens")
    print()

    # Part B
    max_duration = max_context / tokens_per_second
    fits = total_tokens_60 <= max_context
    print(f"  B) Can 60 seconds fit in {max_context} context?")
    print(f"     {total_tokens_60} {'<' if fits else '>'} {max_context} -> {'YES' if fits else 'NO'}")
    print(f"     Maximum duration in one pass: {max_context}/{tokens_per_second} = {max_duration:.1f} seconds")
    print()

    # Part C
    target = 300  # 5 minutes
    total_tokens_300 = target * tokens_per_second
    print(f"  C) Strategy for {target} seconds ({target/60:.0f} minutes):")
    print(f"     {target} * {tokens_per_second} = {total_tokens_300:,} tokens >> {max_context}")
    print()
    print("     Continuation with overlap approach:")
    chunk_duration = 75.0
    overlap_duration = 6.0
    effective_per_chunk = chunk_duration - overlap_duration
    num_chunks = int(target / effective_per_chunk) + 1

    print(f"       Chunk duration: {chunk_duration}s (~{int(chunk_duration * tokens_per_second)} tokens)")
    print(f"       Overlap: {overlap_duration}s for musical coherence")
    print(f"       Effective per chunk: {effective_per_chunk}s")
    print(f"       Number of chunks: ~{num_chunks}")
    print()
    print("     Process:")
    print("       1. First chunk: generate from text prompt")
    print("       2. Subsequent: condition on last 6s of previous chunk")
    print("       3. Crossfade between chunks at overlap regions")
    print("     Text prompt constant across all chunks for style consistency.")


if __name__ == "__main__":
    print("=== Exercise 1: Whisper 30-Second Segments ===")
    exercise_1()
    print("\n=== Exercise 2: AudioLM Hierarchical Tokenization ===")
    exercise_2()
    print("\n=== Exercise 3: Video Frame Sampling ===")
    exercise_3()
    print("\n=== Exercise 4: MusicGen Token Budget ===")
    exercise_4()
    print("\nAll exercises completed!")
