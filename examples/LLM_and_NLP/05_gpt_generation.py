"""
05. GPT Text Generation Example

Text generation using GPT-2
"""

print("=" * 60)
print("GPT Text Generation")
print("=" * 60)

try:
    import torch
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    import torch.nn.functional as F

    # ============================================
    # 1. Load GPT-2
    # ============================================
    print("\n[1] Load GPT-2 Model")
    print("-" * 40)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


    # ============================================
    # 2. Basic Generation (Greedy)
    # ============================================
    print("\n[2] Greedy Generation")
    print("-" * 40)

    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=50,
        do_sample=False  # Greedy
    )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")


    # ============================================
    # 3. Sampling Generation
    # ============================================
    print("\n[3] Temperature Sampling")
    print("-" * 40)

    prompt = "The future of AI is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    for temp in [0.5, 1.0, 1.5]:
        output = model.generate(
            input_ids,
            max_length=40,
            do_sample=True,
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id
        )
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"temp={temp}: {generated[:60]}...")


    # ============================================
    # 4. Top-k / Top-p Sampling
    # ============================================
    print("\n[4] Top-k / Top-p Sampling")
    print("-" * 40)

    prompt = "In the year 2050"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Top-k
    output_topk = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    print(f"Top-k (k=50): {tokenizer.decode(output_topk[0], skip_special_tokens=True)[:70]}...")

    # Top-p (Nucleus)
    output_topp = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    print(f"Top-p (p=0.9): {tokenizer.decode(output_topp[0], skip_special_tokens=True)[:70]}...")


    # ============================================
    # 5. Advanced Generation Parameters
    # ============================================
    print("\n[5] Advanced Generation Parameters")
    print("-" * 40)

    prompt = "Python is a programming language"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=80,
        min_length=30,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        top_k=50,
        no_repeat_ngram_size=2,    # Prevent n-gram repetition
        repetition_penalty=1.2,     # Repetition penalty
        num_return_sequences=2,     # Generate multiple sequences
        pad_token_id=tokenizer.eos_token_id
    )

    print(f"Prompt: {prompt}")
    for i, out in enumerate(output):
        text = tokenizer.decode(out, skip_special_tokens=True)
        print(f"\nGenerated {i+1}: {text}")


    # ============================================
    # 6. Manual Generation Loop
    # ============================================
    print("\n[6] Manual Generation (Step-by-step)")
    print("-" * 40)

    def generate_manual(prompt, max_tokens=20, temperature=1.0):
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[:, -1, :]  # Last token

            # Apply temperature
            probs = F.softmax(logits / temperature, dim=-1)

            # Sampling
            next_token = torch.multinomial(probs, num_samples=1)

            # EOS check
            if next_token.item() == tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return tokenizer.decode(input_ids[0], skip_special_tokens=True)

    result = generate_manual("The robot said", max_tokens=15, temperature=0.8)
    print(f"Manual generation: {result}")


    # ============================================
    # 7. Conditional Generation (Prompt-based)
    # ============================================
    print("\n[7] Conditional Generation")
    print("-" * 40)

    prompts = [
        "Q: What is machine learning?\nA:",
        "Translate English to French: Hello, how are you? ->",
        "Summarize: Artificial intelligence is transforming various industries. ->"
    ]

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        output = model.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Input: {prompt[:50]}...")
        print(f"Output: {result[len(prompt):len(prompt)+60]}...")
        print()


    # ============================================
    # Summary
    # ============================================
    print("=" * 60)
    print("GPT Generation Summary")
    print("=" * 60)

    summary = """
Generation Strategies:
    - Greedy: do_sample=False, deterministic
    - Temperature: Lower = more deterministic, Higher = more diverse
    - Top-k: Sample from top k tokens
    - Top-p (Nucleus): Sample up to cumulative probability p

Key Code:
    output = model.generate(
        input_ids,
        max_length=50,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )
"""
    print(summary)

except ImportError as e:
    print(f"Required packages not installed: {e}")
    print("pip install torch transformers")
