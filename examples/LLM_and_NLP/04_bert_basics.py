"""
04. BERT Basics - HuggingFace BERT Usage Example

BERT model loading, embeddings, classification
"""

print("=" * 60)
print("BERT Basics")
print("=" * 60)

try:
    import torch
    from transformers import BertTokenizer, BertModel, BertForSequenceClassification
    import torch.nn.functional as F

    # ============================================
    # 1. Load Tokenizer and Model
    # ============================================
    print("\n[1] Load BERT Model")
    print("-" * 40)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


    # ============================================
    # 2. Text Encoding
    # ============================================
    print("\n[2] Text Encoding")
    print("-" * 40)

    text = "Hello, how are you?"

    # Tokenization
    tokens = tokenizer.tokenize(text)
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")

    # Encoding
    encoded = tokenizer(text, return_tensors='pt')
    print(f"input_ids: {encoded['input_ids']}")
    print(f"attention_mask: {encoded['attention_mask']}")

    # Decoding
    decoded = tokenizer.decode(encoded['input_ids'][0])
    print(f"Decoded: {decoded}")


    # ============================================
    # 3. BERT Embedding Extraction
    # ============================================
    print("\n[3] BERT Embedding Extraction")
    print("-" * 40)

    model.eval()
    with torch.no_grad():
        outputs = model(**encoded)

    # Output structure
    last_hidden_state = outputs.last_hidden_state  # (batch, seq, hidden)
    pooler_output = outputs.pooler_output          # (batch, hidden) - [CLS] transformation

    print(f"last_hidden_state shape: {last_hidden_state.shape}")
    print(f"pooler_output shape: {pooler_output.shape}")

    # [CLS] token embedding
    cls_embedding = last_hidden_state[0, 0]  # First token
    print(f"[CLS] embedding shape: {cls_embedding.shape}")


    # ============================================
    # 4. Sentence Pair Encoding
    # ============================================
    print("\n[4] Sentence Pair Encoding")
    print("-" * 40)

    text_a = "How old are you?"
    text_b = "I am 25 years old."

    encoded_pair = tokenizer(text_a, text_b, return_tensors='pt')
    print(f"Sentence A: {text_a}")
    print(f"Sentence B: {text_b}")
    print(f"token_type_ids: {encoded_pair['token_type_ids']}")
    # [0, 0, ..., 0, 1, 1, ..., 1] - A is 0, B is 1


    # ============================================
    # 5. Sentence Classification
    # ============================================
    print("\n[5] Sentence Classification")
    print("-" * 40)

    # Load sentiment analysis model
    classifier = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )

    texts = [
        "I love this movie! It's amazing.",
        "This is terrible. I hate it.",
        "The weather is nice today."
    ]

    classifier.eval()
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)

        with torch.no_grad():
            outputs = classifier(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()

        label = "Positive" if pred == 1 else "Negative"
        conf = probs[0, pred].item()
        print(f"[{label}] ({conf:.2%}) {text[:40]}...")


    # ============================================
    # 6. Batch Processing
    # ============================================
    print("\n[6] Batch Processing")
    print("-" * 40)

    texts = ["Hello world", "How are you?", "I'm fine, thanks!"]

    # Batch encoding
    batch_encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors='pt'
    )

    print(f"Batch input_ids shape: {batch_encoded['input_ids'].shape}")

    # Batch inference
    model.eval()
    with torch.no_grad():
        batch_outputs = model(**batch_encoded)

    print(f"Batch output shape: {batch_outputs.last_hidden_state.shape}")


    # ============================================
    # 7. Sentence Similarity
    # ============================================
    print("\n[7] Sentence Similarity")
    print("-" * 40)

    def get_sentence_embedding(text, model, tokenizer):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        # [CLS] token or average pooling
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    sentences = [
        "I love programming",
        "Coding is my passion",
        "I enjoy eating pizza"
    ]

    embeddings = [get_sentence_embedding(s, model, tokenizer) for s in sentences]

    print("Sentence similarity:")
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
            print(f"  '{sentences[i][:20]}...' vs '{sentences[j][:20]}...': {sim.item():.4f}")


    # ============================================
    # Summary
    # ============================================
    print("\n" + "=" * 60)
    print("BERT Summary")
    print("=" * 60)

    summary = """
BERT Usage Patterns:
    # Load
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Encoding
    inputs = tokenizer(text, return_tensors='pt')

    # Embeddings
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS]

    # Classification
    classifier = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2
    )
    logits = classifier(**inputs).logits
"""
    print(summary)

except ImportError as e:
    print(f"Required packages not installed: {e}")
    print("pip install torch transformers")
