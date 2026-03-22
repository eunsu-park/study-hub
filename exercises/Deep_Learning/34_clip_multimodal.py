"""
Exercises for Lesson 34: CLIP and Multimodal Learning
Topic: Deep_Learning

Solutions to practice problems from the lesson.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# === Exercise 1: Understand the InfoNCE Loss ===
# Problem: Trace through CLIP loss with N=4 image-text pairs.

def exercise_1():
    """Manual trace of InfoNCE/CLIP loss with N=4 pairs."""
    torch.manual_seed(42)

    N = 4
    d = 8
    temperature = 0.07

    # Simulated normalized embeddings
    image_emb = F.normalize(torch.randn(N, d), dim=-1)
    text_emb = F.normalize(torch.randn(N, d), dim=-1)

    # Similarity matrix
    sim = (image_emb @ text_emb.T) / temperature
    print(f"  Similarity matrix ({N}x{N}) / temperature:")
    for i in range(N):
        row = " ".join(f"{sim[i,j].item():7.3f}" for j in range(N))
        print(f"    [{row}]")

    # Positive pairs are on the diagonal (i-th image matches i-th text)
    labels = torch.arange(N)

    # Image-to-text loss
    loss_i2t = F.cross_entropy(sim, labels)
    # Text-to-image loss
    loss_t2i = F.cross_entropy(sim.T, labels)
    # Combined
    total_loss = (loss_i2t + loss_t2i) / 2

    diag_entries = [f"{sim[i,i].item():.3f}" for i in range(N)]
    print(f"\n  Positive pairs: diagonal entries {diag_entries}")
    print(f"  loss_i2t = {loss_i2t.item():.4f}")
    print(f"  loss_t2i = {loss_t2i.item():.4f}")
    print(f"  total CLIP loss = {total_loss.item():.4f}")

    # Temperature effect
    print(f"\n  Temperature effect:")
    for temp in [0.01, 0.07, 1.0, 10.0]:
        sim_t = (image_emb @ text_emb.T) / temp
        softmax_row = F.softmax(sim_t[0], dim=0)
        entropy = -(softmax_row * softmax_row.log()).sum().item()
        print(f"    temp={temp:5.2f}: max_prob={softmax_row.max():.4f}, entropy={entropy:.4f}")

    print("  Small temp -> sharp (nearly one-hot); large temp -> uniform.")
    print("  Both directions needed: i2t ensures images find correct text,")
    print("  t2i ensures texts find correct image (asymmetric matches possible).")


# === Exercise 2: Zero-shot Classification ===
# Problem: Implement zero-shot classification using cosine similarity.

def exercise_2():
    """Zero-shot classification with synthetic CLIP-like model."""
    torch.manual_seed(42)

    class SimpleCLIP(nn.Module):
        def __init__(self, img_dim=100, text_dim=50, embed_dim=64):
            super().__init__()
            self.image_encoder = nn.Sequential(
                nn.Linear(img_dim, 128), nn.ReLU(), nn.Linear(128, embed_dim)
            )
            self.text_encoder = nn.Sequential(
                nn.Linear(text_dim, 128), nn.ReLU(), nn.Linear(128, embed_dim)
            )

        def encode_image(self, x):
            return F.normalize(self.image_encoder(x), dim=-1)

        def encode_text(self, x):
            return F.normalize(self.text_encoder(x), dim=-1)

    model = SimpleCLIP()

    # Simulate training (brief)
    n_train = 500
    img_data = torch.randn(n_train, 100)
    txt_data = torch.randn(n_train, 50)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        img_emb = model.encode_image(img_data)
        txt_emb = model.encode_text(txt_data)
        sim = img_emb @ txt_emb.T / 0.07
        labels = torch.arange(n_train)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Zero-shot classification
    class_names = ["cat", "dog", "car", "tree", "bird",
                   "fish", "plane", "boat", "house", "flower"]
    n_classes = len(class_names)

    # Simulate class text embeddings (one per class)
    class_texts = torch.randn(n_classes, 50)
    test_images = torch.randn(20, 100)

    model.eval()
    with torch.no_grad():
        img_emb = model.encode_image(test_images)
        txt_emb = model.encode_text(class_texts)
        similarities = img_emb @ txt_emb.T  # (20, 10)

    # Classify each image
    predictions = similarities.argmax(dim=1)
    print(f"  Zero-shot classification (top-3 per image):")
    for i in range(5):
        top3 = similarities[i].topk(3)
        classes = [class_names[idx] for idx in top3.indices.tolist()]
        scores = [f"{s:.3f}" for s in top3.values.tolist()]
        print(f"    Image {i}: {list(zip(classes, scores))}")

    # Ensemble prompting
    print(f"\n  Prompt ensemble improves confidence by averaging over templates,")
    print(f"  reducing sensitivity to specific wording choices.")


# === Exercise 3: Image Retrieval System ===
# Problem: Text-to-image retrieval using cosine similarity.

def exercise_3():
    """Text-to-image retrieval with cosine similarity."""
    torch.manual_seed(42)

    embed_dim = 32

    # Simulate 50 image embeddings from 5 categories (10 per category)
    categories = ["sports_car", "mountain", "cat", "building", "food"]
    n_per_cat = 10

    # Each category has a distinct embedding cluster
    image_embeddings = []
    image_labels = []
    for i, cat in enumerate(categories):
        center = torch.randn(embed_dim)
        embs = F.normalize(center + torch.randn(n_per_cat, embed_dim) * 0.3, dim=-1)
        image_embeddings.append(embs)
        image_labels.extend([cat] * n_per_cat)

    image_embeddings = torch.cat(image_embeddings)  # (50, 32)

    # Simulate text query embedding
    # Make query close to "sports_car" cluster
    query_emb = F.normalize(image_embeddings[:10].mean(dim=0, keepdim=True), dim=-1)

    # Retrieve top-5
    similarities = (query_emb @ image_embeddings.T).squeeze()
    top5_idx = similarities.topk(5).indices.tolist()
    top5_scores = similarities.topk(5).values.tolist()

    print(f"  Query: 'a red sports car'")
    print(f"  Top-5 retrieved images:")
    correct = 0
    for rank, (idx, score) in enumerate(zip(top5_idx, top5_scores)):
        label = image_labels[idx]
        is_correct = label == "sports_car"
        correct += is_correct
        print(f"    Rank {rank+1}: {label} (sim={score:.4f}) {'[correct]' if is_correct else ''}")

    precision = correct / 5
    print(f"  Precision@5: {precision:.2f}")


# === Exercise 4: CLIP Linear Probe ===
# Problem: Use frozen CLIP features with a linear classifier.

def exercise_4():
    """Linear probe on top of frozen CLIP-like features."""
    torch.manual_seed(42)

    embed_dim = 64
    n_classes = 10

    # Simulate "pretrained" image encoder
    image_encoder = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, embed_dim)
    )

    # Train encoder briefly on contrastive objective
    X_train = torch.randn(2000, 784)
    y_train = torch.randint(0, n_classes, (2000,))
    X_test = torch.randn(500, 784)
    y_test = torch.randint(0, n_classes, (500,))

    # Extract frozen features
    image_encoder.eval()
    with torch.no_grad():
        train_features = F.normalize(image_encoder(X_train), dim=-1)
        test_features = F.normalize(image_encoder(X_test), dim=-1)

    # Linear probe
    linear_probe = nn.Linear(embed_dim, n_classes)
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=0.01)
    loader = DataLoader(TensorDataset(train_features, y_train), batch_size=64, shuffle=True)

    for epoch in range(30):
        for feat, label in loader:
            loss = F.cross_entropy(linear_probe(feat), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        probe_acc = (linear_probe(test_features).argmax(1) == y_test).float().mean().item()
    print(f"  Linear probe accuracy: {probe_acc:.4f}")

    # Full fine-tuning
    torch.manual_seed(42)
    ft_encoder = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, embed_dim)
    )
    ft_head = nn.Linear(embed_dim, n_classes)
    ft_params = list(ft_encoder.parameters()) + list(ft_head.parameters())
    optimizer_ft = torch.optim.Adam(ft_params, lr=1e-5)
    ft_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    for epoch in range(30):
        for xb, yb in ft_loader:
            feat = F.normalize(ft_encoder(xb), dim=-1)
            loss = F.cross_entropy(ft_head(feat), yb)
            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()

    with torch.no_grad():
        ft_features = F.normalize(ft_encoder(X_test), dim=-1)
        ft_acc = (ft_head(ft_features).argmax(1) == y_test).float().mean().item()
    print(f"  Full fine-tuning accuracy: {ft_acc:.4f}")

    # CNN from scratch
    torch.manual_seed(42)
    cnn = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, n_classes),
    )
    optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=0.001)
    cnn_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    for epoch in range(30):
        for xb, yb in cnn_loader:
            loss = F.cross_entropy(cnn(xb), yb)
            optimizer_cnn.zero_grad()
            loss.backward()
            optimizer_cnn.step()

    with torch.no_grad():
        cnn_acc = (cnn(X_test).argmax(1) == y_test).float().mean().item()
    print(f"  CNN from scratch accuracy: {cnn_acc:.4f}")

    print("\n  Risk of full fine-tuning: catastrophic forgetting of pretrained features,")
    print("  especially with small downstream datasets.")


if __name__ == "__main__":
    print("=== Exercise 1: InfoNCE Loss ===")
    exercise_1()
    print("\n=== Exercise 2: Zero-shot Classification ===")
    exercise_2()
    print("\n=== Exercise 3: Image Retrieval ===")
    exercise_3()
    print("\n=== Exercise 4: CLIP Linear Probe ===")
    exercise_4()
    print("\nAll exercises completed!")
