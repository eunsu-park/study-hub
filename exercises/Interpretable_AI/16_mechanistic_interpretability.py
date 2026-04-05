"""
Exercises for Lesson 16: Mechanistic Interpretability
Topic: Interpretable_AI

Solutions to practice problems from the lesson.
"""

import numpy as np


# === Exercise 1: Computing Superposition Capacity ===
# Problem: Analyze how a neural network layer can represent more features
# than it has dimensions through superposition. Compute the capacity
# (number of features representable) and interference costs.

def exercise_1():
    """Compute superposition capacity and feature interference in a toy model."""
    np.random.seed(42)

    def compute_superposition_stats(d_model, n_features, sparsity):
        """Analyze superposition in a d_model-dimensional space with n_features.

        Features are sparse (active with probability 1-sparsity).
        When n_features > d_model, superposition must occur.
        """
        # Generate random feature directions (each feature is a d_model-dim vector)
        W = np.random.randn(d_model, n_features)
        # Normalize each feature direction to unit norm
        norms = np.linalg.norm(W, axis=0, keepdims=True)
        W = W / (norms + 1e-10)

        # Compute interference matrix: I[i,j] = |W[:,i] . W[:,j]|
        interference = np.abs(W.T @ W)
        np.fill_diagonal(interference, 0)  # Self-interference is not relevant

        # Average interference per feature
        mean_interference = interference.mean(axis=1)

        # Maximum interference (worst case)
        max_interference = interference.max(axis=1)

        # Effective capacity: features whose information can be recovered
        # A feature is "recoverable" if its self-projection dominates interference
        # Approximate: feature i recoverable if max_interference[i] < threshold
        threshold = 0.5
        recoverable = (max_interference < threshold).sum()

        # Expected interference adjusted for sparsity
        # When features are sparse, interference is proportional to
        # (1-sparsity)^2 because both features must be active simultaneously
        expected_interference = mean_interference * (1 - sparsity) ** 2

        return {
            "n_features": n_features,
            "d_model": d_model,
            "ratio": n_features / d_model,
            "mean_interference": mean_interference.mean(),
            "max_interference": max_interference.mean(),
            "recoverable_features": recoverable,
            "expected_interference_with_sparsity": expected_interference.mean(),
        }

    print("  Superposition Capacity Analysis")
    print("  " + "=" * 65)

    # Vary the feature-to-dimension ratio
    d_model = 32
    print(f"\n  Model dimension: {d_model}")
    print(f"\n  {'n_features':>12} {'ratio':>8} {'mean_interf':>13} "
          f"{'max_interf':>12} {'recoverable':>13}")

    for n_features in [16, 32, 64, 128, 256, 512]:
        stats = compute_superposition_stats(d_model, n_features, sparsity=0.9)
        print(f"  {stats['n_features']:>12} {stats['ratio']:>8.1f} "
              f"{stats['mean_interference']:>13.4f} "
              f"{stats['max_interference']:>12.4f} "
              f"{stats['recoverable_features']:>13}")

    # Effect of sparsity on superposition
    print(f"\n  Effect of sparsity (n_features=128, d_model=32):")
    print(f"  {'sparsity':>10} {'raw_interf':>12} {'effective_interf':>17} {'usable':>8}")

    for sparsity in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]:
        stats = compute_superposition_stats(d_model, 128, sparsity)
        print(f"  {sparsity:>10.2f} {stats['mean_interference']:>12.4f} "
              f"{stats['expected_interference_with_sparsity']:>17.6f} "
              f"{'Yes' if stats['expected_interference_with_sparsity'] < 0.01 else 'Marginal' if stats['expected_interference_with_sparsity'] < 0.05 else 'No':>8}")

    print(f"\n  Key insight: High sparsity enables superposition.")
    print(f"  Networks can represent far more features than dimensions")
    print(f"  when features are rarely active simultaneously.")


# === Exercise 2: Training a Tiny Sparse Autoencoder ===
# Problem: Train a sparse autoencoder (SAE) to extract interpretable features
# from activations of a simple neural network, demonstrating how SAEs can
# decompose superposed representations.

def exercise_2():
    """Train a tiny sparse autoencoder on synthetic superposed activations."""
    np.random.seed(42)

    # Generate synthetic "activations" that contain superposed features
    d_model = 8        # Activation dimension
    n_true_features = 20  # True number of underlying features
    n_samples = 2000
    sparsity = 0.9      # Each feature active with prob 0.1

    # True feature directions (what we want the SAE to recover)
    true_W = np.random.randn(d_model, n_true_features)
    true_W /= np.linalg.norm(true_W, axis=0, keepdims=True)

    # Generate sparse feature activations
    feature_active = (np.random.rand(n_samples, n_true_features) > sparsity).astype(float)
    feature_magnitudes = np.random.exponential(1.0, (n_samples, n_true_features))
    feature_values = feature_active * feature_magnitudes

    # Generate superposed activations
    activations = feature_values @ true_W.T  # (n_samples, d_model)
    activations += np.random.normal(0, 0.05, activations.shape)  # Small noise

    print(f"  Sparse Autoencoder Training")
    print(f"  Activation dim: {d_model}, True features: {n_true_features}")
    print(f"  Training samples: {n_samples}, Sparsity: {sparsity}")

    # SAE architecture: encoder maps d_model -> d_sae, decoder maps d_sae -> d_model
    d_sae = 32  # Overcomplete (larger than d_model)

    def relu(x):
        return np.maximum(0, x)

    # Initialize SAE weights
    W_enc = np.random.randn(d_model, d_sae) * 0.1
    b_enc = np.zeros(d_sae)
    W_dec = np.random.randn(d_sae, d_model) * 0.1
    b_dec = np.zeros(d_model)

    # Normalize decoder columns
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-10

    lr = 0.005
    l1_coeff = 0.05  # Sparsity penalty
    batch_size = 64

    losses = []
    sparsity_history = []

    for epoch in range(100):
        epoch_loss = 0.0
        epoch_sparsity = 0.0
        n_batches = 0

        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples - batch_size, batch_size):
            batch_idx = indices[start:start + batch_size]
            x = activations[batch_idx]

            # Forward pass
            z = relu(x @ W_enc + b_enc)  # Encoded (sparse)
            x_hat = z @ W_dec + b_dec     # Reconstructed

            # Loss: reconstruction + L1 sparsity
            recon_loss = np.mean((x - x_hat) ** 2)
            l1_loss = l1_coeff * np.mean(np.abs(z))
            total_loss = recon_loss + l1_loss

            # Backward pass (manual gradients)
            d_x_hat = 2 * (x_hat - x) / batch_size
            d_W_dec = z.T @ d_x_hat / batch_size
            d_b_dec = d_x_hat.mean(axis=0)
            d_z = d_x_hat @ W_dec.T + l1_coeff * np.sign(z) / batch_size
            d_z *= (z > 0).astype(float)  # ReLU gradient
            d_W_enc = x.T @ d_z / batch_size
            d_b_enc = d_z.mean(axis=0)

            # Update
            W_enc -= lr * d_W_enc
            b_enc -= lr * d_b_enc
            W_dec -= lr * d_W_dec
            b_dec -= lr * d_b_dec

            # Normalize decoder rows
            W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-10

            epoch_loss += total_loss
            epoch_sparsity += np.mean(z > 0)
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_sparsity = epoch_sparsity / n_batches
        losses.append(avg_loss)
        sparsity_history.append(avg_sparsity)

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1:3d}: loss={avg_loss:.4f}, "
                  f"active_fraction={avg_sparsity:.4f}")

    # Analyze learned features
    print(f"\n  SAE Analysis:")

    # Encode all activations
    z_all = relu(activations @ W_enc + b_enc)
    active_neurons = np.mean(z_all > 0, axis=0)

    # Find dead neurons (never active)
    dead_count = np.sum(active_neurons < 0.01)
    alive_count = d_sae - dead_count
    print(f"    SAE dictionary size: {d_sae}")
    print(f"    Alive neurons: {alive_count}")
    print(f"    Dead neurons: {dead_count}")
    print(f"    Mean activation frequency: {np.mean(active_neurons):.4f}")

    # Check if SAE features align with true features
    # Compute cosine similarity between each SAE decoder row and true feature direction
    similarities = np.abs(W_dec @ true_W)  # (d_sae, n_true_features)
    best_match = similarities.max(axis=1)

    matched_features = np.sum(best_match > 0.7)
    print(f"    SAE features matching true features (cos > 0.7): {matched_features}/{d_sae}")

    # Check how many true features are recovered
    best_recovery = similarities.max(axis=0)
    recovered = np.sum(best_recovery > 0.7)
    print(f"    True features recovered: {recovered}/{n_true_features}")

    # Reconstruction quality
    x_hat_all = z_all @ W_dec + b_dec
    recon_error = np.mean((activations - x_hat_all) ** 2)
    print(f"    Final reconstruction MSE: {recon_error:.6f}")

    print(f"\n  Sparse autoencoders decompose superposed activations")
    print(f"  into interpretable, monosemantic feature directions.")


# === Exercise 3: Activation Patching on a 2-Layer Model ===
# Problem: Implement activation patching (causal intervention) on a simple
# 2-layer network to identify which components are causally responsible
# for specific behaviors.

def exercise_3():
    """Implement activation patching on a 2-layer neural network."""
    np.random.seed(42)

    # Build a simple 2-layer network trained on a binary task
    # Task: output 1 if (feature_0 > 0 AND feature_1 > 0) else 0
    # This creates a model where specific neurons implement the AND logic

    d_input = 4
    d_hidden = 8
    d_output = 1

    def relu(x):
        return np.maximum(0, x)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    # Initialize and train a simple network
    W1 = np.random.randn(d_input, d_hidden) * 0.5
    b1 = np.zeros(d_hidden)
    W2 = np.random.randn(d_hidden, d_output) * 0.5
    b2 = np.zeros(d_output)

    # Training data
    n = 2000
    X = np.random.randn(n, d_input)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(float).reshape(-1, 1)

    # Train
    lr = 0.01
    for epoch in range(200):
        h = relu(X @ W1 + b1)
        out = sigmoid(h @ W2 + b2)
        loss = -np.mean(y * np.log(out + 1e-10) + (1 - y) * np.log(1 - out + 1e-10))

        # Backward
        d_out = (out - y) / n
        d_W2 = h.T @ d_out
        d_b2 = d_out.sum(axis=0)
        d_h = d_out @ W2.T
        d_h *= (h > 0).astype(float)
        d_W1 = X.T @ d_h
        d_b1 = d_h.sum(axis=0)

        W1 -= lr * d_W1
        b1 -= lr * d_b1
        W2 -= lr * d_W2
        b2 -= lr * d_b2

    # Evaluate
    h_all = relu(X @ W1 + b1)
    pred_all = sigmoid(h_all @ W2 + b2)
    acc = np.mean((pred_all > 0.5) == y)
    print(f"  Model accuracy: {acc:.4f}")

    # Activation Patching: For a specific input, replace each hidden neuron's
    # activation with a "clean" reference activation and measure the effect
    # on the output.

    # Select a "positive" example (where the model correctly predicts 1)
    positive_idx = np.where((pred_all.flatten() > 0.5) & (y.flatten() == 1))[0][:5]

    # Reference: average activation across all "negative" examples
    negative_mask = y.flatten() == 0
    h_reference = relu(X[negative_mask] @ W1 + b1).mean(axis=0)

    print(f"\n  Activation Patching Results:")
    print(f"  (Replacing each hidden neuron with mean negative-class activation)")
    print(f"\n  {'Neuron':>8} {'Orig Output':>13} {'Patched Output':>16} "
          f"{'Effect':>8} {'Causal?':>9}")

    cumulative_effects = np.zeros(d_hidden)

    for idx in positive_idx:
        x_test = X[idx:idx+1]
        h_original = relu(x_test @ W1 + b1)
        out_original = sigmoid(h_original @ W2 + b2)[0, 0]

        if idx == positive_idx[0]:
            print(f"\n  Example input: {x_test[0].round(3)}")
            print(f"  True label: {y[idx, 0]:.0f}, Original prediction: {out_original:.4f}")

        for neuron in range(d_hidden):
            h_patched = h_original.copy()
            h_patched[0, neuron] = h_reference[neuron]
            out_patched = sigmoid(h_patched @ W2 + b2)[0, 0]
            effect = out_original - out_patched
            cumulative_effects[neuron] += abs(effect)

            if idx == positive_idx[0]:
                causal = "YES" if abs(effect) > 0.1 else "no"
                print(f"  {neuron:>8} {out_original:>13.4f} {out_patched:>16.4f} "
                      f"{effect:>+8.4f} {causal:>9}")

    # Summary across examples
    print(f"\n  Cumulative causal importance across {len(positive_idx)} examples:")
    ranked_neurons = np.argsort(-cumulative_effects)
    for neuron in ranked_neurons:
        bar = "#" * int(cumulative_effects[neuron] * 20)
        print(f"    Neuron {neuron}: {cumulative_effects[neuron]:.4f} {bar}")

    # Identify the "circuit" (neurons with high causal importance)
    circuit_neurons = ranked_neurons[cumulative_effects[ranked_neurons] > 0.3]
    print(f"\n  Identified circuit neurons (effect > 0.3): {circuit_neurons.tolist()}")
    print(f"  These neurons implement the AND(x0 > 0, x1 > 0) logic.")

    # Verify by checking weight patterns of circuit neurons
    print(f"\n  Weight analysis of circuit neurons:")
    for neuron in circuit_neurons[:3]:
        weights = W1[:, neuron]
        print(f"    Neuron {neuron}: W1 = {weights.round(3)}, b1 = {b1[neuron]:.3f}, "
              f"W2 = {W2[neuron, 0]:.3f}")


# === Exercise 4: Logit Attribution Decomposition ===
# Problem: Decompose the final logit output of a model into additive
# contributions from each component (embedding, attention heads, MLPs)
# to understand which components drive the prediction.

def exercise_4():
    """Analyze logit attribution decomposition in a simplified transformer."""
    np.random.seed(42)

    # Simplified transformer with residual stream
    # Architecture: Embed -> Head1 + Head2 -> MLP -> Unembed
    # All components write to a shared residual stream

    d_model = 16
    vocab_size = 10
    seq_len = 4

    # Initialize component weights
    W_embed = np.random.randn(vocab_size, d_model) * 0.3
    W_head1 = np.random.randn(d_model, d_model) * 0.2
    W_head2 = np.random.randn(d_model, d_model) * 0.2
    W_mlp_in = np.random.randn(d_model, d_model * 2) * 0.2
    W_mlp_out = np.random.randn(d_model * 2, d_model) * 0.2
    W_unembed = np.random.randn(d_model, vocab_size) * 0.3

    def relu(x):
        return np.maximum(0, x)

    # Input sequence
    input_tokens = [3, 7, 1, 5]
    target_token = 7  # We want to understand why the model predicts token 7

    # Forward pass with component tracking
    # Step 1: Embedding
    residual = W_embed[input_tokens[-1]]  # Last token embedding

    # Step 2: Attention Head 1 contribution
    head1_output = residual @ W_head1
    # Simplified: head attends to a weighted average of all positions
    attn_weights_h1 = np.array([0.1, 0.3, 0.1, 0.5])  # Simulated attention
    context_h1 = sum(w * W_embed[t] for w, t in zip(attn_weights_h1, input_tokens))
    head1_contribution = context_h1 @ W_head1
    residual = residual + head1_contribution

    # Step 3: Attention Head 2 contribution
    attn_weights_h2 = np.array([0.2, 0.5, 0.2, 0.1])  # Different attention pattern
    context_h2 = sum(w * W_embed[t] for w, t in zip(attn_weights_h2, input_tokens))
    head2_contribution = context_h2 @ W_head2
    residual = residual + head2_contribution

    # Step 4: MLP contribution
    mlp_input = residual
    mlp_hidden = relu(mlp_input @ W_mlp_in)
    mlp_contribution = mlp_hidden @ W_mlp_out
    residual = residual + mlp_contribution

    # Step 5: Unembed to get logits
    final_logits = residual @ W_unembed

    # Logit attribution: decompose the logit for target token
    # into contributions from each component through the residual stream
    target_unembed = W_unembed[:, target_token]

    # Each component's contribution to the target logit
    embed_logit = W_embed[input_tokens[-1]] @ target_unembed
    head1_logit = head1_contribution @ target_unembed
    head2_logit = head2_contribution @ target_unembed
    mlp_logit = mlp_contribution @ target_unembed
    total_logit = final_logits[target_token]

    print(f"  Logit Attribution Decomposition")
    print(f"  " + "=" * 55)
    print(f"  Input sequence: {input_tokens}")
    print(f"  Target token: {target_token}")
    print(f"\n  Component contributions to logit[{target_token}]:")
    print(f"  {'Component':<25} {'Contribution':>14} {'% of Total':>12}")
    print(f"  {'-' * 51}")

    components = [
        ("Embedding", embed_logit),
        ("Attention Head 1", head1_logit),
        ("Attention Head 2", head2_logit),
        ("MLP Layer", mlp_logit),
    ]

    total_abs = sum(abs(c) for _, c in components)
    for name, contrib in components:
        pct = abs(contrib) / (total_abs + 1e-10) * 100
        bar_len = int(abs(contrib) / (max(abs(c) for _, c in components) + 1e-10) * 20)
        bar = ("+" if contrib > 0 else "-") * max(bar_len, 1)
        print(f"  {name:<25} {contrib:>+14.4f} {pct:>11.1f}% {bar}")

    sum_components = sum(c for _, c in components)
    print(f"  {'-' * 51}")
    print(f"  {'Sum of components':<25} {sum_components:>+14.4f}")
    print(f"  {'Actual logit':<25} {total_logit:>+14.4f}")
    print(f"  {'Residual (nonlinearity)':<25} {total_logit - sum_components:>+14.4f}")

    # Full logit decomposition for all tokens
    print(f"\n  Full logit decomposition (all output tokens):")
    print(f"  {'Token':>6}", end="")
    for name, _ in components:
        short_name = name[:10]
        print(f"  {short_name:>12}", end="")
    print(f"  {'Total':>12}")

    for t in range(vocab_size):
        t_unembed = W_unembed[:, t]
        print(f"  {t:>6}", end="")
        for name, _ in [("Embedding", W_embed[input_tokens[-1]]),
                        ("Head 1", head1_contribution),
                        ("Head 2", head2_contribution),
                        ("MLP", mlp_contribution)]:
            if name == "Embedding":
                c = W_embed[input_tokens[-1]] @ t_unembed
            elif name == "Head 1":
                c = head1_contribution @ t_unembed
            elif name == "Head 2":
                c = head2_contribution @ t_unembed
            else:
                c = mlp_contribution @ t_unembed
            print(f"  {c:>+12.3f}", end="")
        print(f"  {final_logits[t]:>+12.3f}")

    # Identify which component most promotes the target
    dominant = max(components, key=lambda x: x[1])
    print(f"\n  Most promoting component for token {target_token}: {dominant[0]} "
          f"({dominant[1]:+.4f})")

    # Attention pattern analysis
    print(f"\n  Attention pattern analysis:")
    print(f"    Head 1 attends most to position {np.argmax(attn_weights_h1)} "
          f"(token {input_tokens[np.argmax(attn_weights_h1)]})")
    print(f"    Head 2 attends most to position {np.argmax(attn_weights_h2)} "
          f"(token {input_tokens[np.argmax(attn_weights_h2)]})")

    print(f"\n  Logit attribution reveals which transformer components promote")
    print(f"  or suppress each output token, enabling circuit-level analysis.")


if __name__ == "__main__":
    print("=== Exercise 1: Superposition Capacity ===")
    exercise_1()
    print("\n=== Exercise 2: Tiny Sparse Autoencoder ===")
    exercise_2()
    print("\n=== Exercise 3: Activation Patching on 2-Layer Model ===")
    exercise_3()
    print("\n=== Exercise 4: Logit Attribution Decomposition ===")
    exercise_4()
    print("\nAll exercises completed!")
