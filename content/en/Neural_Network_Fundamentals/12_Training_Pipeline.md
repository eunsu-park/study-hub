# Training Pipeline

**Previous**: [Universal Approximation](./11_Universal_Approximation.md) | **Next**: [Building MLP from Scratch](./13_Building_MLP_from_Scratch.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design a complete neural network training pipeline from data to evaluation
2. Implement proper train/validation/test splitting strategies
3. Apply k-fold cross-validation for model selection
4. Perform systematic hyperparameter tuning using grid and random search
5. Implement a training loop with metric logging and checkpointing
6. Diagnose underfitting and overfitting from learning curves
7. Apply the complete pipeline to a classification problem
8. List the steps in a reproducible experimental workflow

---

Knowing how individual components work (activation functions, loss functions, optimizers) is necessary but not sufficient. To build effective neural networks, you need a disciplined training pipeline that handles data splitting, hyperparameter search, model evaluation, and diagnostics. This lesson assembles all the pieces into a complete, repeatable workflow.

---

## 1. The Complete Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                      │
│                                                          │
│  1. Data Preparation                                     │
│     ├── Load & inspect data                              │
│     ├── Handle missing values                            │
│     ├── Feature scaling (standardize / normalize)        │
│     └── Train / Validation / Test split                  │
│                                                          │
│  2. Model Design                                         │
│     ├── Choose architecture (layers, neurons)            │
│     ├── Choose activation functions                      │
│     ├── Choose loss function                             │
│     └── Choose optimizer                                 │
│                                                          │
│  3. Training Loop                                        │
│     ├── Forward pass                                     │
│     ├── Compute loss                                     │
│     ├── Backward pass (gradients)                        │
│     ├── Update weights                                   │
│     └── Log metrics (train loss, val loss, accuracy)     │
│                                                          │
│  4. Hyperparameter Tuning                                │
│     ├── Grid search / Random search                      │
│     ├── Learning rate, batch size, architecture          │
│     └── Select best model based on validation metric     │
│                                                          │
│  5. Evaluation                                           │
│     ├── Evaluate on test set (final, one-time)           │
│     ├── Report metrics with confidence intervals         │
│     └── Analyze errors                                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Data Preparation

### 2.1 Feature Scaling

Neural networks are sensitive to input scale. Always standardize or normalize:

```python
import numpy as np

class StandardScaler:
    """Standardize features to zero mean and unit variance."""

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8  # prevent division by zero
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# IMPORTANT: fit on training data only, transform both train and test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)      # use train statistics!
X_test_scaled = scaler.transform(X_test)    # use train statistics!
```

### 2.2 Data Splitting

```
Total Data
├── Training Set (60-80%)     ← Model learns from this
├── Validation Set (10-20%)   ← Tune hyperparameters, monitor overfitting
└── Test Set (10-20%)         ← Final evaluation ONLY (never tune on this!)

Rule: NEVER look at test data during training or hyperparameter selection.
```

```python
def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split data into train/validation/test sets."""
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])
```

### 2.3 Stratified Splitting

For classification, ensure each split has the same class proportions:

```python
def stratified_split(X, y, test_ratio=0.2, seed=42):
    """Stratified split preserving class proportions."""
    np.random.seed(seed)
    classes = np.unique(y)
    train_idx, test_idx = [], []

    for c in classes:
        c_idx = np.where(y == c)[0]
        np.random.shuffle(c_idx)
        n_test = int(len(c_idx) * test_ratio)
        test_idx.extend(c_idx[:n_test])
        train_idx.extend(c_idx[n_test:])

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
```

---

## 3. The Training Loop

### 3.1 Mini-Batch Training

```python
def create_mini_batches(X, y, batch_size, shuffle=True):
    """Create mini-batches from data."""
    n = X.shape[1]  # number of samples (columns)
    if shuffle:
        perm = np.random.permutation(n)
        X = X[:, perm]
        y = y[:, perm]

    batches = []
    for i in range(0, n, batch_size):
        X_batch = X[:, i:i + batch_size]
        y_batch = y[:, i:i + batch_size]
        batches.append((X_batch, y_batch))
    return batches

def train_epoch(params, X, Y, batch_size, optimizer, forward_fn, backward_fn):
    """Train for one epoch."""
    batches = create_mini_batches(X, Y, batch_size)
    epoch_loss = 0.0

    for X_batch, Y_batch in batches:
        # Forward pass
        y_pred, caches = forward_fn(X_batch, params)
        loss = cross_entropy_loss(y_pred, Y_batch)
        epoch_loss += loss * X_batch.shape[1]

        # Backward pass
        grads = backward_fn(Y_batch, params, caches)

        # Update weights
        params = optimizer.update(params, grads)

    return params, epoch_loss / X.shape[1]
```

### 3.2 Complete Training with Logging

```python
def train(params, X_train, Y_train, X_val, Y_val,
          n_epochs, batch_size, optimizer, forward_fn, backward_fn):
    """Full training loop with validation monitoring."""
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    early_stop = EarlyStopping(patience=20)

    for epoch in range(n_epochs):
        # Training
        params, train_loss = train_epoch(
            params, X_train, Y_train, batch_size,
            optimizer, forward_fn, backward_fn
        )

        # Validation
        val_pred, _ = forward_fn(X_val, params)
        val_loss = cross_entropy_loss(val_pred, Y_val)
        val_acc = np.mean(np.argmax(val_pred, axis=0) == np.argmax(Y_val, axis=0))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        if early_stop.check(val_loss, params):
            params = early_stop.best_params
            break

    return params, history
```

---

## 4. Hyperparameter Tuning

### 4.1 Key Hyperparameters

```
┌────────────────────────────────────────────────────────────┐
│ Hyperparameter    │ Typical Range      │ Search Scale      │
├───────────────────┼────────────────────┼───────────────────┤
│ Learning rate     │ 1e-4 to 1e-1       │ Log scale         │
│ Batch size        │ 16, 32, 64, 128    │ Powers of 2       │
│ Hidden layers     │ 1 to 5             │ Integer            │
│ Neurons per layer │ 32 to 512          │ Powers of 2        │
│ Dropout rate      │ 0.0 to 0.5         │ Linear             │
│ L2 (weight decay) │ 1e-5 to 1e-2      │ Log scale          │
│ Optimizer         │ SGD, Adam, AdamW   │ Categorical        │
└────────────────────────────────────────────────────────────┘
```

### 4.2 Random Search

Random search is more efficient than grid search for most hyperparameter spaces:

```python
def random_search(X_train, Y_train, X_val, Y_val, n_trials=20):
    """Random hyperparameter search."""
    best_val_acc = 0
    best_config = None

    for trial in range(n_trials):
        # Sample hyperparameters
        config = {
            'lr': 10 ** np.random.uniform(-4, -1),
            'hidden_size': np.random.choice([32, 64, 128, 256]),
            'n_layers': np.random.choice([1, 2, 3]),
            'dropout': np.random.uniform(0.0, 0.5),
            'l2_reg': 10 ** np.random.uniform(-5, -2),
        }

        # Build and train model with this config
        # ... (build network, train, evaluate)

        print(f"Trial {trial+1}: {config} -> val_acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = config

    print(f"\nBest config: {best_config}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    return best_config
```

### 4.3 K-Fold Cross-Validation

When data is limited, use k-fold CV for more reliable model evaluation:

```
5-Fold Cross-Validation:

Fold 1:  [VAL] [Train] [Train] [Train] [Train]  → score_1
Fold 2:  [Train] [VAL] [Train] [Train] [Train]  → score_2
Fold 3:  [Train] [Train] [VAL] [Train] [Train]  → score_3
Fold 4:  [Train] [Train] [Train] [VAL] [Train]  → score_4
Fold 5:  [Train] [Train] [Train] [Train] [VAL]  → score_5

Final score = mean(score_1, ..., score_5) ± std
```

```python
def k_fold_cv(X, y, k=5, build_and_train_fn=None):
    """K-fold cross-validation."""
    n = X.shape[1]
    fold_size = n // k
    scores = []

    indices = np.random.permutation(n)

    for i in range(k):
        val_idx = indices[i * fold_size:(i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size],
                                    indices[(i + 1) * fold_size:]])

        X_train, Y_train = X[:, train_idx], y[:, train_idx]
        X_val, Y_val = X[:, val_idx], y[:, val_idx]

        score = build_and_train_fn(X_train, Y_train, X_val, Y_val)
        scores.append(score)
        print(f"Fold {i+1}: {score:.4f}")

    print(f"Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores
```

---

## 5. Learning Curve Diagnostics

### 5.1 Interpreting Learning Curves

```python
import matplotlib.pyplot as plt

def plot_learning_curves(history):
    """Plot training and validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curves')

    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy Curve')

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150)
    plt.show()
```

### 5.2 Diagnosis Guide

```
Symptom                           │ Diagnosis        │ Action
──────────────────────────────────┼──────────────────┼─────────────────
Train ↓, Val ↓ (both improving)  │ Good training    │ Continue
Train ↓, Val ↑ (diverging)       │ Overfitting      │ More regularization
Train ↓ slowly, Val ↓ slowly     │ Underfitting     │ Bigger model, less reg
Train oscillates wildly           │ LR too high      │ Reduce learning rate
Train barely decreases            │ LR too low       │ Increase learning rate
Train ↓, Val plateaus             │ Capacity limit   │ Bigger model or better data
```

---

## 6. Model Evaluation

### 6.1 Classification Metrics

```python
def evaluate_classification(y_pred, y_true):
    """Compute classification metrics."""
    predicted_classes = np.argmax(y_pred, axis=0)
    true_classes = np.argmax(y_true, axis=0)

    accuracy = np.mean(predicted_classes == true_classes)

    # Per-class metrics
    n_classes = y_true.shape[0]
    for c in range(n_classes):
        tp = np.sum((predicted_classes == c) & (true_classes == c))
        fp = np.sum((predicted_classes == c) & (true_classes != c))
        fn = np.sum((predicted_classes != c) & (true_classes == c))
        precision = tp / (tp + fp + 1e-15)
        recall = tp / (tp + fn + 1e-15)
        f1 = 2 * precision * recall / (precision + recall + 1e-15)
        print(f"Class {c}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    print(f"Accuracy: {accuracy:.4f}")
    return accuracy
```

### 6.2 Confusion Matrix

```python
def confusion_matrix(y_pred, y_true, n_classes):
    """Compute confusion matrix."""
    pred = np.argmax(y_pred, axis=0)
    true = np.argmax(y_true, axis=0)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for p, t in zip(pred, true):
        cm[t, p] += 1
    return cm
```

---

## 7. Reproducibility Checklist

```
□ Set random seeds (numpy, python random)
□ Record all hyperparameters
□ Log training metrics (loss, accuracy per epoch)
□ Save best model checkpoint
□ Record software versions (Python, NumPy)
□ Document data preprocessing steps
□ Save the train/val/test split indices
□ Report results with confidence intervals (k-fold or bootstrap)
```

```python
def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)
```

---

## 8. Summary

```
Key Takeaways
═══════════════════════════════════════════════════════
1. Pipeline: data prep → model design → train → tune → evaluate
2. Scale features; fit on train, transform all splits
3. Never touch test data until final evaluation
4. Use mini-batches (32-256) with shuffling each epoch
5. Random search > grid search for hyperparameter tuning
6. K-fold CV for reliable evaluation with limited data
7. Diagnose problems from learning curves
8. Reproducibility: seeds, logging, versioning
═══════════════════════════════════════════════════════
```

---

## Exercises

1. Build a complete training pipeline for Iris classification using only NumPy
2. Implement random search over learning rate and hidden size; find the best config
3. Plot learning curves and identify whether your model is overfitting or underfitting
4. Implement 5-fold cross-validation and report mean ± std accuracy

---

**Previous**: [Universal Approximation](./11_Universal_Approximation.md) | **Next**: [Building MLP from Scratch](./13_Building_MLP_from_Scratch.md)
