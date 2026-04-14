"""
Dataset and DataLoader - Examples
=================================
Lesson 07: Dataset and DataLoader

Demonstrates:
  1. Custom Dataset implementation
  2. DataLoader with batching and shuffling
  3. Built-in datasets (MNIST via synthetic)
  4. Custom collate_fn for variable-length data
  5. random_split for train/val split
"""

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split


class SyntheticDataset(Dataset):
    """A simple synthetic classification dataset."""

    def __init__(self, n_samples=500, n_features=10, n_classes=3):
        torch.manual_seed(42)
        self.X = torch.randn(n_samples, n_features)
        self.y = torch.randint(0, n_classes, (n_samples,))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def example_1_custom_dataset():
    """Create and use a custom Dataset."""
    print("=" * 60)
    print("Example 1: Custom Dataset")
    print("=" * 60)

    dataset = SyntheticDataset(100, 5, 2)
    print(f"Length: {len(dataset)}")
    x, y = dataset[0]
    print(f"Sample 0: features={x.shape}, label={y}")


def example_2_dataloader():
    """DataLoader with batching, shuffling."""
    print("\n" + "=" * 60)
    print("Example 2: DataLoader")
    print("=" * 60)

    dataset = SyntheticDataset(100, 5, 2)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(loader)}")

    for i, (batch_x, batch_y) in enumerate(loader):
        if i == 0:
            print(f"First batch: X={batch_x.shape}, y={batch_y.shape}")
            print(f"Labels: {batch_y}")


def example_3_tensor_dataset():
    """Quick dataset from tensors using TensorDataset."""
    print("\n" + "=" * 60)
    print("Example 3: TensorDataset")
    print("=" * 60)

    X = torch.randn(200, 10)
    y = torch.randint(0, 3, (200,))

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    batch = next(iter(loader))
    print(f"Batch: X={batch[0].shape}, y={batch[1].shape}")


def example_4_collate_fn():
    """Custom collate_fn for variable-length sequences."""
    print("\n" + "=" * 60)
    print("Example 4: Custom collate_fn")
    print("=" * 60)

    class VarLenDataset(Dataset):
        def __init__(self):
            self.data = [
                (torch.randn(3), 0),
                (torch.randn(5), 1),
                (torch.randn(2), 0),
                (torch.randn(7), 1),
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    def pad_collate(batch):
        sequences, labels = zip(*batch)
        lengths = [len(s) for s in sequences]
        max_len = max(lengths)
        padded = torch.zeros(len(sequences), max_len)
        for i, (seq, length) in enumerate(zip(sequences, lengths)):
            padded[i, :length] = seq
        return padded, torch.tensor(labels), torch.tensor(lengths)

    loader = DataLoader(VarLenDataset(), batch_size=2, collate_fn=pad_collate)
    padded, labels, lengths = next(iter(loader))
    print(f"Padded: {padded.shape}")
    print(f"Labels: {labels}")
    print(f"Lengths: {lengths}")


def example_5_split():
    """Train/validation split with random_split."""
    print("\n" + "=" * 60)
    print("Example 5: Train/Val Split")
    print("=" * 60)

    dataset = SyntheticDataset(1000, 10, 3)
    train_set, val_set = random_split(
        dataset, [800, 200],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Total: {len(dataset)}")
    print(f"Train: {len(train_set)}")
    print(f"Val:   {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")


if __name__ == "__main__":
    example_1_custom_dataset()
    example_2_dataloader()
    example_3_tensor_dataset()
    example_4_collate_fn()
    example_5_split()
