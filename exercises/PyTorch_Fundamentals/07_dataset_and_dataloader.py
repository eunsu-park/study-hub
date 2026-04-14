"""
Dataset and DataLoader - Exercises
===================================
Lesson 07: Dataset and DataLoader

Exercises:
  1. Implement a custom Dataset
  2. Create a DataLoader with train/val split
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split


def exercise_1_custom_dataset():
    """Create a Dataset for the function y = sin(x) + noise.

    TODO:
      - Generate 500 x values uniformly in [0, 2*pi]
      - Compute y = sin(x) + gaussian noise (std=0.1)
      - Implement __len__ and __getitem__
      - __getitem__ should return (x_i, y_i) as float tensors

    Returns:
        Dataset: the custom dataset
    """

    class SinDataset(Dataset):
        def __init__(self, n=500):
            # TODO: implement
            raise NotImplementedError

        def __len__(self):
            # TODO: implement
            raise NotImplementedError

        def __getitem__(self, idx):
            # TODO: implement
            raise NotImplementedError

    return SinDataset()


def exercise_2_split_and_load(dataset, train_ratio=0.8, batch_size=32):
    """Split dataset into train/val and create DataLoaders.

    Args:
        dataset: a Dataset
        train_ratio: fraction for training
        batch_size: batch size for DataLoaders

    Returns:
        tuple: (train_loader, val_loader)

    TODO:
      - Compute train/val sizes
      - Use random_split with a fixed generator (seed=42)
      - Create DataLoaders (shuffle=True for train, False for val)
    """
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    print("Exercise 1: Custom Dataset")
    print("-" * 40)
    try:
        ds = exercise_1_custom_dataset()
        assert len(ds) == 500, f"Expected 500, got {len(ds)}"
        x, y = ds[0]
        assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
        assert x.dtype == torch.float32
        print(f"Length: {len(ds)}")
        print(f"Sample: x={x.item():.4f}, y={y.item():.4f}")
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")

    print("\nExercise 2: Split and Load")
    print("-" * 40)
    try:
        ds = exercise_1_custom_dataset()
        train_loader, val_loader = exercise_2_split_and_load(ds)
        train_samples = len(train_loader.dataset)
        val_samples = len(val_loader.dataset)
        print(f"Train: {train_samples}, Val: {val_samples}")
        assert train_samples == 400 and val_samples == 100
        batch = next(iter(train_loader))
        print(f"Batch: x={batch[0].shape}, y={batch[1].shape}")
        print("PASSED")
    except NotImplementedError:
        print("NOT IMPLEMENTED")
