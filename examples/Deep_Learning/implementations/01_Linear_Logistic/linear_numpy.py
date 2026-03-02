"""
Linear Regression - NumPy From-Scratch Implementation

This file implements linear regression using pure NumPy.
Implements gradient descent without any framework
to understand the fundamental principles of deep learning.

Learning Objectives:
1. Forward pass: y_hat = Xw + b
2. Loss computation: MSE = (1/2n) * ||y - y_hat||^2
3. Backward pass: gradient computation
4. Weight update: w = w - lr * dw
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionNumpy:
    """
    Linear Regression implemented with NumPy

    Mathematical Background:
    - Model: y_hat = Xw + b
    - Loss: L = (1/2n) sum((y - y_hat)^2)
    - Gradients:
        dL/dw = (1/n) X^T (y_hat - y)
        dL/db = (1/n) sum(y_hat - y)
    """

    def __init__(self, input_dim: int, output_dim: int = 1):
        """
        Args:
            input_dim: Number of input features
            output_dim: Output dimension (default 1)
        """
        # Xavier/He initialization: maintain variance at 2/n
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))

        # Gradient storage
        self.dW = None
        self.db = None

        # Cache from forward pass (used in backward)
        self._cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: y_hat = Xw + b

        Args:
            X: Input data (batch_size, input_dim)

        Returns:
            y_hat: Predictions (batch_size, output_dim)
        """
        # Cache input (needed in backward)
        self._cache['X'] = X

        # Linear transform: y = Xw + b
        y_hat = np.dot(X, self.W) + self.b

        return y_hat

    def compute_loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        Mean Squared Error loss computation

        L = (1/2n) sum((y - y_hat)^2)

        Args:
            y: True values (batch_size, output_dim)
            y_hat: Predictions (batch_size, output_dim)

        Returns:
            loss: Scalar loss value
        """
        n = y.shape[0]
        loss = (1 / (2 * n)) * np.sum((y - y_hat) ** 2)
        return loss

    def backward(self, y: np.ndarray, y_hat: np.ndarray) -> None:
        """
        Backward pass: Gradient computation

        Chain Rule applied:
        dL/dw = dL/dy_hat x dy_hat/dw
               = (1/n)(y_hat - y) x X^T
               = (1/n) X^T (y_hat - y)

        dL/db = dL/dy_hat x dy_hat/db
               = (1/n) sum(y_hat - y)

        Args:
            y: True values
            y_hat: Predictions
        """
        X = self._cache['X']
        n = y.shape[0]

        # Error
        error = y_hat - y  # (batch_size, output_dim)

        # Gradient computation
        # dL/dW = (1/n) X^T @ error
        self.dW = (1 / n) * np.dot(X.T, error)

        # dL/db = (1/n) sum(error) (per output dimension)
        self.db = (1 / n) * np.sum(error, axis=0, keepdims=True)

    def update(self, lr: float) -> None:
        """
        Weight update (Gradient Descent)

        w = w - lr x dL/dw
        b = b - lr x dL/db

        Args:
            lr: learning rate
        """
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 1000,
        verbose: bool = True
    ) -> list:
        """
        Train the model

        Args:
            X: Training data (n_samples, n_features)
            y: Target values (n_samples, 1) or (n_samples,)
            lr: learning rate
            epochs: Number of training iterations
            verbose: Whether to print progress

        Returns:
            losses: List of losses per epoch
        """
        # Fix y shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        losses = []

        for epoch in range(epochs):
            # 1. Forward pass
            y_hat = self.forward(X)

            # 2. Loss computation
            loss = self.compute_loss(y, y_hat)
            losses.append(loss)

            # 3. Backward pass (gradient computation)
            self.backward(y, y_hat)

            # 4. Weight update
            self.update(lr)

            # Print progress
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediction"""
        return self.forward(X)


def generate_sample_data(n_samples: int = 100, n_features: int = 1, noise: float = 0.1):
    """
    Generate sample data for testing

    y = 2x + 3 + noise
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)

    # True weights (values to be found through training)
    true_w = np.array([[2.0]])
    true_b = 3.0

    y = np.dot(X, true_w) + true_b + noise * np.random.randn(n_samples, 1)

    return X, y, true_w, true_b


def main():
    """Main execution function"""
    print("=" * 60)
    print("Linear Regression - NumPy From-Scratch Implementation")
    print("=" * 60)

    # 1. Generate data
    print("\n1. Generate sample data")
    X, y, true_w, true_b = generate_sample_data(n_samples=100, noise=0.1)
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   True w: {true_w.flatten()}, True b: {true_b}")

    # 2. Create model
    print("\n2. Model initialization")
    model = LinearRegressionNumpy(input_dim=1, output_dim=1)
    print(f"   Initial W: {model.W.flatten()}")
    print(f"   Initial b: {model.b.flatten()}")

    # 3. Train
    # Expected: loss < 0.01 after 100 epochs, learned W ~ 2.0, b ~ 3.0
    print("\n3. Start training")
    losses = model.fit(X, y, lr=0.1, epochs=100, verbose=True)

    # 4. Check results
    print("\n4. Training results")
    print(f"   Learned W: {model.W.flatten()}")
    print(f"   Learned b: {model.b.flatten()}")
    print(f"   True W: {true_w.flatten()}")
    print(f"   True b: {true_b}")
    print(f"   Final Loss: {losses[-1]:.6f}")

    # 5. Visualization
    print("\n5. Visualization")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)

    # Data and prediction line
    y_pred = model.predict(X)
    sorted_idx = np.argsort(X.flatten())
    axes[1].scatter(X, y, alpha=0.5, label='Data')
    axes[1].plot(X[sorted_idx], y_pred[sorted_idx], 'r-', linewidth=2, label='Prediction')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('y')
    axes[1].set_title('Linear Regression Fit')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('linear_regression_result.png', dpi=150)
    plt.close()
    print("   Result image saved: linear_regression_result.png")


if __name__ == "__main__":
    main()
