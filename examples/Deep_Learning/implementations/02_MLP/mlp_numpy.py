"""
Multi-Layer Perceptron - NumPy From-Scratch Implementation

This file implements MLP using pure NumPy.
Directly implements the Backpropagation algorithm
to understand the core principles of deep learning.

Learning Objectives:
1. Forward pass: Multi-layer neural network forward propagation
2. Backward pass: Backpropagation using chain rule
3. Activation functions: ReLU, Sigmoid, Tanh
4. Weight initialization: Xavier, He initialization
"""

import numpy as np
import matplotlib.pyplot as plt


class ActivationFunctions:
    """Activation functions and their derivatives"""

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z):
        # Clipping for numerical stability
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        s = ActivationFunctions.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def softmax(z):
        # Numerical stability: subtract max
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class Layer:
    """
    Single Fully Connected Layer

    z = Wx + b (linear transform)
    a = sigma(z) (activation)
    """

    def __init__(self, input_dim: int, output_dim: int, activation: str = 'relu'):
        """
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            activation: 'relu', 'sigmoid', 'tanh', 'none'
        """
        # He initialization (for ReLU)
        if activation == 'relu':
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        else:
            # Xavier initialization
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1.0 / input_dim)

        self.b = np.zeros((1, output_dim))

        self.activation = activation
        self._get_activation_fn()

        # Gradients
        self.dW = None
        self.db = None

        # Cache (for backward)
        self.cache = {}

    def _get_activation_fn(self):
        """Set activation function"""
        activations = {
            'relu': (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
            'sigmoid': (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
            'tanh': (ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
            'none': (lambda x: x, lambda x: np.ones_like(x)),
        }
        self.act_fn, self.act_derivative = activations[self.activation]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Args:
            x: Input (batch_size, input_dim)

        Returns:
            a: Activation output (batch_size, output_dim)
        """
        # Save to cache (used in backward)
        self.cache['x'] = x

        # Linear transform: z = Wx + b
        z = np.dot(x, self.W) + self.b
        self.cache['z'] = z

        # Activation: a = sigma(z)
        a = self.act_fn(z)
        self.cache['a'] = a

        return a

    def backward(self, da: np.ndarray) -> np.ndarray:
        """
        Backward pass

        Args:
            da: Gradient of output (batch_size, output_dim)

        Returns:
            dx: Gradient of input (batch_size, input_dim)
        """
        x = self.cache['x']
        z = self.cache['z']
        batch_size = x.shape[0]

        # dL/dz = dL/da x da/dz = da x sigma'(z)
        dz = da * self.act_derivative(z)

        # dL/dW = x^T x dL/dz
        self.dW = np.dot(x.T, dz) / batch_size

        # dL/db = sum(dL/dz)
        self.db = np.sum(dz, axis=0, keepdims=True) / batch_size

        # dL/dx = dL/dz x W^T (propagate to next layer)
        dx = np.dot(dz, self.W.T)

        return dx


class MLPNumpy:
    """
    Multi-Layer Perceptron (NumPy implementation)

    Usage:
        model = MLPNumpy([784, 256, 128, 10], activations=['relu', 'relu', 'none'])
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    """

    def __init__(self, layer_dims: list, activations: list = None):
        """
        Args:
            layer_dims: Dimensions for each layer [input, hidden1, hidden2, ..., output]
            activations: Activation function for each layer (excluding last layer)
        """
        self.layers = []
        n_layers = len(layer_dims) - 1

        if activations is None:
            activations = ['relu'] * (n_layers - 1) + ['none']

        for i in range(n_layers):
            layer = Layer(layer_dims[i], layer_dims[i + 1], activations[i])
            self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full network forward pass"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad: np.ndarray) -> None:
        """Full network backward pass"""
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Cross-entropy loss (for classification)

        L = -1/n x sum( y_true x log(y_pred) )
        """
        eps = 1e-15  # Numerical stability
        y_pred = np.clip(y_pred, eps, 1 - eps)

        if y_true.ndim == 1:
            # Sparse labels -> one-hot
            n_classes = y_pred.shape[1]
            y_true_onehot = np.zeros((len(y_true), n_classes))
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            y_true = y_true_onehot

        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss

    def compute_loss_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Cross-entropy gradient (assuming softmax output)

        dL/dz = y_pred - y_true (simplified for softmax + CE)
        """
        if y_true.ndim == 1:
            n_classes = y_pred.shape[1]
            y_true_onehot = np.zeros((len(y_true), n_classes))
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            y_true = y_true_onehot

        return y_pred - y_true

    def update_weights(self, lr: float) -> None:
        """SGD weight update"""
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
        verbose: bool = True
    ) -> list:
        """
        Train the model

        Args:
            X: Training data (n_samples, n_features)
            y: Labels (n_samples,) or (n_samples, n_classes)
            epochs: Number of epochs
            lr: learning rate
            batch_size: Batch size
            verbose: Whether to print progress

        Returns:
            losses: List of losses per epoch
        """
        n_samples = X.shape[0]
        losses = []

        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices] if y.ndim == 1 else y[indices]

            epoch_loss = 0

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward
                y_pred = self.forward(X_batch)

                # Softmax (if last layer is none)
                y_pred = ActivationFunctions.softmax(y_pred)

                # Loss
                loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += loss * len(X_batch)

                # Backward
                loss_grad = self.compute_loss_gradient(y_batch, y_pred)
                self.backward(loss_grad)

                # Update
                self.update_weights(lr)

            epoch_loss /= n_samples
            losses.append(epoch_loss)

            if verbose and (epoch + 1) % (epochs // 10) == 0:
                accuracy = self.evaluate(X, y)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediction"""
        logits = self.forward(X)
        probs = ActivationFunctions.softmax(logits)
        return np.argmax(probs, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy evaluation"""
        predictions = self.predict(X)
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        return np.mean(predictions == y)


def load_mnist_sample(n_samples=1000):
    """Generate MNIST sample data (for testing)"""
    np.random.seed(42)

    # Simple synthetic data (in practice, load actual MNIST)
    n_classes = 10
    n_features = 784  # 28x28

    X = np.random.randn(n_samples, n_features) * 0.5
    y = np.random.randint(0, n_classes, n_samples)

    # Add some per-class patterns
    for i in range(n_classes):
        mask = y == i
        X[mask, i * 78:(i + 1) * 78] += 1.0

    return X, y


def main():
    """Main execution function"""
    print("=" * 60)
    print("Multi-Layer Perceptron - NumPy From-Scratch Implementation")
    print("=" * 60)

    # 1. Generate data
    print("\n1. Generate sample data")
    X_train, y_train = load_mnist_sample(n_samples=1000)
    X_test, y_test = load_mnist_sample(n_samples=200)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # 2. Create model
    print("\n2. MLP model initialization")
    model = MLPNumpy(
        layer_dims=[784, 128, 64, 10],
        activations=['relu', 'relu', 'none']
    )
    print(f"   Layers: {[l.W.shape for l in model.layers]}")

    # 3. Train
    # Expected: train accuracy ~0.90+, test accuracy ~0.85+ after 50 epochs
    print("\n3. Start training")
    losses = model.fit(
        X_train, y_train,
        epochs=50,
        lr=0.1,
        batch_size=32,
        verbose=True
    )

    # 4. Evaluate
    print("\n4. Evaluation results")
    train_acc = model.evaluate(X_train, y_train)
    test_acc = model.evaluate(X_test, y_test)
    print(f"   Train Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")

    # 5. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)

    # Weight distribution (first layer)
    axes[1].hist(model.layers[0].W.flatten(), bins=50, alpha=0.7)
    axes[1].set_xlabel('Weight Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('First Layer Weight Distribution')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('mlp_result.png', dpi=150)
    plt.close()
    print("\nResult image saved: mlp_result.png")


if __name__ == "__main__":
    main()
