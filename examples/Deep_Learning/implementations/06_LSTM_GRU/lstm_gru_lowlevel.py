"""
PyTorch Low-Level LSTM/GRU Implementation

Uses F.linear, torch.sigmoid, torch.tanh instead of nn.LSTM and nn.GRU.
Parameters are managed manually.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from typing import Tuple, List, Optional


class LSTMCellLowLevel:
    """
    Single LSTM Cell (Low-Level PyTorch)

    Does not use nn.LSTMCell.
    """

    def __init__(self, input_size: int, hidden_size: int, device: torch.device = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Xavier initialization
        concat_size = input_size + hidden_size
        std = math.sqrt(2.0 / (concat_size + hidden_size))

        # 4 gates combined into one: [forget, input, candidate, output]
        # Note: multiply BEFORE enabling requires_grad to keep tensors as leaves
        self.W_ih = (torch.randn(
            4 * hidden_size, input_size, device=self.device
        ) * std).requires_grad_(True)
        self.W_hh = (torch.randn(
            4 * hidden_size, hidden_size, device=self.device
        ) * std).requires_grad_(True)
        self.bias = torch.zeros(
            4 * hidden_size,
            requires_grad=True, device=self.device
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: (batch_size, input_size)
            hx: (h_prev, c_prev) each (batch_size, hidden_size)

        Returns:
            h_t, c_t: each (batch_size, hidden_size)
        """
        h_prev, c_prev = hx
        H = self.hidden_size

        # Gate computation
        gates = (x @ self.W_ih.t() + h_prev @ self.W_hh.t() + self.bias)

        # Split gates
        f = torch.sigmoid(gates[:, 0:H])           # Forget
        i = torch.sigmoid(gates[:, H:2*H])         # Input
        g = torch.tanh(gates[:, 2*H:3*H])          # Candidate
        o = torch.sigmoid(gates[:, 3*H:4*H])       # Output

        # Cell & Hidden
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)

        return h_t, c_t

    def parameters(self) -> List[torch.Tensor]:
        return [self.W_ih, self.W_hh, self.bias]


class GRUCellLowLevel:
    """
    Single GRU Cell (Low-Level PyTorch)

    Does not use nn.GRUCell.
    """

    def __init__(self, input_size: int, hidden_size: int, device: torch.device = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        concat_size = input_size + hidden_size
        std = math.sqrt(2.0 / (concat_size + hidden_size))

        # 3 gates: [reset, update, candidate]
        # Note: multiply BEFORE enabling requires_grad to keep tensors as leaves
        self.W_ih = (torch.randn(
            3 * hidden_size, input_size, device=self.device
        ) * std).requires_grad_(True)
        self.W_hh = (torch.randn(
            3 * hidden_size, hidden_size, device=self.device
        ) * std).requires_grad_(True)
        self.bias = torch.zeros(
            3 * hidden_size,
            requires_grad=True, device=self.device
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch_size, input_size)
            h_prev: (batch_size, hidden_size)

        Returns:
            h_t: (batch_size, hidden_size)
        """
        H = self.hidden_size

        # Reset and Update gates
        ih = x @ self.W_ih.t()
        hh = h_prev @ self.W_hh.t()

        r = torch.sigmoid(ih[:, 0:H] + hh[:, 0:H] + self.bias[0:H])
        z = torch.sigmoid(ih[:, H:2*H] + hh[:, H:2*H] + self.bias[H:2*H])

        # Candidate (with reset applied)
        n = torch.tanh(ih[:, 2*H:3*H] + r * hh[:, 2*H:3*H] + self.bias[2*H:3*H])

        # Hidden
        h_t = (1 - z) * h_prev + z * n

        return h_t

    def parameters(self) -> List[torch.Tensor]:
        return [self.W_ih, self.W_hh, self.bias]


class LSTMLowLevel:
    """
    Multi-layer LSTM (Low-Level PyTorch)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        device: torch.device = None
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_directions = 2 if bidirectional else 1

        # Create cells per layer
        self.cells = []
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                in_size = input_size if layer == 0 else hidden_size * self.num_directions
                cell = LSTMCellLowLevel(in_size, hidden_size, self.device)
                self.cells.append(cell)

    def forward(
        self,
        x: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            x: (seq_len, batch_size, input_size)
            hx: (h_0, c_0) each (num_layers * num_directions, batch, hidden)

        Returns:
            output: (seq_len, batch, hidden * num_directions)
            (h_n, c_n): final states
        """
        seq_len, batch_size, _ = x.shape

        # Initial states
        if hx is None:
            h_0 = torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.hidden_size,
                device=self.device
            )
            c_0 = torch.zeros_like(h_0)
        else:
            h_0, c_0 = hx

        h_states = list(h_0)
        c_states = list(c_0)

        output = x
        new_h_states = []
        new_c_states = []

        for layer in range(self.num_layers):
            # Forward direction
            cell_idx = layer * self.num_directions
            cell = self.cells[cell_idx]

            h, c = h_states[cell_idx], c_states[cell_idx]
            forward_outputs = []

            for t in range(seq_len):
                h, c = cell.forward(output[t], (h, c))
                forward_outputs.append(h)

            new_h_states.append(h)
            new_c_states.append(c)

            if self.bidirectional:
                # Backward direction
                cell = self.cells[cell_idx + 1]
                h, c = h_states[cell_idx + 1], c_states[cell_idx + 1]
                backward_outputs = []

                for t in reversed(range(seq_len)):
                    h, c = cell.forward(output[t], (h, c))
                    backward_outputs.insert(0, h)

                new_h_states.append(h)
                new_c_states.append(c)

                output = torch.cat([
                    torch.stack(forward_outputs),
                    torch.stack(backward_outputs)
                ], dim=-1)
            else:
                output = torch.stack(forward_outputs)

            # Dropout (except last layer)
            if self.dropout > 0 and layer < self.num_layers - 1:
                output = F.dropout(output, p=self.dropout, training=True)

        h_n = torch.stack(new_h_states)
        c_n = torch.stack(new_c_states)

        return output, (h_n, c_n)

    def parameters(self) -> List[torch.Tensor]:
        params = []
        for cell in self.cells:
            params.extend(cell.parameters())
        return params

    def zero_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()


class GRULowLevel:
    """
    Multi-layer GRU (Low-Level PyTorch)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        device: torch.device = None
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_directions = 2 if bidirectional else 1

        self.cells = []
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                in_size = input_size if layer == 0 else hidden_size * self.num_directions
                cell = GRUCellLowLevel(in_size, hidden_size, self.device)
                self.cells.append(cell)

    def forward(
        self,
        x: torch.Tensor,
        h_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: (seq_len, batch_size, input_size)
            h_0: (num_layers * num_directions, batch, hidden)

        Returns:
            output: (seq_len, batch, hidden * num_directions)
            h_n: final hidden state
        """
        seq_len, batch_size, _ = x.shape

        if h_0 is None:
            h_0 = torch.zeros(
                self.num_layers * self.num_directions, batch_size, self.hidden_size,
                device=self.device
            )

        h_states = list(h_0)
        output = x
        new_h_states = []

        for layer in range(self.num_layers):
            cell_idx = layer * self.num_directions
            cell = self.cells[cell_idx]

            h = h_states[cell_idx]
            forward_outputs = []

            for t in range(seq_len):
                h = cell.forward(output[t], h)
                forward_outputs.append(h)

            new_h_states.append(h)

            if self.bidirectional:
                cell = self.cells[cell_idx + 1]
                h = h_states[cell_idx + 1]
                backward_outputs = []

                for t in reversed(range(seq_len)):
                    h = cell.forward(output[t], h)
                    backward_outputs.insert(0, h)

                new_h_states.append(h)

                output = torch.cat([
                    torch.stack(forward_outputs),
                    torch.stack(backward_outputs)
                ], dim=-1)
            else:
                output = torch.stack(forward_outputs)

            # Dropout (except last layer)
            if self.dropout > 0 and layer < self.num_layers - 1:
                output = F.dropout(output, p=self.dropout, training=True)

        h_n = torch.stack(new_h_states)

        return output, h_n

    def parameters(self) -> List[torch.Tensor]:
        params = []
        for cell in self.cells:
            params.extend(cell.parameters())
        return params

    def zero_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()


class SequenceClassifier:
    """
    LSTM/GRU-based Sequence Classifier
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        rnn_type: str = 'lstm',
        device: torch.device = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Embedding (multiply before requires_grad to keep as leaf)
        self.embedding = (torch.randn(
            vocab_size, embed_size, device=self.device
        ) * 0.1).requires_grad_(True)

        # RNN
        if rnn_type == 'lstm':
            self.rnn = LSTMLowLevel(
                embed_size, hidden_size, num_layers,
                bidirectional, dropout=0.3, device=self.device
            )
        else:
            self.rnn = GRULowLevel(
                embed_size, hidden_size, num_layers,
                bidirectional, dropout=0.3, device=self.device
            )

        # Classifier
        fc_in = hidden_size * (2 if bidirectional else 1)
        std = math.sqrt(2.0 / (fc_in + num_classes))
        self.fc_weight = (torch.randn(
            num_classes, fc_in, device=self.device
        ) * std).requires_grad_(True)
        self.fc_bias = torch.zeros(num_classes, requires_grad=True, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) token indices

        Returns:
            logits: (batch_size, num_classes)
        """
        # Embedding
        embedded = F.embedding(x, self.embedding)  # (batch, seq, embed)
        embedded = embedded.transpose(0, 1)  # (seq, batch, embed)

        # RNN
        if isinstance(self.rnn, LSTMLowLevel):
            output, (h_n, c_n) = self.rnn.forward(embedded)
        else:
            output, h_n = self.rnn.forward(embedded)

        # Last hidden (concat if bidirectional)
        if self.rnn.bidirectional:
            # Forward's last + Backward's first
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            last_hidden = h_n[-1]

        # Classifier
        logits = last_hidden @ self.fc_weight.t() + self.fc_bias

        return logits

    def parameters(self) -> List[torch.Tensor]:
        params = [self.embedding]
        params.extend(self.rnn.parameters())
        params.extend([self.fc_weight, self.fc_bias])
        return params

    def zero_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()


def train_imdb_sentiment():
    """IMDB sentiment analysis (simplified version)"""
    print("=== LSTM/GRU Sentiment Analysis ===\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dummy data (in practice, use torchtext)
    vocab_size = 10000
    seq_len = 100
    batch_size = 32
    num_samples = 1000

    # Synthetic training data
    X_train = torch.randint(0, vocab_size, (num_samples, seq_len), device=device)
    y_train = torch.randint(0, 2, (num_samples,), device=device)

    # Model
    model = SequenceClassifier(
        vocab_size=vocab_size,
        embed_size=128,
        hidden_size=256,
        num_classes=2,
        num_layers=2,
        bidirectional=True,
        rnn_type='lstm',
        device=device
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # Expected: loss decreasing over 5 epochs, accuracy ~0.50-0.55 (random data baseline)
    lr = 0.001
    epochs = 5
    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0

        for i in range(0, num_samples, batch_size):
            batch_x = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            # Forward
            logits = model.forward(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            # Backward
            model.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # SGD update (use .data to keep leaf tensor status)
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param.data -= lr * param.grad

            total_loss += loss.item() * len(batch_y)
            total_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        avg_loss = total_loss / num_samples
        accuracy = total_correct / num_samples
        loss_history.append(avg_loss)
        acc_history.append(accuracy)
        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(range(1, epochs + 1), loss_history, 'b-o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('LSTM/GRU Training Loss')
    axes[0].grid(True)

    axes[1].plot(range(1, epochs + 1), acc_history, 'r-o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('LSTM/GRU Training Accuracy')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('lstm_gru_lowlevel_training.png', dpi=150)
    plt.close()
    print("Result image saved: lstm_gru_lowlevel_training.png")


def main():
    """Test"""
    print("=== LSTM/GRU Low-Level Test ===\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # LSTM test
    print("Testing LSTM...")
    lstm = LSTMLowLevel(
        input_size=10, hidden_size=20,
        num_layers=2, bidirectional=True, device=device
    )

    x = torch.randn(5, 3, 10, device=device)  # (seq, batch, input)
    output, (h_n, c_n) = lstm.forward(x)

    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")  # (5, 3, 40) bidirectional
    print(f"  h_n: {h_n.shape}")  # (4, 3, 20) 2 layers * 2 directions

    # GRU test
    print("\nTesting GRU...")
    gru = GRULowLevel(
        input_size=10, hidden_size=20,
        num_layers=2, bidirectional=False, device=device
    )

    output, h_n = gru.forward(x)
    print(f"  Output: {output.shape}")  # (5, 3, 20)
    print(f"  h_n: {h_n.shape}")  # (2, 3, 20)

    # Sentiment analysis training
    print()
    train_imdb_sentiment()


if __name__ == "__main__":
    main()
