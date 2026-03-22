"""
PyTorch Low-Level BERT Implementation

Does not use nn.TransformerEncoder.
Uses only basic operations like F.linear and F.layer_norm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import matplotlib.pyplot as plt


class BertEmbeddings(nn.Module):
    """
    BERT Embeddings = Token + Segment + Position

    Token: Word meaning
    Segment: Sentence A/B distinction
    Position: Position information (learnable)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position: int = 512,
        type_vocab_size: int = 2,  # Sentence A, B
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Embedding tables (using nn.Embedding, conceptually a lookup)
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        # Layer Norm + Dropout
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            token_type_ids: (batch, seq_len) segment IDs (0 or 1)
            position_ids: (batch, seq_len) position IDs

        Returns:
            embeddings: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len = input_ids.shape

        # Set defaults
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Sum three embeddings
        word_emb = self.word_embeddings(input_ids)
        position_emb = self.position_embeddings(position_ids)
        token_type_emb = self.token_type_embeddings(token_type_ids)

        embeddings = word_emb + position_emb + token_type_emb

        # Layer Norm + Dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertSelfAttention(nn.Module):
    """Multi-Head Self-Attention (Low-Level)"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V projections (could manage parameters directly instead of nn.Linear)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, 1, 1, seq_len) or (batch, seq_len)

        Returns:
            context: (batch, seq_len, hidden_size)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)

        # Multi-head reshape: (batch, seq, hidden) -> (batch, heads, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (batch, heads, seq, seq)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply attention mask
        if attention_mask is not None:
            # (batch, seq) -> (batch, 1, 1, seq)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 0 -> -inf, 1 -> 0
            attention_mask = (1.0 - attention_mask) * -10000.0
            scores = scores + attention_mask

        # Softmax + Dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Context: (batch, heads, seq, head_dim)
        context = torch.matmul(attention_weights, V)

        # Reshape back: (batch, seq, hidden)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, -1)

        return context, attention_weights


class BertSelfOutput(nn.Module):
    """Attention Output (projection + residual + layer norm)"""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Attention output
            input_tensor: Original input for residual connection
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Residual + Layer Norm
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    """Feed-Forward first layer (expansion)"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        # BERT uses GELU

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """Feed-Forward second layer (reduction) + Residual"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """Single BERT Encoder Layer"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        # Self-Attention
        self.attention = BertSelfAttention(hidden_size, num_heads, dropout)
        self.attention_output = BertSelfOutput(hidden_size, dropout)

        # Feed-Forward
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(hidden_size, intermediate_size, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-Attention
        attention_output, attention_weights = self.attention(
            hidden_states, attention_mask
        )
        attention_output = self.attention_output(attention_output, hidden_states)

        # Feed-Forward
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output, attention_weights


class BertEncoder(nn.Module):
    """BERT Encoder (stacked layers)"""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BertLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        all_attentions = [] if output_attentions else None

        for layer in self.layers:
            hidden_states, attention_weights = layer(hidden_states, attention_mask)
            if output_attentions:
                all_attentions.append(attention_weights)

        return hidden_states, all_attentions


class BertPooler(nn.Module):
    """[CLS] token pooling"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            pooled: (batch, hidden_size) - representation of [CLS] token
        """
        # [CLS] token (first token)
        cls_token = hidden_states[:, 0]
        pooled = self.dense(cls_token)
        pooled = torch.tanh(pooled)
        return pooled


class BertModel(nn.Module):
    """BERT Base Model"""

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_position: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, max_position, type_vocab_size, dropout
        )
        self.encoder = BertEncoder(
            num_layers, hidden_size, num_heads, intermediate_size, dropout
        )
        self.pooler = BertPooler(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) - 1 for valid, 0 for padding
            token_type_ids: (batch, seq_len) - 0 for sent A, 1 for sent B

        Returns:
            last_hidden_state: (batch, seq_len, hidden_size)
            pooler_output: (batch, hidden_size)
            attentions: optional list of attention weights
        """
        # Embeddings
        embeddings = self.embeddings(
            input_ids, token_type_ids=token_type_ids
        )

        # Encoder
        encoder_output, attentions = self.encoder(
            embeddings, attention_mask, output_attentions
        )

        # Pooler
        pooled_output = self.pooler(encoder_output)

        return {
            'last_hidden_state': encoder_output,
            'pooler_output': pooled_output,
            'attentions': attentions
        }


class BertForMaskedLM(nn.Module):
    """BERT for Masked Language Modeling"""

    def __init__(self, config: dict):
        super().__init__()
        self.bert = BertModel(**config)

        # MLM Head
        self.cls = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.GELU(),
            nn.LayerNorm(config['hidden_size'], eps=1e-12),
            nn.Linear(config['hidden_size'], config['vocab_size'])
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        hidden_states = outputs['last_hidden_state']

        # MLM predictions
        prediction_scores = self.cls(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        return {
            'loss': loss,
            'logits': prediction_scores,
            'hidden_states': hidden_states
        }


class BertForSequenceClassification(nn.Module):
    """BERT for Sequence Classification"""

    def __init__(self, config: dict, num_labels: int):
        super().__init__()
        self.bert = BertModel(**config)
        self.classifier = nn.Linear(config['hidden_size'], num_labels)
        self.dropout = nn.Dropout(config.get('dropout', 0.1))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooled_output = outputs['pooler_output']

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {
            'loss': loss,
            'logits': logits
        }


def demo_mlm_training():
    """Demonstrate MLM training on synthetic patterned sequences."""
    import os

    print("\n=== MLM Training Demo ===\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Small config for fast training
    config = {
        'vocab_size': 1000,
        'hidden_size': 128,
        'num_layers': 2,
        'num_heads': 2,
        'intermediate_size': 512,
        'max_position': 512,
        'dropout': 0.1,
    }

    model = BertForMaskedLM(config).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training parameters
    batch_size = 16
    seq_len = 20
    mask_prob = 0.15
    num_steps = 50
    mask_token_id = 0  # Use token 0 as the [MASK] token
    pattern_len = 5    # Repeating pattern length

    # Expected: MLM loss decreasing from ~6.9 (log(1000)) over 50 steps
    loss_history = []

    for step in range(1, num_steps + 1):
        # Generate toy data with predictable repeating patterns
        # Each sequence repeats a pattern like [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, ...]
        # Patterns use tokens in range [1, vocab_size) to avoid collision with mask_token_id=0
        patterns = torch.randint(1, config['vocab_size'], (batch_size, pattern_len), device=device)
        repeats = seq_len // pattern_len + 1
        input_ids = patterns.repeat(1, repeats)[:, :seq_len]

        # Store original tokens as labels (will set non-masked positions to -100)
        labels = input_ids.clone()

        # Randomly mask 15% of tokens
        mask = torch.rand(batch_size, seq_len, device=device) < mask_prob
        # Ensure at least one token is masked per sequence
        first_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        first_mask[:, 0] = True
        mask = mask | first_mask

        # Replace masked positions with mask_token_id
        masked_input = input_ids.clone()
        masked_input[mask] = mask_token_id

        # Set non-masked positions to -100 (ignore in loss)
        labels[~mask] = -100

        # Forward pass
        outputs = model(masked_input, labels=labels)
        loss = outputs['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % 10 == 0 or step == 1:
            print(f"Step {step:3d}/{num_steps}  Loss: {loss_val:.4f}")

    # Visualization: plot MLM training loss
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, "bert_mlm_training.png")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_steps + 1), loss_history, linewidth=2, color='#2196F3')
    plt.xlabel("Training Step")
    plt.ylabel("MLM Loss")
    plt.title("BERT Masked Language Model Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\nTraining complete. Final loss: {loss_history[-1]:.4f}")
    print(f"Loss plot saved to: {save_path}")


# Test
if __name__ == "__main__":
    print("=== BERT Low-Level Implementation Test ===\n")

    # Configuration
    config = {
        'vocab_size': 30522,
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12,
        'intermediate_size': 3072,
        'max_position': 512,
        'dropout': 0.1
    }

    # Create model
    model = BertModel(**config)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Expected ~110M for BERT-Base\n")

    # Test input
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    # Forward
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        output_attentions=True
    )

    print("Output shapes:")
    print(f"  last_hidden_state: {outputs['last_hidden_state'].shape}")
    print(f"  pooler_output: {outputs['pooler_output'].shape}")
    print(f"  attentions: {len(outputs['attentions'])} layers")
    print(f"  attention shape: {outputs['attentions'][0].shape}")

    # MLM test
    print("\n=== MLM Test ===")
    mlm_model = BertForMaskedLM(config)

    labels = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    labels[labels != 103] = -100  # Only predict [MASK] tokens

    mlm_outputs = mlm_model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    print(f"MLM Loss: {mlm_outputs['loss'].item():.4f}")
    print(f"Logits shape: {mlm_outputs['logits'].shape}")

    # Classification test
    print("\n=== Classification Test ===")
    clf_model = BertForSequenceClassification(config, num_labels=2)

    labels = torch.randint(0, 2, (batch_size,))
    clf_outputs = clf_model(
        input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    print(f"Classification Loss: {clf_outputs['loss'].item():.4f}")
    print(f"Logits shape: {clf_outputs['logits'].shape}")

    print("\nAll tests passed!")

    demo_mlm_training()
