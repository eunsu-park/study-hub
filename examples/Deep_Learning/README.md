# Deep_Learning Examples

Runnable example code corresponding to the lessons in the Deep_Learning folder.

## Folder Structure

```
examples/
├── pytorch/                      # PyTorch implementations
│   ├── 01_tensor_autograd.py     # Tensors, automatic differentiation
│   ├── 02_neural_network.py      # MLP, XOR problem
│   ├── 03_backprop.py            # Backpropagation visualization
│   ├── 04_training.py            # Training loop, optimizers
│   ├── 05_cnn_basic.py           # CNN basics
│   ├── 06_cnn_advanced.py        # ResNet, VGG
│   ├── 07_transfer_learning.py   # Transfer learning
│   ├── 08_rnn_basic.py           # RNN
│   ├── 09_lstm_gru.py            # LSTM, GRU
│   ├── 10_transformer.py         # Transformer
│   └── ...
│
└── numpy/                        # Pure NumPy implementations
    ├── 01_tensor_basics.py       # Tensors, manual differentiation
    ├── 02_neural_network_scratch.py  # MLP forward pass
    ├── 03_backprop_scratch.py    # Backpropagation from scratch
    ├── 04_training_scratch.py    # SGD from scratch
    └── 05_conv2d_scratch.py      # Convolution from scratch
```

## PyTorch vs NumPy Implementation Comparison

| Lesson | PyTorch | NumPy | Comparison Point |
|--------|---------|-------|-----------------|
| 01 | Automatic diff | Manual diff | `backward()` vs manual computation |
| 02 | `nn.Module` | Custom class | Forward pass structure |
| 03 | `loss.backward()` | Chain rule impl | Backpropagation principles |
| 04 | `optim.Adam` | SGD from scratch | Optimizer principles |
| 05 | `nn.Conv2d` | for loop | Convolution operation |
| 06+ | PyTorch only | - | Omitted due to complexity |

## How to Run

### Environment Setup

```bash
# Create virtual environment
python -m venv dl-env
source dl-env/bin/activate

# Install PyTorch (with CUDA support)
pip install torch torchvision torchaudio

# Other packages
pip install numpy matplotlib
```

### Execution

```bash
# PyTorch examples
cd Deep_Learning/examples/pytorch
python 01_tensor_autograd.py

# NumPy examples
cd Deep_Learning/examples/numpy
python 01_tensor_basics.py
```

## Learning Path

### Stage 1: Fundamentals (PyTorch + NumPy Comparison)
```
pytorch/01 <-> numpy/01  # Tensors, differentiation
pytorch/02 <-> numpy/02  # Neural network forward pass
pytorch/03 <-> numpy/03  # Backpropagation
pytorch/04 <-> numpy/04  # Training
```

### Stage 2: CNN (NumPy for basics only)
```
pytorch/05 <-> numpy/05  # CNN basics (understanding convolution)
pytorch/06              # Advanced CNN (PyTorch only)
pytorch/07              # Transfer learning (PyTorch only)
```

### Stage 3: Sequence Models (PyTorch only)
```
pytorch/08  # RNN
pytorch/09  # LSTM, GRU
pytorch/10  # Transformer
```

## Learning Value of NumPy Implementations

1. **01-02**: Understand that tensor operations and forward pass are simple matrix multiplications
2. **03**: Understand that backpropagation is repeated application of the chain rule
3. **04**: Understand the weight update principle of gradient descent
4. **05**: Understand that convolution is a sliding window of filters

## When NumPy Implementation Becomes Difficult

- **Advanced CNN**: Skip Connections, Batch Normalization
- **RNN/LSTM**: Backpropagation Through Time (BPTT), gate structures
- **Transformer**: Multi-Head Attention, positional encoding

> From this point on, focus on practical applications using PyTorch only

## References

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [CS231n (Stanford CNN)](http://cs231n.stanford.edu/)
- [3Blue1Brown Neural Networks](https://www.3blue1brown.com/topics/neural-networks)
