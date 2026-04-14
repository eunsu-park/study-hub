# Neural Network Fundamentals Learning Guide

## Overview

Neural networks are the computational backbone of modern deep learning. This topic bridges the gap between classical machine learning and deep learning by building a thorough understanding of how neural networks work from the ground up. Starting from biological inspiration and the perceptron, we progress through activation functions, feedforward architectures, backpropagation, optimization, and regularization -- culminating in a complete MLP implementation using only NumPy.

**Prerequisites**: [Machine Learning](../Machine_Learning/00_Overview.md), [Linear Algebra](../Linear_Algebra/00_Overview.md)

---

## Learning Roadmap

```
Biological Neurons → Perceptron → Activation Functions → Feedforward Networks
                                                              ↓
Universal Approximation ← Batch Normalization ← Regularization ← Weight Init
        ↓                                                         ↑
Training Pipeline → MLP from Scratch              Loss Functions → Backpropagation
        ↓                                                         → Gradient Descent ──┘
From Fundamentals to Deep Learning
```

---

## File List

| File | Topic | Key Content |
|------|-------|-------------|
| [01_Biological_to_Artificial_Neurons.md](./01_Biological_to_Artificial_Neurons.md) | Biological to Artificial Neurons | McCulloch-Pitts model, historical timeline, neuron anatomy |
| [02_Perceptron_and_Linear_Classifiers.md](./02_Perceptron_and_Linear_Classifiers.md) | Perceptron and Linear Classifiers | Perceptron learning rule, convergence theorem, XOR problem |
| [03_Activation_Functions.md](./03_Activation_Functions.md) | Activation Functions | Sigmoid, Tanh, ReLU, Leaky ReLU, GELU, Softmax, selection guide |
| [04_Feedforward_Networks.md](./04_Feedforward_Networks.md) | Feedforward Networks | MLP architecture, matrix formulation, forward pass |
| [05_Loss_Functions.md](./05_Loss_Functions.md) | Loss Functions | MSE, Cross-Entropy, Hinge loss, selection guide |
| [06_Backpropagation.md](./06_Backpropagation.md) | Backpropagation | Chain rule, computational graphs, gradient derivation |
| [07_Gradient_Descent_Variants.md](./07_Gradient_Descent_Variants.md) | Gradient Descent Variants | SGD, Momentum, RMSProp, Adam, learning rate scheduling |
| [08_Weight_Initialization.md](./08_Weight_Initialization.md) | Weight Initialization | Xavier/Glorot, He/Kaiming, symmetry breaking |
| [09_Regularization.md](./09_Regularization.md) | Regularization | L1/L2, Dropout, Early Stopping, data augmentation |
| [10_Batch_Normalization.md](./10_Batch_Normalization.md) | Batch Normalization | Internal covariate shift, BN algorithm, inference mode |
| [11_Universal_Approximation.md](./11_Universal_Approximation.md) | Universal Approximation Theorem | Theory, visualization, practical limitations |
| [12_Training_Pipeline.md](./12_Training_Pipeline.md) | Training Pipeline | Data splitting, validation, hyperparameter tuning |
| [13_Building_MLP_from_Scratch.md](./13_Building_MLP_from_Scratch.md) | Building MLP from Scratch | Complete MLP in NumPy, modular layer design |
| [14_From_Fundamentals_to_Deep_Learning.md](./14_From_Fundamentals_to_Deep_Learning.md) | From Fundamentals to Deep Learning | CNN, RNN, Transformer preview, next steps |

---

## Environment Setup

### Install Required Libraries

```bash
pip install numpy matplotlib
```

### Version Check

```python
import numpy as np
import matplotlib

print(f"NumPy: {np.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
```

### Recommended Versions
- Python: 3.9+
- NumPy: 1.24+
- Matplotlib: 3.7+

---

## Recommended Learning Order

### Stage 1: Foundations (01-03)
- Understand biological inspiration for neural networks
- Master the perceptron and its limitations
- Learn activation functions and their properties

### Stage 2: Architecture and Training (04-07)
- Build feedforward network understanding
- Master loss functions and backpropagation
- Learn gradient descent optimization variants

### Stage 3: Training Best Practices (08-10)
- Proper weight initialization strategies
- Regularization techniques to prevent overfitting
- Batch normalization for stable training

### Stage 4: Theory and Practice (11-13)
- Understand the universal approximation theorem
- Build a complete training pipeline
- Implement a full MLP from scratch

### Stage 5: Bridge to Deep Learning (14)
- Preview of CNN, RNN, and Transformer architectures
- Guidance on next learning steps

---

## Where This Topic Fits

```
Machine Learning (Tier 2)
    │
    ├── Classical ML algorithms (sklearn)
    │
    ▼
Neural Network Fundamentals (Tier 2)  ◄── YOU ARE HERE
    │
    ├── How neural networks work from scratch
    ├── Backpropagation and optimization
    ├── NumPy-only implementations
    │
    ▼
Deep Learning (Tier 3)
    │
    ├── CNN, RNN, Transformer (PyTorch)
    ├── Advanced architectures
    └── GPU training at scale
```

---

## References

### Textbooks
- "Neural Networks and Deep Learning" - Michael Nielsen (free online)
- "Deep Learning" - Goodfellow, Bengio, Courville (the "DL Bible")
- "Pattern Recognition and Machine Learning" - Christopher Bishop

### Online Resources
- [3Blue1Brown Neural Network Series](https://www.3blue1brown.com/topics/neural-networks)
- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.stanford.edu/)
- [Michael Nielsen's Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
