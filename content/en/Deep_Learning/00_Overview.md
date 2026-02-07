# Deep Learning Study Guide

## Introduction

This folder contains materials for systematic deep learning study. PyTorch is used as the main framework, and fundamental concepts are also implemented in NumPy to provide a deep understanding of algorithm principles.

**Target Audience**: Learners who have completed the Machine_Learning folder

---

## Learning Roadmap

```
[Basics]                 [CNN]                    [Sequence]
   │                        │                         │
   ▼                        ▼                         ▼
Tensor/Autograd ──▶ CNN Basics ─────────▶ RNN Basics
   │                        │                         │
   ▼                        ▼                         ▼
Neural Network   ──▶ CNN Advanced ───────▶ LSTM/GRU
Basics                  (ResNet)                      │
   │                        │                         ▼
   ▼                        ▼                   Transformer
Backpropagation ──▶ Transfer Learning                │
Understanding                                         ▼
                                                 Attention
                [Practical]
                     │
                     ▼
              Training Optimization
                     │
                     ▼
              Model Saving/Deployment
                     │
                     ▼
              Practical Projects
```

---

## File List

### Basics (PyTorch + NumPy Comparison)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [01_Tensors_and_Autograd.md](./01_Tensors_and_Autograd.md) | ⭐ | Tensor, Automatic Differentiation, GPU | PyTorch + NumPy |
| [02_Neural_Network_Basics.md](./02_Neural_Network_Basics.md) | ⭐⭐ | Perceptron, MLP, Activation Functions | PyTorch + NumPy |
| [03_Backpropagation.md](./03_Backpropagation.md) | ⭐⭐ | Chain Rule, Gradient Descent, Loss Functions | PyTorch + NumPy |
| [04_Training_Techniques.md](./04_Training_Techniques.md) | ⭐⭐ | Optimizers, Batching, Regularization | PyTorch + NumPy |

### CNN (PyTorch-centric, Basics with NumPy Comparison)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [05_CNN_Basics.md](./05_CNN_Basics.md) | ⭐⭐ | Convolution, Pooling, Filters | PyTorch + NumPy |
| [06_CNN_Advanced.md](./06_CNN_Advanced.md) | ⭐⭐⭐ | ResNet, VGG, EfficientNet | PyTorch only |
| [07_Transfer_Learning.md](./07_Transfer_Learning.md) | ⭐⭐⭐ | Pretrained Models, Fine-tuning | PyTorch only |

### Sequence Models (PyTorch only)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [08_RNN_Basics.md](./08_RNN_Basics.md) | ⭐⭐⭐ | Recurrent Neural Networks, Vanishing Gradients | PyTorch only |
| [09_LSTM_GRU.md](./09_LSTM_GRU.md) | ⭐⭐⭐ | Long-term Dependencies, Gate Structures | PyTorch only |
| [10_Attention_Transformer.md](./10_Attention_Transformer.md) | ⭐⭐⭐⭐ | Attention, Transformer | PyTorch only |

### Practical (PyTorch only)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [11_Training_Optimization.md](./11_Training_Optimization.md) | ⭐⭐⭐ | AMP, Gradient Accumulation, Hyperparameters | PyTorch only |
| [12_Model_Saving_Deployment.md](./12_Model_Saving_Deployment.md) | ⭐⭐⭐ | TorchScript, ONNX, Quantization | PyTorch only |
| [13_Practical_Image_Classification.md](./13_Practical_Image_Classification.md) | ⭐⭐⭐⭐ | CIFAR-10, Data Augmentation, Mixup | PyTorch only |
| [14_Practical_Text_Classification.md](./14_Practical_Text_Classification.md) | ⭐⭐⭐⭐ | Sentiment Analysis, LSTM/Transformer | PyTorch only |

### Generative Models (PyTorch only)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [15_Generative_Models_GAN.md](./15_Generative_Models_GAN.md) | ⭐⭐⭐ | GAN, DCGAN, Loss Functions, StyleGAN | PyTorch only |
| [16_Generative_Models_VAE.md](./16_Generative_Models_VAE.md) | ⭐⭐⭐ | VAE, ELBO, Reparameterization, Beta-VAE | PyTorch only |
| [17_Diffusion_Models.md](./17_Diffusion_Models.md) | ⭐⭐⭐⭐ | DDPM, DDIM, U-Net, Stable Diffusion | PyTorch only |
| [18_Attention_Deep_Dive.md](./18_Attention_Deep_Dive.md) | ⭐⭐⭐⭐ | Flash Attention, Sparse Attention, RoPE, ALiBi | PyTorch only |

### Advanced Techniques (PyTorch only)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [19_Vision_Transformer.md](./19_Vision_Transformer.md) | ⭐⭐⭐⭐ | ViT Architecture, Patch Embedding, CLS Token, DeiT, Swin Transformer | PyTorch only |
| [20_CLIP_Multimodal.md](./20_CLIP_Multimodal.md) | ⭐⭐⭐⭐ | CLIP Principles, Zero-shot Classification, Image-Text Matching, BLIP | PyTorch only |
| [21_Self_Supervised_Learning.md](./21_Self_Supervised_Learning.md) | ⭐⭐⭐⭐ | SimCLR, MoCo, BYOL, MAE, Contrastive Learning | PyTorch only |
| [22_Reinforcement_Learning_Intro.md](./22_Reinforcement_Learning_Intro.md) | ⭐⭐⭐ | MDP Basics, Q-Learning Concepts, Policy Gradient Overview | PyTorch only |

### Object Detection/Segmentation (PyTorch only)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [23_Object_Detection.md](./23_Object_Detection.md) | ⭐⭐⭐⭐ | YOLO, Faster R-CNN, DETR, Mask R-CNN, SAM | PyTorch only |

### Visualization/Tools (PyTorch only)

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [24_TensorBoard.md](./24_TensorBoard.md) | ⭐⭐ | Training visualization, metric logging, hyperparameter tracking, profiling | PyTorch only |

---

## NumPy vs PyTorch Comparison Guide

| Topic | NumPy Implementation | PyTorch Implementation | Comparison Point |
|-------|---------------------|------------------------|------------------|
| Tensor | `np.array` | `torch.tensor` | GPU support, automatic differentiation |
| Forward Pass | Direct matrix operations | `nn.Module.forward` | Code conciseness |
| Backpropagation | Manual derivative computation | `loss.backward()` | Convenience of automatic differentiation |
| Convolution | for loop implementation | `nn.Conv2d` | Performance difference |
| Optimization | Direct SGD implementation | `optim.SGD` | Advanced optimizers |

### Advantages of NumPy Implementation
- Understanding the mathematical principles of algorithms
- Improving matrix operation intuition
- Grasping concepts without framework dependencies

### When NumPy Implementation Becomes Difficult
- Advanced CNN structures (Skip Connection, Batch Norm)
- RNN/LSTM (complex gate structures)
- Transformer (Multi-Head Attention)

---

## Prerequisites

- Python basic syntax
- NumPy array operations
- Content from Machine_Learning folder (regression, classification concepts)
- Basic calculus (partial derivatives, chain rule)
- Linear algebra basics (matrix multiplication)

---

## Environment Setup

### Required Packages

```bash
# PyTorch (with CUDA support)
pip install torch torchvision torchaudio

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Other requirements
pip install numpy matplotlib
```

### GPU Check

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## Recommended Learning Sequence

1. **Basics (1 week)**: 01 → 02 → 03 → 04
   - Run and compare both NumPy and PyTorch code
2. **CNN (1 week)**: 05 → 06 → 07
   - For 05, implementing convolution directly in NumPy is recommended
3. **Sequence (1-2 weeks)**: 08 → 09 → 10
   - Invest sufficient time in Transformer/Attention
4. **Practical (1 week)**: 11 → 12 → 13 → 14
   - Training optimization, model saving, project practice
5. **Generative Models (2 weeks)**: 15 → 16 → 17 → 18
   - Study in order: GAN/VAE/Diffusion
   - Advanced Attention is essential before LLM learning
6. **Advanced Techniques (1-2 weeks)**: 19 → 20 → 21 → 22
   - Study modern architectures like ViT, CLIP
   - For reinforcement learning basics, prior study of the Reinforcement_Learning folder is recommended

---

## Related Materials

- [Machine_Learning/](../Machine_Learning/00_Overview.md) - Prerequisite course
- [Python/](../Python/00_Overview.md) - Advanced Python
- [Data_Analysis/](../Data_Analysis/00_Overview.md) - NumPy/Pandas

---

## Reference Links

- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n (CNN)](http://cs231n.stanford.edu/)
- [CS224n (NLP)](http://web.stanford.edu/class/cs224n/)
