# Deep Learning Study Guide

## Introduction

This folder provides a comprehensive guide to **deep learning theory and practice**, combining conceptual lessons with from-scratch model implementations. PyTorch is the primary framework, and the curriculum follows a pedagogical approach that integrates four levels of learning:

1. **Level 1 (NumPy Scratch)**: Build models from raw NumPy to understand fundamental mechanics
2. **Level 2 (PyTorch Low-Level)**: Implement using PyTorch primitives (tensors, autograd) without high-level APIs
3. **Level 3 (Paper Reproduction)**: Read original papers and reproduce key architectures
4. **Level 4 (Code Analysis)**: Analyze production-quality implementations from frameworks and research repos

This merged approach ensures you understand both the "why" and the "how" of deep learning, from mathematical foundations to practical deployment.

## Target Audience

- Learners who have completed the **Machine_Learning** folder
- Readers comfortable with Python, NumPy, and basic ML concepts (gradient descent, overfitting, train/test splits)
- Anyone seeking a rigorous, implementation-focused deep learning education

## Learning Roadmap

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Foundations │────▶│     CNN      │────▶│   Sequence   │────▶│ Transformers │
│   L01-L06    │     │   L07-L12    │     │   L13-L15    │     │   L16-L22    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                        │
                                                                        ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Practical  │◀────│   Advanced   │◀────│  Generative  │◀────│   Training   │
│   L39-L42    │     │   L34-L38    │     │   L28-L33    │     │   L23-L27    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

**Recommended Path**:
1. Start with Foundations (L01-L06) to master PyTorch basics and backpropagation
2. Progress through CNN (L07-L12) for computer vision fundamentals
3. Learn Sequence Models (L13-L15) for temporal data
4. Master Transformers (L16-L22), the backbone of modern NLP and vision
5. Study Training Essentials (L23-L27) for optimization, loss functions, and normalization
6. Explore Generative Models (L28-L33) for GANs, VAEs, and Diffusion
7. Dive into Advanced topics (L34-L38) for multimodal learning and modern architectures
8. Apply knowledge with Practical projects (L39-L42)

## File List

| Lesson | Filename | Difficulty | Description |
|--------|----------|------------|-------------|
| **Block 1: Foundations** |
| L01 | `01_Tensors_and_Autograd.md` | ⭐ | Tensor operations, autograd, computational graphs |
| L02 | `02_Neural_Network_Basics.md` | ⭐⭐ | Activation functions, loss, forward/backward pass |
| L03 | `03_Backpropagation.md` | ⭐⭐ | Chain rule, gradient flow, vanishing/exploding gradients |
| L04 | `04_Training_Techniques.md` | ⭐⭐ | Regularization, dropout, batch normalization, early stopping |
| L05 | `05_Impl_Linear_Logistic.md` | ⭐⭐ | **Implementation**: Linear & Logistic regression from scratch |
| L06 | `06_Impl_MLP.md` | ⭐⭐ | **Implementation**: Multilayer Perceptron (NumPy → PyTorch) |
| **Block 2: Convolutional Neural Networks** |
| L07 | `07_CNN_Basics.md` | ⭐⭐ | Convolution, pooling, feature maps, LeNet |
| L08 | `08_CNN_Advanced.md` | ⭐⭐⭐ | ResNet, Inception, skip connections, 1x1 convolutions |
| L09 | `09_Transfer_Learning.md` | ⭐⭐⭐ | Pretrained models, fine-tuning, domain adaptation |
| L10 | `10_Impl_CNN_LeNet.md` | ⭐⭐⭐ | **Implementation**: LeNet-5 on MNIST |
| L11 | `11_Impl_VGG.md` | ⭐⭐⭐ | **Implementation**: VGG-16 architecture |
| L12 | `12_Impl_ResNet.md` | ⭐⭐⭐ | **Implementation**: Residual Networks (ResNet-18/34) |
| **Block 3: Sequence Models** |
| L13 | `13_RNN_Basics.md` | ⭐⭐⭐ | Recurrent networks, hidden states, sequence-to-sequence |
| L14 | `14_LSTM_GRU.md` | ⭐⭐⭐ | Long Short-Term Memory, Gated Recurrent Units |
| L15 | `15_Impl_LSTM_GRU.md` | ⭐⭐⭐ | **Implementation**: LSTM/GRU for text classification |
| **Block 4: Attention and Transformers** |
| L16 | `16_Attention_Transformer.md` | ⭐⭐⭐⭐ | Self-attention, multi-head attention, Transformer architecture |
| L17 | `17_Attention_Deep_Dive.md` | ⭐⭐⭐⭐ | Query/Key/Value, scaled dot-product, positional encoding |
| L18 | `18_Impl_Transformer.md` | ⭐⭐⭐⭐ | **Implementation**: Transformer encoder/decoder from scratch |
| L19 | `19_Impl_BERT.md` | ⭐⭐⭐⭐ | **Implementation**: BERT pretraining (masked LM) |
| L20 | `20_Impl_GPT.md` | ⭐⭐⭐⭐ | **Implementation**: GPT-style autoregressive model |
| L21 | `21_Vision_Transformer.md` | ⭐⭐⭐⭐ | Patch embeddings, ViT architecture, DeiT |
| L22 | `22_Impl_ViT.md` | ⭐⭐⭐⭐ | **Implementation**: Vision Transformer for image classification |
| **Block 5: Training Essentials** |
| L23 | `23_Training_Optimization.md` | ⭐⭐⭐ | Learning rate schedules, gradient clipping, mixed precision |
| L24 | `24_Loss_Functions.md` | ⭐⭐⭐ | Cross-entropy, focal loss, contrastive loss, triplet loss |
| L25 | `25_Optimizers.md` | ⭐⭐⭐ | SGD, Adam, AdamW, learning rate warmup |
| L26 | `26_Normalization_Layers.md` | ⭐⭐⭐ | Batch norm, layer norm, group norm, instance norm |
| L27 | `27_TensorBoard.md` | ⭐⭐ | Logging, visualization, hyperparameter tracking |
| **Block 6: Generative Models** |
| L28 | `28_Generative_Models_GAN.md` | ⭐⭐⭐ | Generator, discriminator, adversarial training |
| L29 | `29_Impl_GAN.md` | ⭐⭐⭐ | **Implementation**: DCGAN for image generation |
| L30 | `30_Generative_Models_VAE.md` | ⭐⭐⭐ | Variational autoencoders, latent space, reparameterization |
| L31 | `31_Impl_VAE.md` | ⭐⭐⭐ | **Implementation**: VAE on MNIST/CIFAR-10 |
| L32 | `32_Diffusion_Models.md` | ⭐⭐⭐⭐ | DDPM, score-based models, denoising process |
| L33 | `33_Impl_Diffusion.md` | ⭐⭐⭐⭐ | **Implementation**: Denoising Diffusion Probabilistic Model |
| **Block 7: Multimodal and Advanced Topics** |
| L34 | `34_CLIP_Multimodal.md` | ⭐⭐⭐⭐ | Contrastive learning, vision-language models |
| L35 | `35_Impl_CLIP.md` | ⭐⭐⭐⭐ | **Implementation**: CLIP-style image-text alignment |
| L36 | `36_Self_Supervised_Learning.md` | ⭐⭐⭐⭐ | SimCLR, MoCo, BYOL, contrastive pretraining |
| L37 | `37_Modern_Architectures.md` | ⭐⭐⭐⭐ | EfficientNet, ConvNeXt, Swin Transformer, NFNet |
| L38 | `38_Object_Detection.md` | ⭐⭐⭐⭐ | RCNN, YOLO, RetinaNet, DETR, anchor-free methods |
| **Block 8: Practical and Deployment** |
| L39 | `39_Practical_Image_Classification.md` | ⭐⭐⭐⭐ | End-to-end project: dataset, training, evaluation, deployment |
| L40 | `40_Practical_Text_Classification.md` | ⭐⭐⭐⭐ | End-to-end NLP project: tokenization, fine-tuning, inference |
| L41 | `41_Model_Saving_Deployment.md` | ⭐⭐⭐ | ONNX export, TorchScript, model serving (Flask, TorchServe) |
| L42 | `42_Reinforcement_Learning_Intro.md` | ⭐⭐⭐ | DQN basics, policy gradients, bridge to RL topic |
| L43 | `43_Few_Shot_Learning.md` | ⭐⭐⭐⭐ | Meta-learning, prototypical networks, MAML |
| L44 | `44_Test_Time_Adaptation.md` | ⭐⭐⭐⭐ | BN adaptation, TENT, CoTTA, distribution shift |

**Total: 44 lessons** (30 concept lessons + 14 implementation lessons)

## Implementation Philosophy: The 4-Level Approach

This curriculum integrates theory with hands-on coding through a **4-level progression**:

| Level | Description | Tools | Example Lessons |
|-------|-------------|-------|-----------------|
| **L1: NumPy Scratch** | Build models using only NumPy (no PyTorch `nn.Module`). Implement forward/backward passes manually. | NumPy arrays, manual gradient computation | L05, L06 |
| **L2: PyTorch Low-Level** | Use PyTorch tensors and autograd, but avoid `nn.Linear`, `nn.Conv2d`. Define custom modules. | `torch.Tensor`, `autograd`, custom `nn.Module` | L10, L11, L15 |
| **L3: Paper Reproduction** | Read original papers (Attention Is All You Need, BERT, etc.) and reproduce architectures. | PyTorch, paper pseudocode | L18, L19, L20, L22 |
| **L4: Code Analysis** | Study production implementations (Hugging Face Transformers, torchvision models) and understand design patterns. | GitHub repos, library source code | L37, L38, L41 |

**Why This Approach?**
- **L1** ensures you understand the math (no "magic" libraries)
- **L2** teaches PyTorch idioms while retaining low-level control
- **L3** bridges academic papers to code
- **L4** prepares you for real-world ML engineering

## Prerequisites

- **Programming**: Proficiency in Python (functions, classes, list comprehensions)
- **Mathematics**: Linear algebra (matrix multiplication, eigenvalues), calculus (derivatives, chain rule), basic probability
- **Machine Learning**: Familiarity with supervised learning, loss functions, gradient descent (see `Machine_Learning` folder)
- **Libraries**: NumPy basics (array indexing, broadcasting)

## Environment Setup

### Installation
```bash
# Install PyTorch (CPU version)
pip install torch torchvision matplotlib numpy

# For GPU support (CUDA 11.8 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Optional: TensorBoard for visualization
pip install tensorboard
```

### Verify Installation
```python
import torch
print(torch.__version__)  # e.g., 2.1.0
print(torch.cuda.is_available())  # True if GPU available
```

### Recommended Tools
- **IDE**: VS Code with Python extension, Jupyter notebooks for experimentation
- **GPU**: NVIDIA GPU recommended for L28-L42 (Google Colab free tier works for most lessons)

## Related Materials

- **[Machine_Learning](../Machine_Learning/00_Overview.md)**: Prerequisite for understanding loss functions, regularization, evaluation metrics
- **[LLM_and_NLP](../LLM_and_NLP/00_Overview.md)**: Advanced NLP applications (BERT, GPT fine-tuning, LangChain)
- **[Foundation_Models](../Foundation_Models/00_Overview.md)**: Scaling laws, LoRA, quantization, RAG
- **[Computer_Vision](../Computer_Vision/00_Overview.md)**: Applied CV with OpenCV, object detection, SLAM
- **[Reinforcement_Learning](../Reinforcement_Learning/00_Overview.md)**: DQN, PPO, policy gradients (builds on L42)
- **[Math_for_AI](../Math_for_AI/00_Overview.md)**: Matrix calculus, optimization theory, probability

## Study Tips

1. **Don't Skip Implementations**: Typing out code (even if copying) builds muscle memory. Resist the urge to only read.
2. **Experiment Liberally**: Change hyperparameters, swap activation functions, break code intentionally to see error messages.
3. **Read Papers Alongside Code**: For L18-L22, read the original papers. Notation in papers matches variable names in code.
4. **Debug with Small Data**: Test models on tiny datasets (10 samples) to catch bugs before full training.
5. **Visualize Activations**: Use TensorBoard (L27) to inspect gradients, weights, and feature maps.
6. **Join Communities**: PyTorch forums, r/MachineLearning, Papers with Code discussions.

## Learning Outcomes

After completing this folder, you will be able to:

- ✅ Implement neural networks from scratch using NumPy and PyTorch
- ✅ Explain backpropagation, gradient descent, and autograd internals
- ✅ Build and train CNNs for image classification (ResNet, VGG)
- ✅ Implement Transformers, BERT, and GPT from research papers
- ✅ Train generative models (GANs, VAEs, Diffusion Models)
- ✅ Apply transfer learning and fine-tuning to real-world datasets
- ✅ Optimize training with advanced techniques (mixed precision, gradient clipping, learning rate schedules)
- ✅ Deploy models using ONNX, TorchScript, and web frameworks
- ✅ Read and reproduce state-of-the-art deep learning papers

## Next Steps

- **For NLP**: Proceed to `LLM_and_NLP` for large language models, RAG, and prompt engineering
- **For Vision**: Explore `Computer_Vision` for OpenCV, 3D vision, and SLAM
- **For Efficiency**: Study `Foundation_Models` for quantization, LoRA, and model compression
- **For RL**: Advance to `Reinforcement_Learning` for DQN, PPO, and game agents
- **For Production**: Check `MLOps` for experiment tracking, model serving, and CI/CD

## Additional Resources

- **Official Docs**: [PyTorch Tutorials](https://pytorch.org/tutorials/), [PyTorch Documentation](https://pytorch.org/docs/)
- **Books**: *Deep Learning* (Goodfellow et al.), *Dive into Deep Learning* (d2l.ai)
- **Courses**: Stanford CS230, Fast.ai Practical Deep Learning
- **Papers**: [Papers with Code](https://paperswithcode.com/) for implementations and benchmarks

---

**Happy Learning!** Start with `01_Tensors_and_Autograd.md` and build your deep learning expertise step by step.
