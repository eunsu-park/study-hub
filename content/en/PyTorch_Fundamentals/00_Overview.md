# PyTorch Fundamentals Study Guide

## Overview

PyTorch is a Python-based scientific computing framework that serves as the dominant deep learning library in both research and industry. This topic provides a comprehensive, hands-on introduction to PyTorch's core abstractions -- tensors, autograd, modules, data loading, and training loops -- equipping learners with the practical fluency needed before tackling advanced deep learning architectures.

Unlike a deep learning theory course, this topic focuses on **PyTorch as a tool**: how it works internally, how to write idiomatic PyTorch code, and how to debug, profile, and deploy models efficiently.

---

## What You'll Learn

After completing this topic, you will be able to:

- Create and manipulate tensors on CPU and GPU with full understanding of dtype, device, and memory layout
- Use PyTorch's autograd engine to compute gradients and understand computational graphs
- Define neural network architectures using `nn.Module` with proper parameter management
- Build custom datasets and efficient data pipelines with `Dataset` and `DataLoader`
- Write clean training loops with validation, checkpointing, and logging
- Save, load, and export models for production deployment
- Debug common PyTorch errors (shape mismatches, gradient issues, device mismatches)
- Leverage the broader PyTorch ecosystem (torchvision, Lightning, HuggingFace)

## Prerequisites

- **Python_Advanced**: Classes, decorators, context managers, iterators, type hints
- **Neural_Network_Fundamentals**: Feedforward networks, backpropagation, loss functions, gradient descent
- **Linear Algebra basics**: Matrix multiplication, transpose, broadcasting concepts

## Learning Roadmap

```
                    ┌───────────────────────────────────────────────────────┐
                    │              PyTorch Fundamentals (14 Lessons)       │
                    └───────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  L01: Intro  │────▶│ L02: Tensors │────▶│ L03: Tensor  │────▶│ L04: Autograd│
  │              │     │              │     │   Operations │     │              │
  └──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                       │
                                                                       ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ L08: Train   │◀────│ L07: Dataset │◀────│ L06: Loss &  │◀────│ L05: nn      │
  │   Loop       │     │ & DataLoader │     │  Optimizers  │     │   Module     │
  └──────┬───────┘     └──────────────┘     └──────────────┘     └──────────────┘
         │
         ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ L09: Save &  │────▶│ L10: GPU     │────▶│ L11: Debug   │────▶│ L12: Custom  │
  │   Load       │     │  Training    │     │              │     │  Layers      │
  └──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                       │
                                                                       ▼
                                            ┌──────────────┐     ┌──────────────┐
                                            │ L14: PyTorch │◀────│ L13: Torch   │
                                            │  Ecosystem   │     │  Script      │
                                            └──────────────┘     └──────────────┘
```

**Recommended Path**: Follow lessons sequentially (L01 through L14). Lessons build on each other -- L01-L04 cover foundations, L05-L08 cover the model-building pipeline, and L09-L14 cover production and advanced topics.

---

## File List

| Lesson | Filename | Description |
|--------|----------|-------------|
| L01 | `01_Introduction_to_PyTorch.md` | History, ecosystem, installation, first tensor |
| L02 | `02_Tensors.md` | Creation, attributes, dtype, device, view vs copy |
| L03 | `03_Tensor_Operations.md` | Indexing, slicing, broadcasting, matrix operations |
| L04 | `04_Autograd.md` | requires_grad, backward(), grad, computational graph |
| L05 | `05_nn_Module.md` | Module definition, forward, parameters(), nesting |
| L06 | `06_Loss_Functions_and_Optimizers.md` | CrossEntropyLoss, Adam, SGD, learning rate scheduling |
| L07 | `07_Dataset_and_DataLoader.md` | Dataset, DataLoader, transforms, custom datasets |
| L08 | `08_Training_Loop.md` | Train/eval mode, epochs, batches, validation |
| L09 | `09_Model_Saving_and_Loading.md` | state_dict, checkpoint, ONNX export |
| L10 | `10_GPU_Training.md` | .to(device), DataParallel, mixed precision |
| L11 | `11_Debugging_PyTorch.md` | Shape errors, gradient checking, hooks |
| L12 | `12_Custom_Layers_and_Functions.md` | autograd.Function, custom backward |
| L13 | `13_TorchScript_and_Deployment.md` | Tracing, scripting, mobile deployment |
| L14 | `14_PyTorch_Ecosystem.md` | torchvision, torchaudio, Lightning, HuggingFace |

**Total: 14 lessons**

---

## Environment Setup

### Installation

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# For GPU support (CUDA 12.1 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Recommended Tools

- **IDE**: VS Code with Python and Pylance extensions
- **Debugger**: Python debugger (`breakpoint()`) or VS Code debugger
- **GPU**: NVIDIA GPU recommended for L10; Google Colab free tier works for most lessons

---

## Related Materials

- **[Python_Advanced](../Python_Advanced/00_Overview.md)**: Advanced Python features used throughout PyTorch
- **[Deep_Learning](../Deep_Learning/00_Overview.md)**: Builds on PyTorch fundamentals to implement CNNs, Transformers, GANs
- **[Machine_Learning](../Machine_Learning/00_Overview.md)**: Classical ML concepts (loss, regularization, evaluation)
- **[CUDA](../CUDA/00_Overview.md)**: GPU programming fundamentals for understanding PyTorch's GPU backend

---

## Study Tips

1. **Type every example**: Do not copy-paste. Typing builds familiarity with the API.
2. **Inspect shapes obsessively**: Print `.shape` after every operation until it becomes second nature.
3. **Read error messages carefully**: PyTorch error messages are informative -- they tell you expected vs actual shapes, dtypes, and devices.
4. **Use small tensors for debugging**: Create 2x3 or 3x4 tensors you can verify by hand.
5. **Check the official docs**: The PyTorch documentation is excellent -- make it your primary reference.

---

**Start with [01_Introduction_to_PyTorch.md](./01_Introduction_to_PyTorch.md) and begin your PyTorch journey.**
