# Lesson 13 — LeNet and AlexNet (per-lesson exercise)

Prerequisites: L08 (convolution), L10 (pooling), L11 (batch norm).

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

LeNet-5 (1998) and AlexNet (2012) bracket the period when convnets went from a curiosity to the dominant computer-vision paradigm. Implementing both teaches every standard CNN building block in roughly the same sequence the field discovered them.

---

## Exercise 13.1 — LeNet-5 Forward Pass

**Difficulty**: ★★★

### Problem

LeNet-5 architecture for MNIST (28×28 grayscale digit classification):

| Layer | Type | Output shape | Notes |
|-------|------|--------------|-------|
| input | — | 1 × 28 × 28 | grayscale image |
| conv1 | conv 5×5, 6 filters | 6 × 24 × 24 | |
| pool1 | avg-pool 2×2 | 6 × 12 × 12 | |
| conv2 | conv 5×5, 16 filters | 16 × 8 × 8 | |
| pool2 | avg-pool 2×2 | 16 × 4 × 4 | |
| flatten | — | 256 | |
| fc1 | linear 256→120 | 120 | tanh activation |
| fc2 | linear 120→84 | 84 | tanh activation |
| fc3 | linear 84→10 | 10 | softmax (logits) |

Implement `lenet_forward(input, weights, output)` that runs all eight layers. Reuse:

- `conv2d_direct` from L08 exercise
- `avg_pool_2d` (write following the same pattern as L10's max-pool)
- `gemv` for fully-connected layers
- `tanh` from `<math.h>`
- `softmax` (numerically stable; see DL_Scratch_C L34)

### Verification

Initialize all weights to small random values; pass a random 28×28 input; verify the output is a 10-vector summing to 1. Compute the parameter count by hand:

- conv1: $5 \cdot 5 \cdot 1 \cdot 6 + 6 = 156$
- conv2: $5 \cdot 5 \cdot 6 \cdot 16 + 16 = 2416$
- fc1: $256 \cdot 120 + 120 = 30,840$
- fc2: $120 \cdot 84 + 84 = 10,164$
- fc3: $84 \cdot 10 + 10 = 850$
- Total: ~44,400 parameters

This is tiny by modern standards — the full model fits in 200 KB at fp32.

---

## Exercise 13.2 — AlexNet Sketch

**Difficulty**: ★★

AlexNet (2012, ImageNet winner) scaled the same idea up:

- Input: 3 × 224 × 224 RGB
- 5 convolutions: 96, 256, 384, 384, 256 filters with kernel sizes {11, 5, 3, 3, 3}
- 3 max-pools (2×2 stride 2)
- 3 fully-connected layers: 4096, 4096, 1000 (ImageNet has 1000 classes)
- ReLU activation throughout (replacing tanh)
- Dropout 0.5 on FC layers (training only)
- Total parameters: ~62 million (dominated by FC layers)

Without implementing the whole network, **count the FLOPs and parameters** for each layer. Where does the cost concentrate?

You should find:
- Convolutions account for ~95% of total FLOPs (roughly $5.6 \times 10^8$).
- Fully-connected layers account for ~95% of total parameters.

This explains why modern architectures (ResNet, ViT) keep convs but replace huge FC layers with global average pooling — same accuracy, 10× fewer parameters.

---

## Exercise 13.3 — Pre-trained Inference — Bonus

**Difficulty**: ★★★

Download a pre-trained LeNet checkpoint (any MNIST CNN with this architecture, e.g., from a Keras model exported to raw arrays). Load the weights into your C implementation and verify that the predictions match the Python framework's output for the first 10 test images.

The most common bug: weight layout differences. PyTorch stores conv weights as `[C_out, C_in, kH, kW]`; some frameworks use `[kH, kW, C_in, C_out]`. Your `conv2d_direct` from L08 expects the former — re-shuffle the imported weights if they came from the latter.
