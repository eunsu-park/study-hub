# 01. Why C/C++ for Deep Learning?

**Next**: [Memory Layout and Strides](./02_Memory_Layout_and_Strides.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why production LLM inference engines are written in C/C++
2. Identify the key limitations of Python for systems-level ML work
3. Navigate the llama.cpp and llm.c source trees
4. Set up a C build environment for deep learning experiments
5. Write and compile a minimal tensor "hello world" in C

---

## 1. The Python Gap

Python's dominance in deep learning research is well-earned: fast iteration, expressive NumPy/PyTorch APIs, rich ecosystem. But production inference is a different problem.

| Concern | Python | C/C++ |
|---------|--------|-------|
| Memory layout control | Limited (NumPy strides help, but GC overhead) | Full — struct-level layout, manual allocation |
| SIMD vectorization | Via compiler hints; unreliable | Direct AVX2/AVX-512 intrinsics |
| Allocation latency | GC pauses; `malloc` hidden inside torch | Arena allocator — zero allocation after init |
| Binary distribution | Requires Python runtime, pip environment | Single static binary, WASM target |
| Latency (small batch) | 1–10 ms framework overhead per call | Microsecond dispatch |
| Quantized inference | Limited (bitsandbytes, GPTQ wrappers) | Native INT4/INT8 with custom `dp4a` kernels |

For **batch inference at scale**, Python + PyTorch is fine — GPU utilization dominates. For **single-request latency**, **edge deployment**, **embedded systems**, or **WASM browser inference**, C is the right tool.

---

## 2. The C/C++ LLM Ecosystem

Several influential projects show what is possible in pure C/C++:

### llama.cpp
The flagship C++ LLM inference library. Supports GGUF model format, CPU + CUDA + Metal backends, and runs on everything from a Raspberry Pi to an H100 cluster.

```
llama.cpp/
├── ggml.c          ← Tensor library (the heart)
├── ggml-alloc.c    ← Memory allocator
├── llama.cpp       ← Model loading + inference loop
├── common/         ← Tokenization, sampling utilities
└── examples/       ← CLI, server, embedding tools
```

Key insight: `ggml.c` is ~17,000 lines of C implementing tensors, autograd, quantized matmul, and multi-threaded scheduling. This course builds a simplified version from scratch.

### llm.c (Karpathy)
A 1,000-line C file that trains GPT-2 from scratch. Pedagogically pure: no dependencies beyond standard C and optionally cuBLAS. This course reproduces its training loop in Block 7.

```c
// The heart of llm.c: single training step
encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C);
layernorm_forward(acts.ln1, acts.ln1_mean, acts.ln1_rstd, acts.encoded, ...);
matmul_forward(acts.qkv, acts.ln1, params.qkvw, params.qkvb, B, T, C, 3*C);
attention_forward(acts.attn, acts.preatt, acts.att, acts.qkv, B, T, NH, HS);
// ... 10 more layers ...
crossentropy_forward(model->mean_loss, acts.losses, acts.probs, targets, B, T, Vp);
```

### whisper.cpp
Real-time speech recognition in C++. Shows how the same C tensor primitives extend to encoder-only Transformer architectures with mel spectrogram preprocessing.

### ggml / GGUF
`ggml` is the underlying tensor library. `GGUF` (GGML Unified Format) is the model file format used by the entire llama.cpp ecosystem — a binary container for quantized weights and metadata. We load GGUF files in L43.

---

## 3. What This Course Builds

By the end of this course, you will have written:

```
dl_scratch_c/
├── tensor/
│   ├── tensor.h / tensor.c      ← Tensor struct, strides, views
│   ├── ops.c                    ← Element-wise, matmul, reduction
│   ├── simd_matmul.c            ← AVX2 SGEMM (L04)
│   └── autograd.c               ← Backward pass engine (L05–L06)
├── memory/
│   └── arena.c                  ← Arena allocator (L07)
├── cnn/
│   ├── conv2d.c                 ← Convolution forward + backward (L08–L09)
│   ├── batchnorm.c              ← Batch normalization (L11)
│   └── models/
│       ├── lenet.c              ← LeNet-5 (L13)
│       └── resnet.c             ← ResNet-20 (L16)
├── transformer/
│   ├── tokenizer.c              ← BPE tokenizer (L21)
│   ├── attention.c              ← MHA + KV cache (L25–L26)
│   ├── gpt2.c                   ← GPT-2 forward pass (L29)
│   └── llama.c                  ← Llama 2/3 (RoPE, GQA, SwiGLU) (L30)
├── training/
│   ├── adamw.c                  ← AdamW optimizer (L35)
│   ├── dataloader.c             ← mmap data loader (L36)
│   └── train_gpt2.c             ← Full training loop (L38)
└── inference/
    ├── quantize.c               ← INT8/INT4 quantization (L40)
    ├── flash_attn_cpu.c         ← FlashAttention-2 CPU (L41)
    ├── gguf_reader.c            ← GGUF file parser (L43)
    └── inference_engine.c       ← Final CLI engine (L45)
```

---

## 4. C vs C++ in This Course

We use **C11 for the core tensor library** and **C++17 for model code** that benefits from templates and RAII.

| Component | Language | Reason |
|-----------|----------|--------|
| `tensor.c`, `ops.c` | C11 | Maximum portability, direct SIMD |
| `autograd.c`, `arena.c` | C11 | Explicit memory model |
| `attention.cpp`, `gpt2.cpp` | C++17 | `std::vector`, templates for configs |
| `gguf_reader.cpp` | C++17 | `std::map`, struct constructors |
| Build system | `Makefile` | Simple, reproducible |

---

## 5. Hello Tensor — First Program

Let us write the minimal starting point: a flat float array with shape metadata.

```c
// hello_tensor.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DIMS 8

typedef struct {
    float   *data;
    size_t   shape[MAX_DIMS];
    size_t   strides[MAX_DIMS];
    int      ndim;
    size_t   numel;
} Tensor;

Tensor *tensor_alloc(int ndim, size_t *shape) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    t->ndim  = ndim;
    t->numel = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->numel   *= shape[i];
    }
    // Row-major (C-order) strides
    t->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--)
        t->strides[i] = t->strides[i + 1] * shape[i + 1];

    t->data = (float *)calloc(t->numel, sizeof(float));
    return t;
}

void tensor_free(Tensor *t) {
    free(t->data);
    free(t);
}

void tensor_print(const Tensor *t) {
    printf("Tensor [");
    for (int i = 0; i < t->ndim; i++)
        printf("%zu%s", t->shape[i], i < t->ndim - 1 ? " x " : "");
    printf("]  numel=%zu\n", t->numel);
    // Print first 8 values
    printf("  data: [");
    size_t show = t->numel < 8 ? t->numel : 8;
    for (size_t i = 0; i < show; i++)
        printf("%.4f%s", t->data[i], i < show - 1 ? ", " : "");
    if (t->numel > 8) printf(", ...");
    printf("]\n");
}

int main(void) {
    size_t shape[] = {2, 3};
    Tensor *t = tensor_alloc(2, shape);

    // Fill with 0, 1, 2, ...
    for (size_t i = 0; i < t->numel; i++)
        t->data[i] = (float)i;

    tensor_print(t);

    // Access element at [1][2] using strides
    size_t row = 1, col = 2;
    float val = t->data[row * t->strides[0] + col * t->strides[1]];
    printf("  t[1][2] = %.1f  (expected 5.0)\n", val);

    tensor_free(t);
    return 0;
}
```

**Build and run**:
```bash
gcc -std=c11 -O2 -Wall hello_tensor.c -o hello_tensor
./hello_tensor
```

**Expected output**:
```
Tensor [2 x 3]  numel=6
  data: [0.0000, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000]
  t[1][2] = 5.0  (expected 5.0)
```

This 50-line program is the seed of everything that follows. By L07, this struct will support automatic differentiation.

---

## 6. Build System Overview

Every lesson in this course uses a standalone `Makefile`:

```makefile
CC      = gcc
CFLAGS  = -std=c11 -O2 -march=native -Wall -Wextra
LDFLAGS = -lm

TARGET  = hello_tensor
SRCS    = hello_tensor.c

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
```

From Block 3 onward, we add OpenBLAS:
```makefile
CFLAGS  += $(shell pkg-config --cflags openblas)
LDFLAGS += $(shell pkg-config --libs   openblas)
```

---

## Key Takeaways

- Python's GC, object overhead, and lack of SIMD control make it unsuitable for latency-critical LLM inference
- llama.cpp, llm.c, and ggml prove that a small C codebase can match or beat production ML frameworks for inference
- The `Tensor` struct (data pointer + shape + strides) is the fundamental building block; we will extend it with autograd in L05
- Every lesson produces a working, testable program — never leave a broken state

---

**Next**: [02. Memory Layout and Strides](./02_Memory_Layout_and_Strides.md) — Dive into row-major layout, stride arithmetic, and how views enable zero-copy reshaping.
