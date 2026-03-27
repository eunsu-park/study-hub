# DL_Scratch_C Examples Makefile
# Requires: gcc (C11), OpenBLAS or Accelerate framework, STB (header-only)
#
# Usage:
#   make           - build all examples
#   make clean     - remove build artifacts
#   make L08       - build Lesson 08 (convolution) example
#   make test      - run all tests

CC       = gcc
CFLAGS   = -Wall -Wextra -std=c11 -O2 -g
MATH     = -lm

# OpenBLAS (Linux/most systems)
BLAS     = -lopenblas
# Accelerate (macOS) — uncomment if on Mac:
# BLAS   = -framework Accelerate -DACCELERATE_BLAS

INCLUDES = -I./include
STB_IMPL = -DSTB_IMAGE_IMPLEMENTATION -DSTB_IMAGE_RESIZE_IMPLEMENTATION

# --- Lesson Examples ---

TARGETS = \
	08_convolution/conv2d_demo \
	09_conv_backward/conv_backward_demo \
	10_pooling/pooling_demo \
	11_batchnorm/batchnorm_demo \
	12_data_pipeline/cifar10_loader \
	13_lenet/lenet_demo \
	14_training/train_cifar10 \
	15_vgg/vgg_demo \
	16_resnet/resnet20_demo \
	17_depthwise/dwconv_demo \
	18_se_block/se_demo \
	19_efficientnet/efficientnet_demo \
	20_benchmark/cnn_benchmark \
	21_tokenization/bpe_demo \
	22_embedding/embed_demo \
	23_positional/rope_demo \
	24_layernorm/layernorm_demo \
	25_attention/mha_demo \
	26_kvcache/kvcache_demo \
	27_ffn/ffn_demo \
	28_transformer/block_demo \
	29_gpt2/gpt2_forward \
	30_llama/llama_forward \
	31_vit/vit_demo \
	34_loss/ce_loss_demo \
	35_optimizers/optimizer_demo \
	36_training/train_shakespeare \
	38_training_gpt2/train_gpt2 \
	39_sampling/sampling_demo \
	40_quantization/quant_demo \
	41_flashattn/flashattn_demo \
	43_gguf/gguf_reader \
	44_parallel/parallel_infer

.PHONY: all clean test

all: $(TARGETS)

# Lesson 08: 2D Convolution
08_convolution/conv2d_demo: 08_convolution/conv2d_demo.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(BLAS) $(MATH)

# Lesson 09: Convolution Backward
09_conv_backward/conv_backward_demo: 09_conv_backward/conv_backward_demo.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(BLAS) $(MATH)

# Lesson 10: Pooling
10_pooling/pooling_demo: 10_pooling/pooling_demo.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(MATH)

# Lesson 11: Batch Normalization
11_batchnorm/batchnorm_demo: 11_batchnorm/batchnorm_demo.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(MATH)

# Lesson 12: Data Pipeline (requires STB headers)
12_data_pipeline/cifar10_loader: 12_data_pipeline/cifar10_loader.c
	$(CC) $(CFLAGS) $(INCLUDES) $(STB_IMPL) $< -o $@ $(MATH)

# Lesson 13: LeNet-5 + AlexNet
13_lenet/lenet_demo: 13_lenet/lenet_demo.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(BLAS) $(MATH)

# Lesson 14: CIFAR-10 Training
14_training/train_cifar10: 14_training/train_cifar10.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(BLAS) $(MATH)

# Lesson 16: ResNet-20
16_resnet/resnet20_demo: 16_resnet/resnet20_demo.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(BLAS) $(MATH)

# Lesson 29: GPT-2 Forward (requires gpt2_124M.bin)
29_gpt2/gpt2_forward: 29_gpt2/gpt2_forward.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(BLAS) $(MATH)

# Lesson 36: Shakespeare training
36_training/train_shakespeare: 36_training/train_shakespeare.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(BLAS) $(MATH)

# Lesson 38: GPT-2 training (OpenMP for data loading)
38_training_gpt2/train_gpt2: 38_training_gpt2/train_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(BLAS) $(MATH) -fopenmp

# Lesson 44: Parallel inference (OpenMP)
44_parallel/parallel_infer: 44_parallel/parallel_infer.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(BLAS) $(MATH) -fopenmp

# Generic rule for remaining targets
%: %.c
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ $(BLAS) $(MATH)

clean:
	rm -f $(TARGETS)

# Quick test: run demos that don't need external files
test: 08_convolution/conv2d_demo 10_pooling/pooling_demo 11_batchnorm/batchnorm_demo 24_layernorm/layernorm_demo
	./08_convolution/conv2d_demo
	./10_pooling/pooling_demo
	./11_batchnorm/batchnorm_demo
	./24_layernorm/layernorm_demo
	@echo "All self-contained tests passed!"

# Target aliases for convenience
L08: 08_convolution/conv2d_demo
L09: 09_conv_backward/conv_backward_demo
L10: 10_pooling/pooling_demo
L11: 11_batchnorm/batchnorm_demo
L12: 12_data_pipeline/cifar10_loader
L13: 13_lenet/lenet_demo
L14: 14_training/train_cifar10
L16: 16_resnet/resnet20_demo
L29: 29_gpt2/gpt2_forward
L36: 36_training/train_shakespeare
L38: 38_training_gpt2/train_gpt2
