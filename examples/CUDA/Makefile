# CUDA Examples Makefile
# Requires: CUDA Toolkit (nvcc), CUB (bundled with CUDA 11+), NCCL (for L37)
#
# Usage:
#   make           - build all examples
#   make clean     - remove build artifacts
#   make L02       - build Lesson 02 (vector_add) example
#   make test      - run self-contained tests
#
# Note: Some lessons require specific hardware (e.g., L37 needs multi-GPU NVLink)
#       and external libraries (NCCL). Build individual lessons as needed.

NVCC     = nvcc
CFLAGS   = -O2 -arch=sm_80 -std=c++17 -Xcompiler -Wall
MATH     = -lm
CUBLAS   = -lcublas
CUSPARSE = -lcusparse
CUFFT    = -lcufft
CURAND   = -lcurand
NCCL     = -lnccl

INCLUDES = -I./include

# --- Lesson Examples ---

TARGETS = \
	02_programming_model/vector_add \
	03_thread_indexing/matrix_transpose \
	04_memory_model/bandwidth_bench \
	05_shared_memory/tiled_matmul \
	06_warp_execution/warp_reduce \
	07_atomics/histogram_atomic \
	08_coalescing/stride_bench \
	09_occupancy/occupancy_demo \
	10_roofline/roofline_bench \
	12_streams/double_buffer \
	13_cuda_graphs/graph_demo \
	14_reduction/parallel_reduce \
	15_scan/prefix_sum \
	16_sort/radix_sort \
	17_stencil/heat_stencil \
	18_histogram/privatized_hist \
	19_sparse/spmv_demo \
	20_nbody/nbody_sim \
	21_monte_carlo/pi_estimate \
	22_fft/conv_fft \
	23_pde/heat2d \
	24_lbm/lbm_d2q9 \
	25_md/lennard_jones \
	26_image/gaussian_blur \
	27_rng/mcmc_demo \
	28_thrust/thrust_demo \
	29_cublas/gemm_demo \
	30_mixed_precision/fp16_gemm \
	31_coop_groups/cg_reduce \
	32_gemm/gemm_v4 \
	33_softmax_ln/fused_softmax \
	34_flashattn/flashattn2 \
	35_quantized/int8_gemm \
	36_fused/bias_act_fused

.PHONY: all clean test

all: $(TARGETS)

# --- Block 1: GPU Architecture & CUDA Fundamentals ---

# Lesson 02: Vector Add (first CUDA kernel)
02_programming_model/vector_add: 02_programming_model/vector_add.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(MATH)

# Lesson 03: Thread Indexing / Matrix Transpose
03_thread_indexing/matrix_transpose: 03_thread_indexing/matrix_transpose.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 04: Memory Model / Bandwidth Benchmark
04_memory_model/bandwidth_bench: 04_memory_model/bandwidth_bench.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 05: Shared Memory / Tiled Matmul
05_shared_memory/tiled_matmul: 05_shared_memory/tiled_matmul.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(CUBLAS)

# Lesson 06: Warp Execution / Warp Reduce
06_warp_execution/warp_reduce: 06_warp_execution/warp_reduce.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 07: Atomics / Histogram
07_atomics/histogram_atomic: 07_atomics/histogram_atomic.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# --- Block 2: Performance Engineering ---

# Lesson 08: Memory Coalescing / Stride Benchmark
08_coalescing/stride_bench: 08_coalescing/stride_bench.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 09: Occupancy Demo
09_occupancy/occupancy_demo: 09_occupancy/occupancy_demo.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 10: Roofline Benchmark
10_roofline/roofline_bench: 10_roofline/roofline_bench.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(CUBLAS)

# Lesson 12: Streams / Double Buffering
12_streams/double_buffer: 12_streams/double_buffer.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 13: CUDA Graphs
13_cuda_graphs/graph_demo: 13_cuda_graphs/graph_demo.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# --- Block 3: Parallel Algorithms ---

# Lesson 14: Parallel Reduction
14_reduction/parallel_reduce: 14_reduction/parallel_reduce.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 15: Prefix Sum / Scan
15_scan/prefix_sum: 15_scan/prefix_sum.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 16: Radix Sort
16_sort/radix_sort: 16_sort/radix_sort.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 17: Stencil / Heat Equation
17_stencil/heat_stencil: 17_stencil/heat_stencil.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 18: Privatized Histogram
18_histogram/privatized_hist: 18_histogram/privatized_hist.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 19: SpMV (requires cuSPARSE)
19_sparse/spmv_demo: 19_sparse/spmv_demo.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(CUSPARSE)

# --- Block 4: Scientific Simulation ---

# Lesson 20: N-Body Simulation
20_nbody/nbody_sim: 20_nbody/nbody_sim.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 21: Monte Carlo (requires cuRAND)
21_monte_carlo/pi_estimate: 21_monte_carlo/pi_estimate.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(CURAND)

# Lesson 22: FFT Convolution (requires cuFFT)
22_fft/conv_fft: 22_fft/conv_fft.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(CUFFT)

# Lesson 23: 2D Heat Equation (PDE Solver)
23_pde/heat2d: 23_pde/heat2d.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 24: Lattice Boltzmann (LBM)
24_lbm/lbm_d2q9: 24_lbm/lbm_d2q9.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 25: Molecular Dynamics (Lennard-Jones)
25_md/lennard_jones: 25_md/lennard_jones.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 26: Image Processing (Gaussian Blur)
26_image/gaussian_blur: 26_image/gaussian_blur.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 27: MCMC / Stochastic (requires cuRAND)
27_rng/mcmc_demo: 27_rng/mcmc_demo.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(CURAND)

# --- Block 5: CUDA C++ Ecosystem ---

# Lesson 28: Thrust (header-only)
28_thrust/thrust_demo: 28_thrust/thrust_demo.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 29: cuBLAS GEMM
29_cublas/gemm_demo: 29_cublas/gemm_demo.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(CUBLAS)

# Lesson 30: Mixed Precision / Tensor Cores (FP16 GEMM)
30_mixed_precision/fp16_gemm: 30_mixed_precision/fp16_gemm.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(CUBLAS)

# Lesson 31: Cooperative Groups
31_coop_groups/cg_reduce: 31_coop_groups/cg_reduce.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) -rdc=true $< -o $@

# --- Block 6: High-Performance Kernels ---

# Lesson 32: GEMM from Scratch (v4 = float4 vectorized)
32_gemm/gemm_v4: 32_gemm/gemm_v4.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(CUBLAS)

# Lesson 33: Fused Softmax / LayerNorm
33_softmax_ln/fused_softmax: 33_softmax_ln/fused_softmax.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 34: FlashAttention-2 Kernel
34_flashattn/flashattn2: 34_flashattn/flashattn2.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# Lesson 35: INT8 GEMM (dp4a)
35_quantized/int8_gemm: 35_quantized/int8_gemm.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@ $(CUBLAS)

# Lesson 36: Fused Kernel Patterns
36_fused/bias_act_fused: 36_fused/bias_act_fused.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) $< -o $@

# --- Block 7: Multi-GPU & Capstone ---
# Lesson 37 (multi-GPU NCCL) and 38 (capstone) require
# environment-specific setup; build manually:
#   nvcc $(CFLAGS) 37_multi_gpu/allreduce_demo.cu -o 37_multi_gpu/allreduce_demo -lnccl
#   nvcc $(CFLAGS) 38_capstone/fluid_sim.cu       -o 38_capstone/fluid_sim

clean:
	rm -f $(TARGETS)
	find . -name "*.o" -delete

# Quick test: kernels that need only a GPU with no external data files
test: 02_programming_model/vector_add \
      06_warp_execution/warp_reduce \
      14_reduction/parallel_reduce \
      17_stencil/heat_stencil
	./02_programming_model/vector_add
	./06_warp_execution/warp_reduce
	./14_reduction/parallel_reduce
	./17_stencil/heat_stencil
	@echo "All self-contained GPU tests passed!"

# Target aliases for convenience
L02: 02_programming_model/vector_add
L03: 03_thread_indexing/matrix_transpose
L04: 04_memory_model/bandwidth_bench
L05: 05_shared_memory/tiled_matmul
L06: 06_warp_execution/warp_reduce
L07: 07_atomics/histogram_atomic
L08: 08_coalescing/stride_bench
L09: 09_occupancy/occupancy_demo
L10: 10_roofline/roofline_bench
L12: 12_streams/double_buffer
L13: 13_cuda_graphs/graph_demo
L14: 14_reduction/parallel_reduce
L15: 15_scan/prefix_sum
L16: 16_sort/radix_sort
L17: 17_stencil/heat_stencil
L18: 18_histogram/privatized_hist
L19: 19_sparse/spmv_demo
L20: 20_nbody/nbody_sim
L21: 21_monte_carlo/pi_estimate
L22: 22_fft/conv_fft
L23: 23_pde/heat2d
L24: 24_lbm/lbm_d2q9
L25: 25_md/lennard_jones
L26: 26_image/gaussian_blur
L27: 27_rng/mcmc_demo
L28: 28_thrust/thrust_demo
L29: 29_cublas/gemm_demo
L30: 30_mixed_precision/fp16_gemm
L31: 31_coop_groups/cg_reduce
L32: 32_gemm/gemm_v4
L33: 33_softmax_ln/fused_softmax
L34: 34_flashattn/flashattn2
L35: 35_quantized/int8_gemm
L36: 36_fused/bias_act_fused
