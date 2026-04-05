# CUDA 프로그래밍 — 학습 가이드

## 소개

이 폴더는 **포괄적이고 범용적인 CUDA 커리큘럼**을 제공합니다 — GPU 아키텍처 기초부터 고성능 커스텀 커널, 과학 시뮬레이션, 멀티 GPU 분산 컴퓨팅까지. 핵심은 GPU를 단순히 딥러닝 가속기가 아닌 **병렬 컴퓨팅 플랫폼**으로 이해하는 것입니다.

**왜 범용 CUDA인가?**
CUDA는 딥러닝 시대보다 훨씬 이전에 과학 계산을 위해 개발되었습니다. N-체 시뮬레이션, 분자 동역학, 전산 유체 역학, 몬테카를로 방법, FFT 기반 신호 처리가 모두 GPU에서 네이티브로 실행됩니다. 이러한 응용을 이해하면 어떤 도메인에서도 전이 가능한 병렬 알고리즘 직관이 형성됩니다 — DL 커널도 궁극적으로 잘 최적화된 병렬 알고리즘에 불과합니다.

커리큘럼은 다음 진행 방식을 따릅니다:

```
하드웨어 → 프로그래밍 모델 → 성능 엔지니어링
       → 병렬 알고리즘 → 과학 응용
       → 라이브러리 생태계 → 커스텀 HPC 커널 → 멀티 GPU
```

## 대상 독자

- GPU를 프로그래밍하고 싶은 시스템 프로그래머 (**C_Advanced**, **CPP_Advanced**)
- 시뮬레이션을 가속화해야 하는 과학자와 엔지니어 (물리, 화학, 금융, 신호처리)
- PyTorch/TensorFlow 아래에서 무슨 일이 일어나는지 이해하고 싶은 ML 엔지니어
- CPU 병렬성의 한계에 부딪혀 GPU 규모의 처리량이 필요한 모든 분

## 선수 지식

| 토픽 | 필요 수준 |
|------|----------|
| **C_Advanced** | 능숙 — 포인터, 메모리, 컴파일 |
| **CPP_Advanced** | 기본 — 클래스, 템플릿, 연산자 오버로딩 |
| **Computer_Architecture** | 기본 — 캐시 계층, SIMD, 메모리 대역폭, Amdahl의 법칙 |
| Linear_Algebra | 권장 — GEMM 레슨(Block 6)을 위해 |
| Numerical_Simulation | 권장 — PDE/시뮬레이션 레슨(Block 4)을 위해 |
| Deep_Learning | 권장 — DL 커널 레슨(Block 6)을 위해 |

## 학습 로드맵

```
┌─────────────────────────┐
│  Block 1: GPU 아키텍처   │  L01–L07
│  & CUDA 기초             │  SIMT, 스레드 계층, 메모리 모델, atomic
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│  Block 2: 성능           │  L08–L13
│  엔지니어링              │  Coalescing, occupancy, roofline, profiling
└──────────┬──────────────┘
           │
┌──────────▼──────────────┐
│  Block 3: 병렬           │  L14–L19
│  알고리즘               │  Reduction, scan, sort, stencil, histogram
└─────┬────────────┬───────┘
      │            │
┌─────▼──────┐  ┌─▼───────────────────────────┐
│  Block 4:  │  │  Block 5: CUDA C++ 생태계    │  L28–L31
│  과학 계산  │  │  Thrust, CUB, cuBLAS,        │
│  L20–L27   │  │  Tensor Core, Coop. Groups   │
└─────┬──────┘  └──────────────┬───────────────┘
      └──────────┬─────────────┘
                 │
      ┌──────────▼──────────────┐
      │  Block 6: HPC 커널      │  L32–L36
      │  GEMM, Softmax, LN,     │
      │  FlashAttention, 양자화  │
      └──────────┬──────────────┘
                 │
      ┌──────────▼──────────────┐
      │  Block 7: 멀티 GPU      │  L37–L38
      │  & 캡스톤               │
      └─────────────────────────┘
```

## 파일 목록

| 레슨 | 파일명 | 난이도 | 설명 |
|------|--------|--------|------|
| **Block 1: GPU 아키텍처 & CUDA 기초** |
| L01 | `01_GPU_Architecture_Overview.md` | ⭐⭐ | SIMT vs SIMD, SM 구조, warp 실행, 피크 FLOPS/대역폭 |
| L02 | `02_CUDA_Programming_Model.md` | ⭐⭐ | `<<<grid, block>>>` 문법, 스레드 계층, 첫 커널 |
| L03 | `03_Thread_Indexing_and_Grids.md` | ⭐⭐⭐ | 1D/2D/3D 그리드, 전역 인덱스 계산, 경계 처리 |
| L04 | `04_CUDA_Memory_Model.md` | ⭐⭐⭐ | Global/shared/registers/L1-L2/constant/texture, 대역폭 벤치마크 |
| L05 | `05_Shared_Memory_and_Tiling.md` | ⭐⭐⭐ | 타일드 matmul, `__syncthreads()`, bank conflict, ncu 프로파일링 |
| L06 | `06_Warp_Execution_and_Divergence.md` | ⭐⭐⭐ | Warp divergence, 술어 실행, `__shfl_sync`, warp 리덕션 |
| L07 | `07_Atomic_Operations.md` | ⭐⭐⭐ | `atomicAdd/CAS/Exch`, 히스토그램 커널, 충돌 비용 측정 |
| **Block 2: 성능 엔지니어링** |
| L08 | `08_Memory_Coalescing.md` | ⭐⭐⭐ | 128바이트 트랜잭션, AoS vs SoA, stride 패널티, ncu 메트릭 |
| L09 | `09_Occupancy_and_Launch_Config.md` | ⭐⭐⭐⭐ | 레지스터 압력, 공유 메모리 한계, `__launch_bounds__` |
| L10 | `10_Roofline_Model.md` | ⭐⭐⭐⭐ | 산술 강도, 연산/메모리 한계, roofline 차트 |
| L11 | `11_Profiling_with_NCU_NSYS.md` | ⭐⭐⭐⭐ | Nsight Compute 커널 메트릭, Nsight Systems 타임라인 분석 |
| L12 | `12_Streams_and_Async.md` | ⭐⭐⭐⭐ | CUDA 스트림, `cudaMemcpyAsync`, double-buffering 파이프라인 |
| L13 | `13_CUDA_Graphs.md` | ⭐⭐⭐⭐ | 그래프 캡처, 스트림 캡처, CPU 런치 오버헤드 감소 |
| **Block 3: 병렬 알고리즘** |
| L14 | `14_Parallel_Reduction.md` | ⭐⭐⭐ | 트리 리덕션, warp shuffle, 다중 단계, CUB device reduce |
| L15 | `15_Parallel_Scan_Prefix_Sum.md` | ⭐⭐⭐⭐ | Hillis-Steele, Blelloch work-efficient scan, stream compaction |
| L16 | `16_Parallel_Sort.md` | ⭐⭐⭐⭐ | Bitonic sort, radix sort(CUB), merge sort, thrust::sort |
| L17 | `17_Stencil_Computations.md` | ⭐⭐⭐ | 1D/2D/3D stencil, 열방정식, halo cell, 주기 경계 조건 |
| L18 | `18_Histogram_and_Binning.md` | ⭐⭐⭐ | Atomic 히스토그램, 사유화(공유 메모리), 2D 빈닝 |
| L19 | `19_Sparse_Matrix_Ops.md` | ⭐⭐⭐⭐ | COO/CSR/CSC 형식, SpMV(cuSPARSE), SpGEMM |
| **Block 4: 과학 계산 & 시뮬레이션** |
| L20 | `20_N_Body_Simulation.md` | ⭐⭐⭐⭐ | 중력 N-체, Barnes-Hut 개념, 공유 메모리 타일 |
| L21 | `21_Monte_Carlo_Methods.md` | ⭐⭐⭐ | cuRAND, π 추정, Black-Scholes 옵션 가격, 분산 감소 |
| L22 | `22_FFT_on_GPU.md` | ⭐⭐⭐⭐ | cuFFT API, 1D/2D/3D, 배치 FFT, FFT를 통한 합성곱 |
| L23 | `23_PDE_Solvers_Heat_Equation.md` | ⭐⭐⭐ | 2D 열방정식(명시적 FD), stencil 커널, 안정성 조건 |
| L24 | `24_Fluid_Dynamics_LBM.md` | ⭐⭐⭐⭐ | Lattice Boltzmann D2Q9, collision + streaming 커널, 시각화 |
| L25 | `25_Molecular_Dynamics.md` | ⭐⭐⭐⭐ | Lennard-Jones 포텐셜, velocity Verlet, neighbor list, 에너지 보존 |
| L26 | `26_Image_Processing_GPU.md` | ⭐⭐⭐ | Gaussian blur, Sobel, bilateral filter, 히스토그램 평활화 |
| L27 | `27_Random_Number_and_Stochastic.md` | ⭐⭐⭐⭐ | 준무작위 수열, 병렬 MCMC, Metropolis-Hastings |
| **Block 5: CUDA C++ 생태계** |
| L28 | `28_Thrust_and_CUB.md` | ⭐⭐⭐ | Thrust STL 등가, CUB 블록/디바이스 기본 요소 |
| L29 | `29_cuBLAS_and_cuSPARSE.md` | ⭐⭐⭐⭐ | `cublasSgemm`, 배치 GEMM, `cublasGemmEx`(Tensor Core 경로) |
| L30 | `30_Mixed_Precision_and_Tensor_Cores.md` | ⭐⭐⭐⭐⭐ | FP16/BF16/FP8, WMMA API, loss scaling, Tensor Core FLOPS 측정 |
| L31 | `31_Cooperative_Groups.md` | ⭐⭐⭐⭐ | `cooperative_groups`, 그리드 수준 동기화, coalesced groups |
| **Block 6: 고성능 커스텀 커널** |
| L32 | `32_GEMM_from_Scratch.md` | ⭐⭐⭐⭐⭐ | Naive→shared→레지스터 타일링→float4 벡터화; cuBLAS 80%+ 목표 |
| L33 | `33_Softmax_and_LayerNorm_Kernels.md` | ⭐⭐⭐⭐ | Online softmax, 퓨전 LayerNorm/RMSNorm(warp shuffle), backward |
| L34 | `34_FlashAttention_Kernel.md` | ⭐⭐⭐⭐⭐ | FlashAttention-2: Q/K/V 타일링, online softmax, causal mask |
| L35 | `35_Quantized_Kernels_INT8.md` | ⭐⭐⭐⭐ | INT8 GEMM, `dp4a`, dequant 퓨전 에필로그, INT4 weight-only |
| L36 | `36_Fused_Kernel_Patterns.md` | ⭐⭐⭐⭐ | Bias+activation+dropout 퓨전, CUTLASS 에필로그 퓨전 |
| **Block 7: 멀티 GPU & 캡스톤** |
| L37 | `37_Multi_GPU_and_NCCL.md` | ⭐⭐⭐⭐⭐ | NVLink, P2P 전송, NCCL AllReduce, 텐서 병렬 |
| L38 | `38_Capstone_CUDA_Application.md` | ⭐⭐⭐⭐⭐ | 선택: (A) 유체 시뮬레이션 + 시각화 / (B) LLM 추론 엔진 |

**총 38개 레슨**

## 이 코스가 다른 점

대부분의 CUDA 코스는 딥러닝 커널에만 집중합니다. 이 코스는 **전체 GPU 컴퓨팅 생태계**를 다룹니다:

| 도메인 | 레슨 | 핵심 기법 |
|--------|------|---------|
| 물리 | L20(N-체), L25(분자동역학) | 입자 시스템, 포텐셜 에너지, neighbor list |
| 유체 역학 | L24(LBM) | Lattice Boltzmann, collision 연산자 |
| 수치 해석 | L23(PDE), L22(FFT) | 유한 차분, 스펙트럼 방법 |
| 통계 | L21, L27 | 몬테카를로, MCMC, 준무작위 |
| 영상 처리 | L26 | 합성곱, 엣지 검출, bilateral filter |
| 선형 대수 | L32(GEMM), L19(희소) | 밀집/희소 행렬 연산 |
| 딥러닝 | L33–L36 | FlashAttention, 양자화, 퓨전 연산 |

Block 3의 병렬 알고리즘 기초(reduction, scan, sort, stencil)는 **모든** 위 도메인의 빌딩 블록입니다.

## 난이도 곡선

```
Block 1 │▓▓░░░░░│  중 — CUDA 문법은 새롭지만 접근 가능
Block 2 │▓▓▓░░░░│  중상 — 하드웨어 내부 추론 필요
Block 3 │▓▓▓░░░░│  중상 — 병렬 사고 전환 필요 (첫 번째 벽)
Block 4 │▓▓░░░░░│  중~중상 — 도메인마다 다름; LBM/MD는 물리 배경 도움
Block 5 │▓▓░░░░░│  중 — 라이브러리 API 학습
Block 6 │▓▓▓▓▓▓▓│  전문 — 커스텀 GEMM과 FlashAttention (두 번째 벽)
Block 7 │▓▓▓▓▓░░│  고급 — 분산 + 통합 도전
```

**최고 난이도 레슨**: L30(Tensor Core WMMA), L32(GEMM 밑바닥), L34(FlashAttention 커널), L37(NCCL + 텐서 병렬)

## 핵심 마일스톤

| 완료 후 | 달성 가능한 것 |
|---------|--------------|
| L05 | NumPy보다 빠른 타일드 matmul 작성; 공유 메모리 숙달 |
| L10 | 어떤 커널이든 `ncu`/`nsys`로 프로파일링하고 주요 병목 설명 |
| L14 | 병렬 리덕션 구현 — 기본적인 GPU 기본 요소 |
| L19 | reduction/scan/sort/stencil을 사용하여 어떤 병렬 알고리즘도 구현 |
| L24 | 실시간 시각화와 함께 2D 유체 시뮬레이션(LBM) GPU에서 실행 |
| L27 | GPU 가속 몬테카를로 및 MCMC 샘플링 구현 |
| L32 | cuBLAS 성능의 80%+ 달성하는 커스텀 SGEMM 구축 |
| L34 | FlashAttention-2를 처음부터 구현; PyTorch와 검증 |
| L38 | 처음부터 완전한 CUDA 응용 프로그램 구축 (시뮬레이션 또는 추론) |

## 환경 설정

### NVIDIA GPU 필요

```bash
# GPU 및 CUDA 버전 확인
nvidia-smi
nvcc --version

# CUDA Toolkit 설치 (없는 경우)
# Ubuntu: https://developer.nvidia.com/cuda-downloads
# macOS: Metal 사용 (코드 예제에 Metal 대안 포함)

# 검증: 첫 번째 예제 실행
cd study-hub/examples/CUDA/02_CUDA_Programming_Model/
nvcc -O2 -o vector_add vector_add.cu
./vector_add
```

### 프로파일링 도구

```bash
# Nsight Compute (커널 수준 프로파일링)
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
    --metrics l1tex__t_bytes.sum \
    ./my_kernel

# Nsight Systems (시스템 수준 타임라인)
nsys profile --stats=true ./my_application
```

### 빌드 시스템

```makefile
# CUDA 예제용 일반적인 Makefile
NVCC    = nvcc
NVFLAGS = -O3 -arch=sm_80 --use_fast_math -Xcompiler -Wall
LIBS    = -lcublas -lcurand -lcufft

my_kernel: my_kernel.cu
	$(NVCC) $(NVFLAGS) -o $@ $^ $(LIBS)
```

> **GPU 아키텍처 플래그**: A100은 `-arch=sm_80`, RTX 4090은 `-arch=sm_89`, H100은 `-arch=sm_90`. RTX 2080/T4에는 `-arch=sm_75` 사용.

## 관련 토픽

- **[DL_Scratch_C](../DL_Scratch_C/00_Overview.md)**: C/C++ DL 구현 코스 — 여기서 다루는 많은 커널(attention, GEMM)이 그곳에서 구축된 것을 GPU 가속화
- **[Deep_Learning](../Deep_Learning/00_Overview.md)**: PyTorch 기반 DL — PyTorch가 내부에서 무엇을 호출하는지 이해
- **[Computer_Architecture](../Computer_Architecture/00_Overview.md)**: CPU 아키텍처 선수 과목 — 캐시, SIMD, 메모리 계층
- **[Numerical_Simulation](../Numerical_Simulation/00_Overview.md)**: Block 4에서 GPU 가속화하는 CPU 기반 수치 방법

## 학습 팁

1. **최적화 전에 프로파일하라**: 첫 번째 커널에 `ncu`를 사용하세요. 실제 병목이 무엇인지 놀랍게도 의외입니다.
2. **스레드가 아닌 warp으로 생각하라**: 모든 분기나 메모리 접근은 32개의 스레드에 동시에 영향을 미칩니다. 멘탈 모델: warp = 하나의 하드웨어 실행 단위.
3. **측정하라, 추측하지 마라**: GPU 성능은 직관에 어긋나는 경우가 많습니다. "느린" 커널은 연산이 아닌 메모리에서 멈추는 경우가 많습니다. roofline 모델(L10)이 답을 줍니다.
4. **Thrust부터 시작하라**: 프로토타이핑에는 Thrust/CUB를 사용하세요. 벤치마킹으로 Thrust가 병목임이 입증될 때만 커스텀 커널을 작성하세요.
5. **도메인 지식이 가속화됩니다**: LBM 레슨(L24)은 유체를 이해하면 쉽고; N-체(L20)는 뉴턴 역학을 알면 쉽습니다. 물리 맥락을 건너뛰지 마세요.

## 학습 성과

이 코스를 완료하면 다음을 할 수 있습니다:

- ✅ 올바른 스레드 인덱싱과 메모리 접근 패턴으로 CUDA C 커널 작성
- ✅ Nsight Compute로 커널을 프로파일링하고 주요 병목 식별
- ✅ 처음부터 병렬 reduction, scan, sort, stencil 연산 구현
- ✅ GPU 가속 과학 시뮬레이션 구축 (N-체, LBM, PDE 솔버, 몬테카를로)
- ✅ cuBLAS, cuFFT, cuRAND, cuSPARSE, Thrust를 생산적으로 사용
- ✅ cuBLAS 성능의 80%+를 달성하는 고성능 SGEMM 작성
- ✅ 알고리즘 논문에서 FlashAttention-2를 CUDA로 구현
- ✅ NCCL AllReduce를 사용하여 GPU 간 그래디언트 동기화
- ✅ 자신의 도메인에서 연산 집약적 문제에 GPU 가속 적용

---

`01_GPU_Architecture_Overview.md`로 시작하여 하드웨어 멘탈 모델을 구축한 다음, `02_CUDA_Programming_Model.md`로 첫 번째 커널을 작성하세요.
