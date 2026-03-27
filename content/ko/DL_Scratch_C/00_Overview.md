# C/C++로 딥러닝 밑바닥 구현 — 학습 가이드

## 소개

이 폴더는 **순수 C/C++로 딥러닝 전체 시스템을 구현**합니다 — PyTorch도, TensorFlow도, Python도 없습니다. 텐서 라이브러리와 자동미분부터 시작해 합성곱 신경망(LeNet → ResNet → EfficientNet → ViT), 완전한 Transformer 아키텍처(GPT-2 → Llama)를 구현하고, 최종적으로 llama.cpp에 필적하는 배포 가능한 LLM 추론 엔진으로 마무리합니다.

**왜 C/C++인가?**
현대의 LLM 추론 엔진들(llama.cpp, llm.c, whisper.cpp, ggml)은 C/C++로 작성됩니다. Python은 메모리 레이아웃, SIMD 벡터화, 할당 패턴을 프로덕션 성능에 필요한 수준으로 세밀하게 제어할 수 없기 때문입니다. C로 구현하면 GPU나 CPU가 데이터를 실제로 어떻게 처리하는지 *정확히* 이해하게 됩니다 — 숨겨주는 추상화가 없습니다.

이 커리큘럼은 **5단계 구현 철학**을 따릅니다:

| 단계 | 설명 | 예시 |
|------|------|------|
| **L1: Naive C** | 정확하지만 최적화되지 않음; 수학 검증 | 첫 번째 matmul, naive convolution |
| **L2: Cache-Aware** | 루프 타일링, 메모리 레이아웃, BLAS 호출 | 최적화된 SGEMM, im2col |
| **L3: SIMD** | 처리량을 위한 AVX2/AVX-512 intrinsics | 벡터화된 내부 루프 |
| **L4: Systems** | Arena allocator, mmap I/O, 스레딩 | 메모리 맵핑 데이터 로더, OpenMP |
| **L5: Production** | GGUF 로딩, INT4 양자화, 투기적 디코딩 | 최종 추론 엔진 |

## 대상 독자

- **Deep_Learning**(PyTorch 기반)을 완료하고 추상화 아래에서 무슨 일이 일어나는지 이해하고 싶은 엔지니어
- ML 도메인에 진입하는 시스템 프로그래머(**C_Advanced**, **CPP_Advanced**)
- 프레임워크 오버헤드 없이 모델 내부를 구현하고 수정하고 싶은 연구자
- llama.cpp, ggml, 또는 llm.c를 소스 코드 수준에서 이해하고 싶은 모든 분

## 선수 지식

| 토픽 | 필요 수준 |
|------|----------|
| **C_Advanced** | 능숙 — 포인터, 동적 할당, 파일 I/O, `Makefile` |
| **CPP_Advanced** | 능숙 — 템플릿, RAII, 연산자 오버로딩, C++17 |
| **Linear_Algebra** | 강함 — 행렬 곱셈, 브로드캐스팅, SVD(개념) |
| **Deep_Learning** | 완료 — 역전파, attention, GPT/Llama 아키텍처 |
| **Computer_Architecture** | 기본 — 캐시 계층, SIMD 개념, 메모리 대역폭 |
| Foundation_Models | 권장 — 스케일링, KV 캐시, 양자화, GGUF |
| OS_Theory | 권장 — mmap, 가상 메모리, 스레드 모델 |

## 학습 로드맵

```
┌─────────────────────┐
│  Block 1: 텐서 엔진  │  L01–L07
│  + 자동미분          │  C 텐서 라이브러리, AVX2 matmul, autograd 엔진
└──────────┬──────────┘
           │
     ┌─────▼──────────────┐          ┌──────────────────────┐
     │  Block 2: CNN 기초  │  L08–L14 │  Block 3: 모던 CNN    │  L15–L20
     │  Conv2D, BN, LeNet  │          │  ResNet / ViT 포석    │
     └─────┬──────────────┘          │  EfficientNet        │
           └────────────┬────────────└──────────┬───────────┘
                        │                       │
           ┌────────────▼───────────────────────┘
           │  Block 4: 토크나이제이션  L21–L23
           │  BPE, 임베딩, RoPE
           └────────────┬────────────┘
                        │
     ┌──────────────────▼──────────┐     ┌──────────────────────┐
     │  Block 5: Transformer       │     │  Block 6: ViT        │
     │  순방향 패스                 │     │  + 멀티모달           │
     │  GPT-2 / Llama / GQA / RoPE │     │  L31–L33             │
     │  L24–L30                    │     └──────────────────────┘
     └──────────────┬──────────────┘
                    │
     ┌──────────────▼──────────────┐
     │  Block 7: 처음부터 학습      │  L34–L38
     │  AdamW, 역전파, llm.c       │
     └──────────────┬──────────────┘
                    │
     ┌──────────────▼──────────────┐
     │  Block 8: 모던 추론          │  L39–L45
     │  양자화, FlashAttn,          │
     │  투기적 디코딩, GGUF 엔진    │
     └─────────────────────────────┘
```

## 파일 목록

| 레슨 | 파일명 | 난이도 | 설명 |
|------|--------|--------|------|
| **Block 1: 텐서 엔진 + 자동미분** |
| L01 | `01_Why_C_for_DL.md` | ⭐⭐ | C/C++로 DL을 구현하는 이유; llama.cpp, llm.c 개요 |
| L02 | `02_Memory_Layout_and_Strides.md` | ⭐⭐⭐ | 스트라이드 산술, shape/view, 캐시라인 정렬 |
| L03 | `03_Tensor_Ops_BLAS.md` | ⭐⭐⭐ | element-wise 연산, 리덕션, naive matmul vs OpenBLAS |
| L04 | `04_Optimized_Matmul.md` | ⭐⭐⭐⭐ | 루프 타일링, 레지스터 블로킹, AVX2 SGEMM |
| L05 | `05_Autograd_Engine.md` | ⭐⭐⭐⭐ | 연산 그래프, 위상 정렬, C에서의 `backward()` |
| L06 | `06_Autograd_Tensor_Ops.md` | ⭐⭐⭐⭐ | matmul/softmax/cross-entropy backward, 유한차분 검증 |
| L07 | `07_Memory_Manager.md` | ⭐⭐⭐⭐ | Arena allocator, 참조 카운팅, 제로카피 뷰 |
| **Block 2: CNN — 기초** |
| L08 | `08_Convolution_from_Scratch.md` | ⭐⭐⭐ | Naive conv2D, stride/padding/dilation, im2col 트릭 |
| L09 | `09_Convolution_Backward.md` | ⭐⭐⭐⭐ | input/filter/bias gradient, 수치 검증 |
| L10 | `10_Pooling_Layers.md` | ⭐⭐⭐ | Max/average/global pooling, 순방향/역방향 |
| L11 | `11_Batch_Normalization.md` | ⭐⭐⭐⭐ | BN 학습/평가 모드, 이동 통계, 역방향 패스 |
| L12 | `12_Data_Pipeline_Images.md` | ⭐⭐⭐ | STB 이미지 로딩, NCHW/NHWC, 데이터 증강 |
| L13 | `13_LeNet_and_AlexNet.md` | ⭐⭐⭐ | C로 구현한 LeNet-5 + AlexNet, CIFAR-10 학습 파이프라인 |
| L14 | `14_Training_CNN_CIFAR10.md` | ⭐⭐⭐⭐ | End-to-end CNN 학습: 로더 → 순방향 → 손실 → 역방향 |
| **Block 3: CNN — 모던 아키텍처** |
| L15 | `15_VGG_and_Deep_Networks.md` | ⭐⭐⭐ | VGG-16/19, 깊이 vs. vanishing gradient, 파라미터 수 |
| L16 | `16_ResNet_and_Skip_Connections.md` | ⭐⭐⭐⭐ | Residual block, identity/projection shortcut, backward |
| L17 | `17_Depthwise_Separable_Conv.md` | ⭐⭐⭐ | Depthwise + pointwise, MobileNet 스타일, FLOP 비교 |
| L18 | `18_Squeeze_Excitation_and_Attention.md` | ⭐⭐⭐ | SE block (채널 어텐션), CBAM, ViT 포석 |
| L19 | `19_EfficientNet_Scaling.md` | ⭐⭐⭐⭐ | 복합 스케일링, NAS 개념, EfficientNet-B0 |
| L20 | `20_Modern_CNN_Benchmark.md` | ⭐⭐⭐ | CIFAR-10/100: LeNet vs ResNet-20 vs EfficientNet |
| **Block 4: 토크나이제이션 & 임베딩** |
| L21 | `21_Tokenization_BPE.md` | ⭐⭐⭐ | BPE 알고리즘, byte-level BPE(GPT-2), tiktoken 파일 |
| L22 | `22_Embedding_Table.md` | ⭐⭐⭐ | 룩업 테이블, weight tying, 바이너리 가중치 로딩 |
| L23 | `23_Positional_Encodings.md` | ⭐⭐⭐ | Sinusoidal, Learned PE, 실수 산술로 구현한 RoPE |
| **Block 5: Transformer 순방향 패스** |
| L24 | `24_Layer_Normalization.md` | ⭐⭐⭐ | LayerNorm vs RMSNorm, 순/역방향, gamma/beta |
| L25 | `25_Attention_Mechanism.md` | ⭐⭐⭐⭐ | MHA: Q/K/V 투영, scaled dot-product, causal mask |
| L26 | `26_KV_Cache.md` | ⭐⭐⭐⭐ | 사전 할당 KV 버퍼, append-only, 메모리 분석 |
| L27 | `27_FFN_and_Activations.md` | ⭐⭐⭐ | GELU(GPT-2) vs SwiGLU(Llama): `silu(gate) * up` |
| L28 | `28_Transformer_Block.md` | ⭐⭐⭐⭐ | Pre-norm + residual + attn + FFN; PyTorch 출력과 비교 |
| L29 | `29_GPT2_Forward_Pass.md` | ⭐⭐⭐⭐ | GPT-2(124M) 전체 순방향, 실제 가중치, 로짓 검증 |
| L30 | `30_Llama_Architecture.md` | ⭐⭐⭐⭐ | Llama 2/3: RMSNorm, SwiGLU, RoPE, GQA |
| **Block 6: Vision Transformer** |
| L31 | `31_Vision_Transformer_ViT.md` | ⭐⭐⭐⭐ | 패치 임베딩, [CLS] 토큰, 2D PE, ViT-Base |
| L32 | `32_ViT_Training_and_Fine_Tuning.md` | ⭐⭐⭐⭐ | Warm-up + cosine LR, CutMix, ImageNet 스타일 학습 |
| L33 | `33_Multimodal_CLIP_Style.md` | ⭐⭐⭐⭐ | InfoNCE 손실, 이미지+텍스트 인코더, 코사인 유사도 |
| **Block 7: LLM 처음부터 학습** |
| L34 | `34_Cross_Entropy_Loss.md` | ⭐⭐⭐ | Log-softmax + NLL, 수치 안정성, 퓨전 역방향 |
| L35 | `35_Optimizers.md` | ⭐⭐⭐ | SGD 모멘텀, Adam, AdamW, gradient clipping, LR 스케줄 |
| L36 | `36_Training_Loop.md` | ⭐⭐⭐⭐ | mmap 데이터 로더, mini-batch 샘플링, 손실 로깅 |
| L37 | `37_Backprop_Through_Transformer.md` | ⭐⭐⭐⭐⭐ | Attention backward, softmax-QK^T gradient, 전체 역전파 |
| L38 | `38_Training_GPT2_Small.md` | ⭐⭐⭐⭐⭐ | GPT-2 small end-to-end, llm.c 재현, 벤치마크 |
| **Block 8: 모던 추론** |
| L39 | `39_Sampling_Strategies.md` | ⭐⭐⭐ | Greedy, temperature, top-k, top-p, min-p, 반복 페널티 |
| L40 | `40_Quantization_Int8_Int4.md` | ⭐⭐⭐⭐ | Absmax INT8, per-channel, INT4 weight-only(GGUF 스타일) |
| L41 | `41_FlashAttention_CPU.md` | ⭐⭐⭐⭐⭐ | FA1/FA2 타일링, IO-복잡도, CPU 구현 |
| L42 | `42_Speculative_Decoding.md` | ⭐⭐⭐⭐⭐ | Draft-verify 루프, rejection sampling, 속도 측정 |
| L43 | `43_GGUF_and_Loading.md` | ⭐⭐⭐⭐ | GGUF 포맷 파싱, Q4_K_M 로딩, 실제 Llama-3 추론 |
| L44 | `44_Parallel_Inference.md` | ⭐⭐⭐⭐ | OpenMP/pthreads 텐서 병렬, bandwidth 병목 분석 |
| L45 | `45_Capstone_Inference_Engine.md` | ⭐⭐⭐⭐⭐ | 완전한 CLI 엔진: GGUF + INT4 + KV 캐시 + GQA + 샘플링 |

**총 45개 레슨**

## 난이도 곡선

```
Block 1 │▓▓▓▓░░░│  중상 — C 자동미분이 첫 번째 큰 벽
Block 2 │▓▓▓░░░░│  중 — Conv backward는 까다롭지만 가능
Block 3 │▓▓▓▓░░░│  중상 — skip connection backward
Block 4 │▓▓░░░░░│  중 — 토크나이제이션은 비교적 직관적
Block 5 │▓▓▓▓░░░│  중상 — 조립이 관건
Block 6 │▓▓▓▓▓░░│  고급 — ViT 패치 임베딩 + contrastive loss
Block 7 │▓▓▓▓▓▓▓│  전문 — 전체 Transformer 역전파가 최고 난이도
Block 8 │▓▓▓▓▓░░│  고급 — 시스템 엔지니어링 + 알고리즘 깊이
```

**최고 난이도 레슨**: L05(C에서의 autograd), L09(conv backward), L37(전체 Transformer 역전파), L41(FlashAttention CPU), L42(투기적 디코딩)

## 핵심 마일스톤

| 완료 후 | 달성 가능한 것 |
|---------|--------------|
| L07 | 수치 검증이 포함된 2-layer MLP forward+backward를 C로 실행 |
| L14 | LeNet/AlexNet을 CIFAR-10에서 순수 C로 학습 |
| L20 | ResNet-20과 EfficientNet-B0 구현; CNN 계보 완전 이해 |
| L29 | HuggingFace 로짓과 일치하는 실제 가중치로 GPT-2(124M) 실행 |
| L33 | CNN과 Transformer가 만나는 CLIP 스타일 멀티모달 모델 구축 |
| L38 | C로 GPT-2 small을 처음부터 학습(Karpathy의 llm.c 재현) |
| L45 | 실제 양자화된 Llama GGUF를 로딩하여 CLI에서 텍스트 생성 |

## 환경 설정

```bash
# macOS
xcode-select --install
brew install openblas

# Ubuntu/Debian
sudo apt-get install build-essential libopenblas-dev

# 예제 빌드 (각 레슨에 Makefile 포함)
cd study-hub/examples/DL_Scratch_C/01_Why_C_for_DL/
make && ./hello_tensor

# 전체 블록 예제 빌드
make -C study-hub/examples/DL_Scratch_C/
```

### 권장 컴파일러 플래그

```makefile
CFLAGS   = -std=c11   -O2 -march=native -Wall -Wextra
CXXFLAGS = -std=c++17 -O2 -march=native -Wall -Wextra
LIBS     = -lopenblas -lm -lpthread
```

### 선택적 도구

- **Valgrind** — Block 1–2 진행 중 메모리 누수 감지
- **perf / Instruments** — Block 4(matmul 최적화)를 위한 CPU 프로파일링
- **Python + PyTorch** — 수치 정확도 테스트를 위한 참조 출력

## 관련 토픽

- **[Deep_Learning](../Deep_Learning/00_Overview.md)**: PyTorch 기반의 동반 코스 — 동일한 아키텍처, 높은 수준의 추상화
- **[CUDA](../CUDA/00_Overview.md)**: 이 코스에서 구축한 커널(attention, GEMM)의 GPU 가속
- **[C_Advanced](../C_Advanced/00_Overview.md)**: 시스템 프로그래밍 선수 과목
- **[Foundation_Models](../Foundation_Models/00_Overview.md)**: 스케일링, 양자화 이론, GGUF 에코시스템
- **[Computer_Architecture](../Computer_Architecture/00_Overview.md)**: 캐시 계층, SIMD, roofline 모델

## 학습 팁

1. **먼저 수치적으로 검증하라**: 레이어를 최적화하기 전에 Python/NumPy 레퍼런스를 작성하고 소수점 6자리 이상까지 출력을 비교하라.
2. **점진적으로 구축하라**: 각 레슨의 코드는 다음으로 넘어가기 전에 컴파일되고 정확한 출력을 내야 한다. 기술 부채를 쌓지 마라.
3. **실제 소스를 읽어라**: 이 레슨들과 병행해서 llama.cpp와 llm.c를 공부하라 — 맥락을 알면 코드 선택이 더 잘 이해된다.
4. **최적화하기 전에 프로파일하라**: `perf stat`이나 Instruments로 시간이 실제로 어디서 소비되는지 확인하라.
5. **메모리가 병목이다**: DL 추론의 거의 모든 성능 퍼즐은 "메모리 대역폭 부족"으로 귀결된다. FLOP/s가 아닌 바이트/초로 생각하라.

## 학습 성과

이 코스를 완료하면 다음을 할 수 있습니다:

- ✅ C로 자동미분을 포함한 텐서 라이브러리를 처음부터 구현
- ✅ 순수 C로 CNN(LeNet → ResNet → EfficientNet) 구축 및 학습
- ✅ GQA와 RoPE를 포함한 완전한 Transformer 아키텍처(GPT-2, Llama) 구현
- ✅ C로 처음부터 언어 모델 학습(llm.c 결과 재현)
- ✅ INT8/INT4 양자화 적용 및 perplexity 저하 측정
- ✅ CPU에서 FlashAttention-2 타일링 로직 구현
- ✅ 실제 GGUF 모델 파일을 로딩하여 CLI에서 LLM 추론 실행
- ✅ llama.cpp / ggml 소스 코드를 자신 있게 읽고 확장

---

`01_Why_C_for_DL.md`로 시작해서 전체 랜드스케이프를 파악한 다음, `02_Memory_Layout_and_Strides.md`로 기반 데이터 모델을 구축하세요.
