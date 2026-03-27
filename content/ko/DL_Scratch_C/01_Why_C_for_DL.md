# 01. 왜 딥러닝에 C/C++를 사용하는가?

**다음**: [메모리 레이아웃과 스트라이드](./02_Memory_Layout_and_Strides.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 프로덕션 LLM 추론 엔진이 C/C++로 작성된 이유를 설명
2. 시스템 수준 ML 작업에서 Python의 핵심 한계를 파악
3. llama.cpp 및 llm.c 소스 트리 탐색
4. 딥러닝 실험을 위한 C 빌드 환경 설정
5. C로 최소한의 텐서 "hello world" 작성 및 컴파일

---

## 1. Python의 한계

딥러닝 연구에서 Python의 우위는 정당합니다: 빠른 반복, 표현력 있는 NumPy/PyTorch API, 풍부한 생태계. 그러나 프로덕션 추론은 다른 문제입니다.

| 관심사 | Python | C/C++ |
|--------|--------|-------|
| 메모리 레이아웃 제어 | 제한적 (NumPy stride 도움이 되지만 GC 오버헤드) | 완전 — 구조체 수준 레이아웃, 수동 할당 |
| SIMD 벡터화 | 컴파일러 힌트를 통해; 신뢰 불가 | 직접 AVX2/AVX-512 intrinsics |
| 할당 지연 | GC 일시 정지; torch 내부 숨겨진 `malloc` | Arena allocator — 초기화 후 할당 없음 |
| 바이너리 배포 | Python 런타임, pip 환경 필요 | 단일 정적 바이너리, WASM 대상 |
| 지연 시간 (소형 배치) | 호출당 1–10 ms 프레임워크 오버헤드 | 마이크로초 디스패치 |
| 양자화 추론 | 제한적 (bitsandbytes, GPTQ 래퍼) | 커스텀 `dp4a` 커널로 네이티브 INT4/INT8 |

**대규모 배치 추론**에서는 Python + PyTorch로 충분합니다 — GPU 활용률이 지배적입니다. **단일 요청 지연 시간**, **엣지 배포**, **임베디드 시스템**, 또는 **WASM 브라우저 추론**에서는 C가 적합한 도구입니다.

---

## 2. C/C++ LLM 생태계

여러 영향력 있는 프로젝트들이 순수 C/C++에서 무엇이 가능한지 보여줍니다:

### llama.cpp
대표적인 C++ LLM 추론 라이브러리. GGUF 모델 형식, CPU + CUDA + Metal 백엔드를 지원하며, Raspberry Pi부터 H100 클러스터까지 모든 환경에서 실행됩니다.

```
llama.cpp/
├── ggml.c          ← 텐서 라이브러리 (핵심)
├── ggml-alloc.c    ← 메모리 할당자
├── llama.cpp       ← 모델 로딩 + 추론 루프
├── common/         ← 토크나이제이션, 샘플링 유틸리티
└── examples/       ← CLI, 서버, 임베딩 도구
```

핵심 포인트: `ggml.c`는 텐서, 자동미분, 양자화 matmul, 멀티스레드 스케줄링을 구현하는 ~17,000줄의 C 코드입니다. 이 코스에서는 단순화된 버전을 처음부터 구축합니다.

### llm.c (Karpathy)
GPT-2를 처음부터 학습하는 1,000줄짜리 C 파일. 표준 C와 선택적으로 cuBLAS 외에 의존성이 없는 교육학적으로 순수한 코드입니다. 이 코스에서는 Block 7에서 이 학습 루프를 재현합니다.

```c
// llm.c의 핵심: 단일 학습 스텝
encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C);
layernorm_forward(acts.ln1, acts.ln1_mean, acts.ln1_rstd, acts.encoded, ...);
matmul_forward(acts.qkv, acts.ln1, params.qkvw, params.qkvb, B, T, C, 3*C);
attention_forward(acts.attn, acts.preatt, acts.att, acts.qkv, B, T, NH, HS);
// ... 10개 레이어 더 ...
crossentropy_forward(model->mean_loss, acts.losses, acts.probs, targets, B, T, Vp);
```

### whisper.cpp
C++로 구현된 실시간 음성 인식. 동일한 C 텐서 기본 요소가 mel 스펙트로그램 전처리가 포함된 인코더 전용 Transformer 아키텍처로 확장되는 방법을 보여줍니다.

### ggml / GGUF
`ggml`은 기반 텐서 라이브러리입니다. `GGUF`(GGML Unified Format)는 전체 llama.cpp 생태계에서 사용되는 모델 파일 형식으로, 양자화된 가중치와 메타데이터를 위한 바이너리 컨테이너입니다. L43에서 GGUF 파일을 로딩합니다.

---

## 3. 이 코스에서 구축하는 것

이 코스를 마치면 다음을 작성하게 됩니다:

```
dl_scratch_c/
├── tensor/
│   ├── tensor.h / tensor.c      ← 텐서 구조체, stride, view
│   ├── ops.c                    ← element-wise, matmul, 리덕션
│   ├── simd_matmul.c            ← AVX2 SGEMM (L04)
│   └── autograd.c               ← 역방향 패스 엔진 (L05–L06)
├── memory/
│   └── arena.c                  ← Arena allocator (L07)
├── cnn/
│   ├── conv2d.c                 ← Convolution 순방향 + 역방향 (L08–L09)
│   ├── batchnorm.c              ← Batch normalization (L11)
│   └── models/
│       ├── lenet.c              ← LeNet-5 (L13)
│       └── resnet.c             ← ResNet-20 (L16)
├── transformer/
│   ├── tokenizer.c              ← BPE 토크나이저 (L21)
│   ├── attention.c              ← MHA + KV 캐시 (L25–L26)
│   ├── gpt2.c                   ← GPT-2 순방향 패스 (L29)
│   └── llama.c                  ← Llama 2/3 (RoPE, GQA, SwiGLU) (L30)
├── training/
│   ├── adamw.c                  ← AdamW 옵티마이저 (L35)
│   ├── dataloader.c             ← mmap 데이터 로더 (L36)
│   └── train_gpt2.c             ← 전체 학습 루프 (L38)
└── inference/
    ├── quantize.c               ← INT8/INT4 양자화 (L40)
    ├── flash_attn_cpu.c         ← FlashAttention-2 CPU (L41)
    ├── gguf_reader.c            ← GGUF 파일 파서 (L43)
    └── inference_engine.c       ← 최종 CLI 엔진 (L45)
```

---

## 4. 이 코스에서의 C vs C++

핵심 텐서 라이브러리에는 **C11**을, 템플릿과 RAII의 이점을 활용하는 모델 코드에는 **C++17**을 사용합니다.

| 컴포넌트 | 언어 | 이유 |
|---------|------|------|
| `tensor.c`, `ops.c` | C11 | 최대 이식성, 직접 SIMD |
| `autograd.c`, `arena.c` | C11 | 명시적 메모리 모델 |
| `attention.cpp`, `gpt2.cpp` | C++17 | `std::vector`, 설정을 위한 템플릿 |
| `gguf_reader.cpp` | C++17 | `std::map`, 구조체 생성자 |
| 빌드 시스템 | `Makefile` | 간단하고 재현 가능 |

---

## 5. Hello Tensor — 첫 번째 프로그램

최소한의 시작점을 작성해 봅시다: shape 메타데이터가 있는 평면 float 배열입니다.

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
    // 행 우선 (C-order) 스트라이드
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
    // 처음 8개 값 출력
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

    // 0, 1, 2, ... 로 채우기
    for (size_t i = 0; i < t->numel; i++)
        t->data[i] = (float)i;

    tensor_print(t);

    // 스트라이드를 사용한 [1][2] 요소 접근
    size_t row = 1, col = 2;
    float val = t->data[row * t->strides[0] + col * t->strides[1]];
    printf("  t[1][2] = %.1f  (예상값: 5.0)\n", val);

    tensor_free(t);
    return 0;
}
```

**빌드 및 실행**:
```bash
gcc -std=c11 -O2 -Wall hello_tensor.c -o hello_tensor
./hello_tensor
```

**예상 출력**:
```
Tensor [2 x 3]  numel=6
  data: [0.0000, 1.0000, 2.0000, 3.0000, 4.0000, 5.0000]
  t[1][2] = 5.0  (예상값: 5.0)
```

이 50줄짜리 프로그램이 이후의 모든 것의 씨앗입니다. L07에 이르면 이 구조체가 자동미분을 지원하게 됩니다.

---

## 6. 빌드 시스템 개요

이 코스의 모든 레슨은 독립적인 `Makefile`을 사용합니다:

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

Block 3부터는 OpenBLAS를 추가합니다:
```makefile
CFLAGS  += $(shell pkg-config --cflags openblas)
LDFLAGS += $(shell pkg-config --libs   openblas)
```

---

## 핵심 요약

- Python의 GC, 객체 오버헤드, SIMD 제어 부재는 지연 시간에 민감한 LLM 추론에 부적합합니다
- llama.cpp, llm.c, ggml은 작은 C 코드베이스가 추론에서 프로덕션 ML 프레임워크를 능가할 수 있음을 증명합니다
- `Tensor` 구조체(데이터 포인터 + shape + strides)가 기본 빌딩 블록입니다; L05에서 자동미분으로 확장합니다
- 모든 레슨은 동작하고 테스트 가능한 프로그램을 산출합니다 — 깨진 상태를 남겨두지 마십시오

---

**다음**: [02. 메모리 레이아웃과 스트라이드](./02_Memory_Layout_and_Strides.md) — 행 우선 레이아웃, 스트라이드 산술, 뷰가 어떻게 제로 카피 재형성을 가능하게 하는지 알아봅니다.
