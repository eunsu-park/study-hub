# 43. GGUF 형식과 로딩

**이전**: [Speculative Decoding](./42_Speculative_Decoding.md) | **다음**: [병렬 추론](./44_Parallel_Inference.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. GGUF 바이너리 형식 설명: magic number, 헤더, 메타데이터 키-값 쌍, tensor info 배열, tensor 데이터 섹션
2. 바이너리 파일에서 GGUF 메타데이터 값(문자열, 정수, 부동소수점) 읽기
3. tensor info 레코드를 파싱하여 이름, 형태, quantization 타입 추출
4. GGML quantization 타입 코드를 C 구조체와 크기에 매핑
5. 제로 카피 모델 로딩을 위해 `mmap()` 사용 및 GGUF 데이터에 대한 기본 Llama forward pass 스케치

---

## 1. GGUF 파일 형식 개요

GGUF(Generic GPU Unified Format)는 2023년 말부터 llama.cpp의 기본 형식입니다. 이전의 GGML 바이너리 형식을 대체했습니다. 주요 설계 목표: 자기 기술적, 확장 가능한 메타데이터, 엔디안 안전, mmap 친화적.

```
GGUF 파일 레이아웃:
┌─────────────────────────────────────────────────────────┐
│ 헤더                                                    │
│   magic:         uint32  = 0x46554747 ("GGUF")          │
│   version:       uint32  = 3 (현재)                     │
│   n_tensors:     uint64                                 │
│   n_kv:          uint64  (메타데이터 키-값의 수)         │
├─────────────────────────────────────────────────────────┤
│ 메타데이터 키-값 쌍  (n_kv 항목)                        │
│   각 KV: key (문자열), type (uint32), value (가변)      │
├─────────────────────────────────────────────────────────┤
│ Tensor Info 배열  (n_tensors 항목)                      │
│   각 항목: 이름 (문자열), n_dims, dims[], type, offset  │
├─────────────────────────────────────────────────────────┤
│ [32의 배수까지 정렬 패딩]                               │
├─────────────────────────────────────────────────────────┤
│ Tensor 데이터 섹션                                      │
│   Tensor 0 데이터 (원시 바이트, 잠재적으로 quantized)   │
│   [정렬 패딩]                                           │
│   Tensor 1 데이터                                       │
│   ...                                                   │
└─────────────────────────────────────────────────────────┘
```

모든 멀티바이트 정수는 리틀 엔디안입니다. 문자열은 (uint64 길이, uint8[] 바이트)로 저장되며 null 종결자는 없습니다.

---

## 2. GGUF 타입 정의

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define GGUF_MAGIC   0x46554747u  // "GGUF" 리틀 엔디안
#define GGUF_VERSION 3

// GGUF 메타데이터 값 타입
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} GGUFValueType;

// GGML tensor quantization 타입
typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q6_K = 14,
} GGMLType;

const char *ggml_type_to_str(GGMLType t) {
    switch (t) {
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q8_1: return "Q8_1";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        default:             return "UNKNOWN";
    }
}

// 요소당 바이트 (quantized 타입의 경우: 블록당 바이트 / 블록당 요소)
// 정렬된 블록 타입의 경우 이는 근사치입니다 (모든 요소가 동일한 크기가 아님)
size_t ggml_type_size(GGMLType t) {
    switch (t) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_Q4_0: return 18;   // 32-요소 블록당 18 바이트
        case GGML_TYPE_Q8_0: return 34;   // 32-요소 블록당 34 바이트
        case GGML_TYPE_Q4_K: return 144;  // 256-요소 블록당 144 바이트
        case GGML_TYPE_Q6_K: return 210;  // 256-요소 블록당 210 바이트
        default:             return 0;
    }
}

size_t ggml_blck_size(GGMLType t) {
    switch (t) {
        case GGML_TYPE_F32:  return 1;
        case GGML_TYPE_F16:  return 1;
        case GGML_TYPE_Q4_0: return 32;
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q4_K: return 256;
        case GGML_TYPE_Q6_K: return 256;
        default:             return 1;
    }
}

// 주어진 타입의 n 요소에 대한 총 바이트
size_t ggml_row_size(GGMLType t, size_t n_elements) {
    return (n_elements / ggml_blck_size(t)) * ggml_type_size(t);
}
```

---

## 3. GGUF 헤더 읽기

```c
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
} GGUFHeader;

// GGUF 문자열 읽기 (uint64 길이 + 바이트, null 종결자 없음)
// 힙 할당된 null 종결 문자열 반환; 호출자가 free해야 함
char *gguf_read_string(FILE *f) {
    uint64_t len;
    if (fread(&len, sizeof(len), 1, f) != 1) return NULL;
    if (len > 1024 * 1024) { fprintf(stderr, "String too long\n"); return NULL; }
    char *s = malloc(len + 1);
    if (fread(s, 1, len, f) != len) { free(s); return NULL; }
    s[len] = '\0';
    return s;
}

// GGUF 파일 헤더 읽기 및 magic/version 검증
int gguf_read_header(GGUFHeader *hdr, FILE *f) {
    if (fread(hdr, sizeof(*hdr), 1, f) != 1) {
        fprintf(stderr, "Failed to read GGUF header\n");
        return -1;
    }
    if (hdr->magic != GGUF_MAGIC) {
        fprintf(stderr, "Not a GGUF file (magic=0x%08X, expected 0x%08X)\n",
                hdr->magic, GGUF_MAGIC);
        return -1;
    }
    if (hdr->version < 2 || hdr->version > 3) {
        fprintf(stderr, "Unsupported GGUF version: %u\n", hdr->version);
        return -1;
    }
    printf("GGUF v%u: %llu tensors, %llu metadata keys\n",
           hdr->version,
           (unsigned long long)hdr->n_tensors,
           (unsigned long long)hdr->n_kv);
    return 0;
}
```

---

## 4. 메타데이터 키-값 쌍 읽기

```c
// GGUF 값 건너뛰기 (알 수 없는 키를 지나 파일 포인터 이동)
void gguf_skip_value(FILE *f, uint32_t type) {
    uint64_t len;
    char *s;
    uint32_t arr_type;
    uint64_t arr_count;

    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:   fseek(f, 1, SEEK_CUR); break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:  fseek(f, 2, SEEK_CUR); break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: fseek(f, 4, SEEK_CUR); break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: fseek(f, 8, SEEK_CUR); break;
        case GGUF_TYPE_STRING:
            s = gguf_read_string(f); free(s); break;
        case GGUF_TYPE_ARRAY:
            fread(&arr_type, 4, 1, f);
            fread(&arr_count, 8, 1, f);
            for (uint64_t i = 0; i < arr_count; i++)
                gguf_skip_value(f, arr_type);
            break;
        default:
            fprintf(stderr, "Unknown GGUF value type: %u\n", type);
    }
}

// 모든 메타데이터 KV 쌍 읽기, 키-값 정보 출력
// 몇 가지 알려진 키를 metadata 구조체에 추출
typedef struct {
    uint32_t n_ctx_train;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_layer;
    float    rope_freq_base;
    char     arch[64];
} LlamaMetadata;

void gguf_read_metadata(LlamaMetadata *meta, FILE *f, uint64_t n_kv) {
    memset(meta, 0, sizeof(*meta));

    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = gguf_read_string(f);
        uint32_t type;
        fread(&type, sizeof(type), 1, f);

        // 알려진 키 확인
        if (strcmp(key, "llama.context_length") == 0 && type == GGUF_TYPE_UINT32)
            fread(&meta->n_ctx_train, 4, 1, f);
        else if (strcmp(key, "llama.embedding_length") == 0 && type == GGUF_TYPE_UINT32)
            fread(&meta->n_embd, 4, 1, f);
        else if (strcmp(key, "llama.attention.head_count") == 0 && type == GGUF_TYPE_UINT32)
            fread(&meta->n_head, 4, 1, f);
        else if (strcmp(key, "llama.block_count") == 0 && type == GGUF_TYPE_UINT32)
            fread(&meta->n_layer, 4, 1, f);
        else if (strcmp(key, "llama.rope.freq_base") == 0 && type == GGUF_TYPE_FLOAT32)
            fread(&meta->rope_freq_base, 4, 1, f);
        else if (strcmp(key, "general.architecture") == 0 && type == GGUF_TYPE_STRING) {
            char *arch = gguf_read_string(f);
            strncpy(meta->arch, arch, 63);
            free(arch);
        }
        else {
            gguf_skip_value(f, type);
        }

        free(key);
    }

    printf("Architecture: %s\n", meta->arch);
    printf("  Layers: %u, Heads: %u, Embed: %u, Context: %u\n",
           meta->n_layer, meta->n_head, meta->n_embd, meta->n_ctx_train);
    if (meta->rope_freq_base > 0.0f)
        printf("  RoPE freq base: %.1f\n", meta->rope_freq_base);
}
```

---

## 5. Tensor Info 읽기

```c
#define MAX_TENSOR_DIMS 4
#define MAX_TENSOR_NAME 256

typedef struct {
    char     name[MAX_TENSOR_NAME];
    uint32_t n_dims;
    uint64_t dims[MAX_TENSOR_DIMS];
    GGMLType type;
    uint64_t offset;   // tensor 데이터 섹션 시작부터의 바이트 오프셋
} GGUFTensorInfo;

// GGUF 파일에서 하나의 tensor info 레코드 읽기
int gguf_read_one_tensor_info(GGUFTensorInfo *info, FILE *f) {
    char *name = gguf_read_string(f);
    if (!name) return -1;
    strncpy(info->name, name, MAX_TENSOR_NAME - 1);
    free(name);

    fread(&info->n_dims, sizeof(uint32_t), 1, f);
    if (info->n_dims > MAX_TENSOR_DIMS) {
        fprintf(stderr, "Too many dims: %u\n", info->n_dims);
        return -1;
    }
    for (uint32_t d = 0; d < info->n_dims; d++)
        fread(&info->dims[d], sizeof(uint64_t), 1, f);

    uint32_t type_int;
    fread(&type_int, sizeof(uint32_t), 1, f);
    info->type = (GGMLType)type_int;

    fread(&info->offset, sizeof(uint64_t), 1, f);
    return 0;
}

// 모든 tensor info 읽기 및 요약 출력
GGUFTensorInfo *gguf_read_tensor_info(FILE *f, uint64_t n_tensors) {
    GGUFTensorInfo *infos = malloc(n_tensors * sizeof(GGUFTensorInfo));
    uint64_t total_bytes = 0;

    for (uint64_t i = 0; i < n_tensors; i++) {
        if (gguf_read_one_tensor_info(&infos[i], f) < 0) {
            free(infos);
            return NULL;
        }

        // tensor 바이트 크기 계산
        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < infos[i].n_dims; d++)
            n_elements *= infos[i].dims[d];
        uint64_t bytes = ggml_row_size(infos[i].type, n_elements);
        total_bytes += bytes;

        if (i < 10 || i >= n_tensors - 3) {  // 처음 10개와 마지막 3개 출력
            printf("  [%3llu] %-48s %s [",
                   (unsigned long long)i, infos[i].name,
                   ggml_type_to_str(infos[i].type));
            for (uint32_t d = 0; d < infos[i].n_dims; d++)
                printf("%s%llu", d?",":"", (unsigned long long)infos[i].dims[d]);
            printf("] = %llu bytes\n", (unsigned long long)bytes);
        }
    }
    printf("Total tensor data: %.2f GB\n", total_bytes / 1e9);
    return infos;
}
```

---

## 6. 메모리 맵 로딩

```c
typedef struct {
    void   *mmap_ptr;
    size_t  mmap_size;
    int     fd;
    // tensor 데이터 섹션 시작 포인터
    const uint8_t *tensor_data;
    uint64_t       tensor_data_offset;  // 파일 내 오프셋
} GGUFMmap;

// 제로 카피 tensor 접근을 위해 전체 GGUF 파일을 mmap
int gguf_mmap(GGUFMmap *mmap_out, const char *path, uint64_t tensor_data_offset) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }

    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return -1; }

    void *ptr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) { perror("mmap"); close(fd); return -1; }

    // 커널에 알림: 대형 모델 파일에 대한 순차 접근 패턴
    madvise(ptr, st.st_size, MADV_SEQUENTIAL);

    mmap_out->mmap_ptr             = ptr;
    mmap_out->mmap_size            = st.st_size;
    mmap_out->fd                   = fd;
    mmap_out->tensor_data_offset   = tensor_data_offset;
    mmap_out->tensor_data          = (const uint8_t *)ptr + tensor_data_offset;

    printf("mmap'd %.2f GB model file\n", st.st_size / 1e9);
    return 0;
}

void gguf_unmap(GGUFMmap *m) {
    munmap(m->mmap_ptr, m->mmap_size);
    close(m->fd);
}

// 주어진 info에서 tensor 데이터 포인터 가져오기
const void *gguf_tensor_data(const GGUFMmap *m, const GGUFTensorInfo *info) {
    return m->tensor_data + info->offset;
}
```

메모리 매핑은 모델을 별도의 힙 할당으로 복사하는 것을 피합니다. OS는 필요에 따라 페이지를 로드하고 메모리 압박이 있을 때 디스크에서 제거 및 재로드할 수 있습니다 — RAM보다 큰 모델에 중요합니다.

---

## 7. 기본 Q4_K Dequantization 스케치

```c
// Q4_K 블록 구조 (블록당 256 가중치, 단순화됨)
// 실제 llama.cpp 구조체는 더 복잡한 scale 인코딩을 가짐
#pragma pack(push, 1)
typedef struct {
    uint16_t d;        // 슈퍼 블록 scale (FP16)
    uint16_t dmin;     // 슈퍼 블록 minimum (FP16)
    uint8_t  scales[12]; // 8개 서브 블록 scale (6-bit 각각, 패킹됨)
    uint8_t  qs[128];  // 4비트 각 256 가중치
} BlockQ4K;
#pragma pack(pop)

// 최소 FP16에서 FP32 변환
static float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0x1F) return sign ? -1.0f/0.0f : 1.0f/0.0f;  // inf/nan
    if (exp == 0) { float v = mant / (float)(1 << 24); return sign ? -v : v; }
    uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result; memcpy(&result, &f, 4); return result;
}

// 하나의 Q4_K 블록을 256 float으로 dequantize (스케치 — 서브 블록 scale 생략)
void dequantize_block_q4k_sketch(float *out, const BlockQ4K *blk) {
    float d    = fp16_to_float(blk->d);
    float dmin = fp16_to_float(blk->dmin);

    // 6-bit 패킹 값에서 서브 블록 scale 디코딩 (8개 서브 블록)
    // 각 서브 블록은 32 가중치를 커버
    for (int sub = 0; sub < 8; sub++) {
        // 단순화: scale 바이트 추출 (실제 패킹은 더 복잡)
        uint8_t raw_scale = blk->scales[sub < 6 ? sub : sub - 1] & 0x3F;
        float   sub_d     = d    * raw_scale;
        float   sub_m     = dmin * raw_scale;

        for (int j = 0; j < 32; j++) {
            int idx  = sub * 32 + j;
            uint8_t byte = blk->qs[idx / 2];
            uint8_t q    = (idx % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            out[idx] = sub_d * q - sub_m;  // 부호 없음 [0,15], min으로 이동
        }
    }
}
```

---

## 8. 전체 GGUF 로딩 진입점

```c
typedef struct {
    GGUFHeader      header;
    LlamaMetadata   meta;
    GGUFTensorInfo *tensor_infos;
    GGUFMmap        mmap;
    // 편의상: 이름으로 조회 (데모를 위한 단순 선형 스캔)
} GGUFModel;

int gguf_load(GGUFModel *model, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }

    if (gguf_read_header(&model->header, f) < 0) { fclose(f); return -1; }
    gguf_read_metadata(&model->meta, f, model->header.n_kv);
    model->tensor_infos = gguf_read_tensor_info(f, model->header.n_tensors);
    if (!model->tensor_infos) { fclose(f); return -1; }

    // Tensor 데이터는 32바이트 정렬 후 시작
    long pos = ftell(f);
    long aligned = (pos + 31) & ~31L;
    uint64_t tensor_data_offset = (uint64_t)aligned;

    fclose(f);

    if (gguf_mmap(&model->mmap, path, tensor_data_offset) < 0) {
        free(model->tensor_infos);
        return -1;
    }
    return 0;
}

// 이름으로 tensor 찾기 (선형 스캔; 프로덕션에서는 해시 테이블 사용)
const GGUFTensorInfo *gguf_find_tensor(const GGUFModel *m, const char *name) {
    for (uint64_t i = 0; i < m->header.n_tensors; i++)
        if (strcmp(m->tensor_infos[i].name, name) == 0)
            return &m->tensor_infos[i];
    return NULL;
}

void gguf_free(GGUFModel *model) {
    gguf_unmap(&model->mmap);
    free(model->tensor_infos);
}
```

---

## 핵심 요약

- GGUF는 자기 기술적 바이너리 형식입니다: magic number → 헤더 → 메타데이터 KV → tensor info → tensor 데이터, tensor 데이터 전에 32바이트 정렬.
- GGUF의 문자열은 길이 접두사(uint64 길이 + 원시 바이트, null 종결자 없음)로 저장됩니다 — 항상 `len + 1`을 할당하고 수동으로 null 종결자를 추가합니다.
- tensor 데이터 섹션은 quantized 형식의 각 tensor에 대한 원시 바이트를 포함합니다; tensor info의 offset 필드는 파일이 아닌 이 섹션의 시작을 기준으로 합니다.
- `MAP_PRIVATE`와 `MADV_SEQUENTIAL`을 사용한 `mmap()`은 모델 로딩의 표준 접근 방식입니다: 제로 카피, 요청 시 페이지 로드, 메모리 압박 시 OS가 콜드 페이지를 제거할 수 있습니다.
- GGML quantization 타입은 데이터 형식과 블록 구조를 모두 인코딩합니다; `ggml_row_size()`는 tensor 바이트 크기를 계산하기 위해 블록 인식 산술을 사용해야 합니다.
- Q4_K 블록은 2단계 scale 계층(슈퍼 블록 scale + 서브 블록 scale)을 각각 6비트로 패킹하여 사용합니다 — dequantization은 이 패킹된 scale을 추출하기 위해 신중한 비트 조작이 필요합니다.
- 프로덕션에서 tensor 조회는 잠재적으로 수천 개의 tensor에 대한 선형 스캔 대신 이름을 키로 하는 해시 맵을 사용해야 합니다.

---

**이전**: [Speculative Decoding](./42_Speculative_Decoding.md) | **다음**: [병렬 추론](./44_Parallel_Inference.md)
