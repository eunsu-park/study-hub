# 43. GGUF Format and Loading

**Previous**: [Speculative Decoding](./42_Speculative_Decoding.md) | **Next**: [Parallel Inference](./44_Parallel_Inference.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the GGUF binary format: magic number, header, metadata key-value pairs, tensor info array, and tensor data section
2. Read GGUF metadata values (strings, integers, floats) from a binary file
3. Parse tensor info records to extract name, shape, and quantization type
4. Map GGML quantization type codes to C structs and sizes
5. Use `mmap()` for zero-copy model loading and sketch a basic Llama forward pass over GGUF data

---

## 1. GGUF File Format Overview

GGUF (Generic GPU Unified Format) is the native format for llama.cpp as of late 2023. It replaced the earlier GGML binary format. Key design goals: self-describing, extensible metadata, endian-safe, mmap-friendly.

```
GGUF file layout:
┌─────────────────────────────────────────────────────────┐
│ Header                                                  │
│   magic:         uint32  = 0x46554747 ("GGUF")          │
│   version:       uint32  = 3 (current)                  │
│   n_tensors:     uint64                                 │
│   n_kv:          uint64  (number of metadata key-values)│
├─────────────────────────────────────────────────────────┤
│ Metadata Key-Value pairs  (n_kv entries)                │
│   Each KV: key (string), type (uint32), value (varies) │
├─────────────────────────────────────────────────────────┤
│ Tensor Info array  (n_tensors entries)                  │
│   Each entry: name (string), n_dims, dims[], type, offset│
├─────────────────────────────────────────────────────────┤
│ [Alignment padding to next multiple of 32]              │
├─────────────────────────────────────────────────────────┤
│ Tensor Data section                                     │
│   Tensor 0 data (raw bytes, potentially quantized)      │
│   [alignment padding]                                   │
│   Tensor 1 data                                         │
│   ...                                                   │
└─────────────────────────────────────────────────────────┘
```

All multi-byte integers are little-endian. Strings are stored as (uint64 length, uint8[] bytes) with no null terminator.

---

## 2. GGUF Type Definitions

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define GGUF_MAGIC   0x46554747u  // "GGUF" little-endian
#define GGUF_VERSION 3

// GGUF metadata value types
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

// GGML tensor quantization types
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

// Bytes per element (for quantized types: bytes per block / elements per block)
// For aligned block types this is approximate (not all elements are same size)
size_t ggml_type_size(GGMLType t) {
    switch (t) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_Q4_0: return 18;   // 18 bytes per 32-element block
        case GGML_TYPE_Q8_0: return 34;   // 34 bytes per 32-element block
        case GGML_TYPE_Q4_K: return 144;  // 144 bytes per 256-element block
        case GGML_TYPE_Q6_K: return 210;  // 210 bytes per 256-element block
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

// Total bytes for n elements of a given type
size_t ggml_row_size(GGMLType t, size_t n_elements) {
    return (n_elements / ggml_blck_size(t)) * ggml_type_size(t);
}
```

---

## 3. Reading the GGUF Header

```c
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
} GGUFHeader;

// Read a GGUF string (uint64 length + bytes, no null terminator)
// Returns heap-allocated null-terminated string; caller must free
char *gguf_read_string(FILE *f) {
    uint64_t len;
    if (fread(&len, sizeof(len), 1, f) != 1) return NULL;
    if (len > 1024 * 1024) { fprintf(stderr, "String too long\n"); return NULL; }
    char *s = malloc(len + 1);
    if (fread(s, 1, len, f) != len) { free(s); return NULL; }
    s[len] = '\0';
    return s;
}

// Read the GGUF file header and validate magic/version
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

## 4. Reading Metadata Key-Value Pairs

```c
// Skip past a GGUF value (to advance file pointer past unknown keys)
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

// Read all metadata KV pairs, printing key-value info
// Extracts a few known keys into metadata struct
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

        // Check for known keys
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

## 5. Reading Tensor Info

```c
#define MAX_TENSOR_DIMS 4
#define MAX_TENSOR_NAME 256

typedef struct {
    char     name[MAX_TENSOR_NAME];
    uint32_t n_dims;
    uint64_t dims[MAX_TENSOR_DIMS];
    GGMLType type;
    uint64_t offset;   // byte offset from start of tensor data section
} GGUFTensorInfo;

// Read one tensor info record from GGUF file
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

// Read all tensor infos and print a summary
GGUFTensorInfo *gguf_read_tensor_info(FILE *f, uint64_t n_tensors) {
    GGUFTensorInfo *infos = malloc(n_tensors * sizeof(GGUFTensorInfo));
    uint64_t total_bytes = 0;

    for (uint64_t i = 0; i < n_tensors; i++) {
        if (gguf_read_one_tensor_info(&infos[i], f) < 0) {
            free(infos);
            return NULL;
        }

        // Compute tensor byte size
        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < infos[i].n_dims; d++)
            n_elements *= infos[i].dims[d];
        uint64_t bytes = ggml_row_size(infos[i].type, n_elements);
        total_bytes += bytes;

        if (i < 10 || i >= n_tensors - 3) {  // print first 10 and last 3
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

## 6. Memory-Mapped Loading

```c
typedef struct {
    void   *mmap_ptr;
    size_t  mmap_size;
    int     fd;
    // Pointer to start of tensor data section
    const uint8_t *tensor_data;
    uint64_t       tensor_data_offset;  // offset in file
} GGUFMmap;

// mmap the entire GGUF file for zero-copy tensor access
int gguf_mmap(GGUFMmap *mmap_out, const char *path, uint64_t tensor_data_offset) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }

    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return -1; }

    void *ptr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) { perror("mmap"); close(fd); return -1; }

    // Advise kernel: sequential access pattern for large model files
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

// Get a pointer to tensor data given its info
const void *gguf_tensor_data(const GGUFMmap *m, const GGUFTensorInfo *info) {
    return m->tensor_data + info->offset;
}
```

Memory mapping avoids copying the model into a separate heap allocation. The OS loads pages on-demand and can evict and reload them from disk as needed — critical for models larger than RAM.

---

## 7. Basic Q4_K Dequantization Sketch

```c
// Q4_K block structure (256 weights per block, simplified)
// Real llama.cpp struct has more complex scale encoding
#pragma pack(push, 1)
typedef struct {
    uint16_t d;        // super-block scale (FP16)
    uint16_t dmin;     // super-block minimum (FP16)
    uint8_t  scales[12]; // 8 sub-block scales (6-bit each, packed)
    uint8_t  qs[128];  // 256 weights at 4 bits each
} BlockQ4K;
#pragma pack(pop)

// Minimal FP16 to FP32 conversion
static float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0x1F) return sign ? -1.0f/0.0f : 1.0f/0.0f;  // inf/nan
    if (exp == 0) { float v = mant / (float)(1 << 24); return sign ? -v : v; }
    uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result; memcpy(&result, &f, 4); return result;
}

// Dequantize one Q4_K block to 256 floats (sketch — omits sub-block scales)
void dequantize_block_q4k_sketch(float *out, const BlockQ4K *blk) {
    float d    = fp16_to_float(blk->d);
    float dmin = fp16_to_float(blk->dmin);

    // Sub-block scale decoding from 6-bit packed values (8 sub-blocks)
    // Each sub-block covers 32 weights
    for (int sub = 0; sub < 8; sub++) {
        // Simplified: extract scale byte (real packing is more complex)
        uint8_t raw_scale = blk->scales[sub < 6 ? sub : sub - 1] & 0x3F;
        float   sub_d     = d    * raw_scale;
        float   sub_m     = dmin * raw_scale;

        for (int j = 0; j < 32; j++) {
            int idx  = sub * 32 + j;
            uint8_t byte = blk->qs[idx / 2];
            uint8_t q    = (idx % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
            out[idx] = sub_d * q - sub_m;  // unsigned [0,15], shifted by min
        }
    }
}
```

---

## 8. Full GGUF Loading Entry Point

```c
typedef struct {
    GGUFHeader      header;
    LlamaMetadata   meta;
    GGUFTensorInfo *tensor_infos;
    GGUFMmap        mmap;
    // For convenience: lookup by name (simple linear scan for demo)
} GGUFModel;

int gguf_load(GGUFModel *model, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }

    if (gguf_read_header(&model->header, f) < 0) { fclose(f); return -1; }
    gguf_read_metadata(&model->meta, f, model->header.n_kv);
    model->tensor_infos = gguf_read_tensor_info(f, model->header.n_tensors);
    if (!model->tensor_infos) { fclose(f); return -1; }

    // Tensor data starts after alignment to 32 bytes
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

// Find tensor by name (linear scan; use hash table in production)
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

## Key Takeaways

- GGUF is a self-describing binary format: magic number → header → metadata KV → tensor info → tensor data, with 32-byte alignment before tensor data.
- Strings in GGUF are length-prefixed (uint64 length + raw bytes, no null terminator) — always allocate `len + 1` and add the null terminator manually.
- The tensor data section contains raw bytes for each tensor in its quantized format; the offset field in tensor info is relative to the start of this section, not the file.
- `mmap()` with `MAP_PRIVATE` and `MADV_SEQUENTIAL` is the standard approach for loading models: zero-copy, demand-paged, and allows the OS to evict cold pages under memory pressure.
- GGML quantization types encode both data format and block structure; `ggml_row_size()` must use block-aware arithmetic to compute tensor byte sizes.
- Q4_K blocks use a two-level scale hierarchy (super-block scale + sub-block scales) packed at 6 bits each — dequantization requires careful bit manipulation to extract these packed scales.
- In production, tensor lookup should use a hash map keyed by name rather than linear scan over potentially thousands of tensors.

---

**Previous**: [Speculative Decoding](./42_Speculative_Decoding.md) | **Next**: [Parallel Inference](./44_Parallel_Inference.md)
