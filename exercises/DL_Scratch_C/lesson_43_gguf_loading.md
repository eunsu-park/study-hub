# Lesson 43 — GGUF Format and Model Loading (per-lesson exercise)

Prerequisites: L40 (quantization), basic C file I/O.

Compile: `gcc -std=c11 -Wall -Wextra -O2 -o ex ex.c -lm`

GGUF (GPT-Generated Unified Format) is the file format used by `llama.cpp` and most CPU-first inference engines for serving quantized LLMs. It packs model weights, tokenizer, and metadata into one file with a memory-map-friendly layout.

The format is:

```
| Magic 'GGUF' (4 bytes) | Version (uint32) |
| TensorCount (uint64) | MetadataKVCount (uint64) |
| Metadata KV pairs |
| Tensor info entries |
| --- alignment --- |
| Tensor data |
```

Reading it correctly — without a library — teaches you exactly how a model file works.

---

## Exercise 43.1 — Header Parsing

**Difficulty**: ★★

### Problem

Implement `gguf_read_header(FILE *fp, GGUFHeader *out)` that reads the first 24 bytes and validates the magic. Return error codes for: bad magic, unsupported version (we accept v3), I/O error.

### Starter

```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint32_t magic;          /* 0x46554747 ASCII "GGUF" little-endian */
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
} GGUFHeader;

int gguf_read_header(FILE *fp, GGUFHeader *out) {
    if (fread(out, sizeof(*out), 1, fp) != 1) return -1;
    if (memcmp(&out->magic, "GGUF", 4) != 0) return -2;
    if (out->version < 2 || out->version > 3) return -3;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s file.gguf\n", argv[0]); return 1; }
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { perror("fopen"); return 1; }

    GGUFHeader h;
    int rc = gguf_read_header(fp, &h);
    if (rc != 0) { printf("header error: %d\n", rc); fclose(fp); return 1; }

    printf("GGUF version=%u  tensors=%llu  metadata_kv=%llu\n",
           h.version,
           (unsigned long long)h.tensor_count,
           (unsigned long long)h.metadata_kv_count);
    fclose(fp);
    return 0;
}
```

---

## Exercise 43.2 — Reading One Tensor

**Difficulty**: ★★★

After the header, GGUF stores per-tensor metadata in this order: name (length-prefixed string), n_dimensions (uint32), dimensions (`n_dimensions` × uint64), dtype (uint32), offset (uint64).

The actual tensor bytes live in the data section, at the file offset = `data_section_start + tensor.offset`. The data section is aligned to 32 bytes by default.

Implement:

```c
typedef struct {
    char name[64];
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t dtype;
    uint64_t offset;        /* byte offset INTO the data section */
} GGUFTensorInfo;

int gguf_read_tensor_info(FILE *fp, GGUFTensorInfo *out);
```

Verify by listing the first 10 tensors in a Llama-2 7B GGUF (`tok_embeddings.weight`, `output.weight`, etc.). Their dimensions tell you the model architecture without ever loading weights.

---

## Exercise 43.3 — Memory-Mapped Loading — Bonus

**Difficulty**: ★★★

Use `mmap()` (POSIX) or `MapViewOfFile` (Windows) to map the file directly into the process address space. After parsing the headers, set up a pointer table: `weights[i] = mmap_base + data_start + tensor[i].offset`.

This is the trick that makes `llama.cpp` start near-instantly — there is no "loading" phase. The kernel demand-pages the weights as the inference layer touches them.

A 7B model in Q4_K_M occupies about 4 GB on disk. Loading vs. mmap timings:

| Method | Time to first token |
|--------|---------------------|
| `fread` whole file | 1.5–4 seconds (depends on disk) |
| `mmap` | <50 ms |

The catch: every byte read by inference is a page fault until the page is in the page cache. Subsequent runs from the page cache are as fast as if you had `fread`'d everything. The first run is "free" but slightly slower per token.

---

## Exercise 43.4 — Reading a Q4_0 Block — Bonus

**Difficulty**: ★★★★

Q4_0 quantization stores 32 weights as: one fp16 scale + 32 nibbles (4-bit signed integers). Implement:

```c
void dequantize_q4_0(const uint8_t *block, float *out_32);
```

Each block is exactly 18 bytes: 2 (fp16 scale) + 16 (32 × 4-bit). The dequantized weight is `scale * (nibble - 8)` (the `-8` recenters the unsigned nibble to signed). Test on a known block extracted from a real model file.

This connects L40 (quantization theory) to L43 (file format) in one short routine — and is the mechanism by which a 13B model fits in 8 GB of RAM.
