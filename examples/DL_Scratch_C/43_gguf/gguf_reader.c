/*
 * gguf_reader.c -- GGUF format writer/reader demo
 *
 * Writes a small model's metadata and quantized tensors to a binary file
 * in GGUF format, then reads it back and verifies correctness.
 * Demonstrates the GGUF header structure, metadata KV pairs, tensor info,
 * and tensor data sections.
 *
 * Compile: gcc -std=c11 -Wall -Wextra -O2 -o gguf_reader gguf_reader.c -lm
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- GGUF constants ---- */

#define GGUF_MAGIC   0x46554747u  /* "GGUF" little-endian */
#define GGUF_VERSION 3
#define ALIGNMENT    32

/* Metadata value types */
enum {
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_STRING  = 8,
};

/* GGML tensor types */
enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_Q8_0 = 8,
};

/* ---- GGUF write helpers ---- */

static void write_string(FILE *f, const char *s) {
    uint64_t len = (uint64_t)strlen(s);
    fwrite(&len, sizeof(uint64_t), 1, f);
    fwrite(s, 1, (size_t)len, f);
}

static void write_kv_uint32(FILE *f, const char *key, uint32_t val) {
    write_string(f, key);
    uint32_t type = GGUF_TYPE_UINT32;
    fwrite(&type, sizeof(uint32_t), 1, f);
    fwrite(&val, sizeof(uint32_t), 1, f);
}

static void write_kv_float32(FILE *f, const char *key, float val) {
    write_string(f, key);
    uint32_t type = GGUF_TYPE_FLOAT32;
    fwrite(&type, sizeof(uint32_t), 1, f);
    fwrite(&val, sizeof(float), 1, f);
}

static void write_kv_string(FILE *f, const char *key, const char *val) {
    write_string(f, key);
    uint32_t type = GGUF_TYPE_STRING;
    fwrite(&type, sizeof(uint32_t), 1, f);
    write_string(f, val);
}

/* ---- Write GGUF file ---- */

static int write_gguf_file(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror("fopen write"); return -1; }

    /* Define our tiny model */
    const int d_model = 16;
    const int n_heads = 2;
    const int n_layers = 1;
    const int vocab = 64;

    /* Tensors to write */
    typedef struct {
        const char *name;
        uint32_t n_dims;
        uint64_t dims[2];
        uint32_t type;
        size_t   data_size;
    } TensorDef;

    TensorDef tensors[] = {
        {"token_embd.weight",   2, {(uint64_t)vocab, (uint64_t)d_model},  GGML_TYPE_F32,
         (size_t)vocab * d_model * sizeof(float)},
        {"blk.0.attn_q.weight", 2, {(uint64_t)d_model, (uint64_t)d_model}, GGML_TYPE_F32,
         (size_t)d_model * d_model * sizeof(float)},
        {"blk.0.attn_k.weight", 2, {(uint64_t)d_model, (uint64_t)d_model}, GGML_TYPE_F32,
         (size_t)d_model * d_model * sizeof(float)},
        {"blk.0.ffn_norm.weight", 1, {(uint64_t)d_model, 0}, GGML_TYPE_F32,
         (size_t)d_model * sizeof(float)},
    };
    int n_tensors = 4;
    int n_kv = 5;  /* metadata keys */

    /* Write header */
    uint32_t magic = GGUF_MAGIC;
    uint32_t version = GGUF_VERSION;
    uint64_t nt = (uint64_t)n_tensors;
    uint64_t nk = (uint64_t)n_kv;
    fwrite(&magic, sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);
    fwrite(&nt, sizeof(uint64_t), 1, f);
    fwrite(&nk, sizeof(uint64_t), 1, f);

    /* Write metadata */
    write_kv_string(f, "general.architecture", "llama");
    write_kv_uint32(f, "llama.block_count", (uint32_t)n_layers);
    write_kv_uint32(f, "llama.embedding_length", (uint32_t)d_model);
    write_kv_uint32(f, "llama.attention.head_count", (uint32_t)n_heads);
    write_kv_float32(f, "llama.rope.freq_base", 10000.0f);

    /* Write tensor info */
    uint64_t data_offset = 0;
    for (int i = 0; i < n_tensors; i++) {
        write_string(f, tensors[i].name);
        fwrite(&tensors[i].n_dims, sizeof(uint32_t), 1, f);
        for (uint32_t d = 0; d < tensors[i].n_dims; d++)
            fwrite(&tensors[i].dims[d], sizeof(uint64_t), 1, f);
        fwrite(&tensors[i].type, sizeof(uint32_t), 1, f);
        fwrite(&data_offset, sizeof(uint64_t), 1, f);
        data_offset += tensors[i].data_size;
        /* Align to ALIGNMENT */
        data_offset = (data_offset + ALIGNMENT - 1) & ~(uint64_t)(ALIGNMENT - 1);
    }

    /* Pad to alignment before tensor data */
    long pos = ftell(f);
    long aligned = (pos + ALIGNMENT - 1) & ~(long)(ALIGNMENT - 1);
    for (long p = pos; p < aligned; p++) {
        uint8_t zero = 0;
        fwrite(&zero, 1, 1, f);
    }

    /* Write tensor data (random values for demo) */
    srand(42);
    for (int i = 0; i < n_tensors; i++) {
        size_t n_floats = tensors[i].data_size / sizeof(float);
        for (size_t j = 0; j < n_floats; j++) {
            float val = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.2f;
            fwrite(&val, sizeof(float), 1, f);
        }
        /* Pad tensor data to alignment */
        pos = ftell(f);
        aligned = (pos + ALIGNMENT - 1) & ~(long)(ALIGNMENT - 1);
        for (long p = pos; p < aligned; p++) {
            uint8_t zero = 0;
            fwrite(&zero, 1, 1, f);
        }
    }

    long file_size = ftell(f);
    fclose(f);
    printf("Written GGUF file: %s (%ld bytes)\n\n", path, file_size);
    return 0;
}

/* ---- Read GGUF file ---- */

static char *read_string(FILE *f) {
    uint64_t len;
    if (fread(&len, sizeof(uint64_t), 1, f) != 1) return NULL;
    if (len > 1024 * 1024) return NULL;
    char *s = (char *)malloc((size_t)len + 1);
    if (fread(s, 1, (size_t)len, f) != (size_t)len) { free(s); return NULL; }
    s[len] = '\0';
    return s;
}

static int read_gguf_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen read"); return -1; }

    /* Read header */
    uint32_t magic, version;
    uint64_t n_tensors, n_kv;
    fread(&magic, sizeof(uint32_t), 1, f);
    fread(&version, sizeof(uint32_t), 1, f);
    fread(&n_tensors, sizeof(uint64_t), 1, f);
    fread(&n_kv, sizeof(uint64_t), 1, f);

    printf("--- GGUF Header ---\n");
    printf("  Magic:    0x%08X %s\n", magic,
           magic == GGUF_MAGIC ? "(valid)" : "(INVALID)");
    printf("  Version:  %u\n", version);
    printf("  Tensors:  %llu\n", (unsigned long long)n_tensors);
    printf("  Metadata: %llu key-value pairs\n\n", (unsigned long long)n_kv);

    if (magic != GGUF_MAGIC) { fclose(f); return -1; }

    /* Read metadata */
    printf("--- Metadata ---\n");
    for (uint64_t i = 0; i < n_kv; i++) {
        char *key = read_string(f);
        uint32_t type;
        fread(&type, sizeof(uint32_t), 1, f);

        printf("  [%llu] %s (type=%u) = ", (unsigned long long)i, key, type);

        if (type == GGUF_TYPE_UINT32) {
            uint32_t val;
            fread(&val, sizeof(uint32_t), 1, f);
            printf("%u\n", val);
        } else if (type == GGUF_TYPE_FLOAT32) {
            float val;
            fread(&val, sizeof(float), 1, f);
            printf("%.1f\n", val);
        } else if (type == GGUF_TYPE_STRING) {
            char *val = read_string(f);
            printf("\"%s\"\n", val);
            free(val);
        } else {
            printf("(unknown type, skipping)\n");
        }
        free(key);
    }

    /* Read tensor info */
    printf("\n--- Tensor Info ---\n");
    typedef struct {
        char name[256];
        uint32_t n_dims;
        uint64_t dims[4];
        uint32_t type;
        uint64_t offset;
    } TInfo;

    TInfo *infos = (TInfo *)malloc((size_t)n_tensors * sizeof(TInfo));
    for (uint64_t i = 0; i < n_tensors; i++) {
        char *name = read_string(f);
        strncpy(infos[i].name, name, 255);
        infos[i].name[255] = '\0';
        free(name);

        fread(&infos[i].n_dims, sizeof(uint32_t), 1, f);
        for (uint32_t d = 0; d < infos[i].n_dims; d++)
            fread(&infos[i].dims[d], sizeof(uint64_t), 1, f);
        fread(&infos[i].type, sizeof(uint32_t), 1, f);
        fread(&infos[i].offset, sizeof(uint64_t), 1, f);

        printf("  [%llu] %-30s type=%s dims=[",
               (unsigned long long)i, infos[i].name,
               infos[i].type == GGML_TYPE_F32 ? "F32" :
               infos[i].type == GGML_TYPE_Q8_0 ? "Q8_0" : "?");
        for (uint32_t d = 0; d < infos[i].n_dims; d++)
            printf("%s%llu", d ? ", " : "", (unsigned long long)infos[i].dims[d]);
        printf("] offset=%llu\n", (unsigned long long)infos[i].offset);
    }

    /* Align to tensor data section */
    long pos = ftell(f);
    long aligned = (pos + ALIGNMENT - 1) & ~(long)(ALIGNMENT - 1);
    fseek(f, aligned, SEEK_SET);
    long tensor_data_start = aligned;

    /* Read and verify first tensor */
    printf("\n--- Tensor Data Verification ---\n");
    srand(42);  /* Same seed as writer */

    for (uint64_t i = 0; i < n_tensors; i++) {
        fseek(f, tensor_data_start + (long)infos[i].offset, SEEK_SET);

        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < infos[i].n_dims; d++)
            n_elements *= infos[i].dims[d];

        /* Read first few values and compare with expected */
        int check = (int)(n_elements < 8 ? n_elements : 8);
        float values[8];
        fread(values, sizeof(float), (size_t)check, f);

        /* Regenerate expected values */
        float expected[8];
        for (int j = 0; j < (int)n_elements; j++) {
            float val = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.2f;
            if (j < check) expected[j] = val;
        }

        float max_err = 0.0f;
        for (int j = 0; j < check; j++) {
            float err = fabsf(values[j] - expected[j]);
            if (err > max_err) max_err = err;
        }

        printf("  %s: %llu elements, max verification error = %.2e %s\n",
               infos[i].name, (unsigned long long)n_elements,
               max_err, max_err < 1e-6f ? "OK" : "MISMATCH");
    }

    free(infos);
    fclose(f);
    return 0;
}

/* ---- main ---- */

int main(void) {
    const char *path = "/tmp/demo_model.gguf";

    printf("=== GGUF Format Writer/Reader Demo ===\n\n");

    /* Write */
    printf("--- Writing GGUF File ---\n");
    if (write_gguf_file(path) < 0) return 1;

    /* Read back */
    printf("--- Reading GGUF File ---\n");
    if (read_gguf_file(path) < 0) return 1;

    /* Print GGUF format summary */
    printf("\n--- GGUF Format Structure ---\n");
    printf("  +----------------------------------+\n");
    printf("  | Header                           |\n");
    printf("  |   magic:    uint32 (0x46554747)  |\n");
    printf("  |   version:  uint32               |\n");
    printf("  |   n_tensors: uint64              |\n");
    printf("  |   n_kv:      uint64              |\n");
    printf("  +----------------------------------+\n");
    printf("  | Metadata KV pairs (n_kv entries) |\n");
    printf("  |   key (string), type, value      |\n");
    printf("  +----------------------------------+\n");
    printf("  | Tensor Info (n_tensors entries)   |\n");
    printf("  |   name, n_dims, dims[], type, off|\n");
    printf("  +----------------------------------+\n");
    printf("  | [Alignment padding to 32 bytes]  |\n");
    printf("  +----------------------------------+\n");
    printf("  | Tensor Data                      |\n");
    printf("  |   Raw bytes (quantized/float)    |\n");
    printf("  +----------------------------------+\n");
    printf("\n  Strings: uint64 length + raw bytes (no null terminator)\n");
    printf("  All integers: little-endian\n");
    printf("  Tensor offsets: relative to tensor data section start\n");

    /* Clean up temp file */
    remove(path);

    return 0;
}
