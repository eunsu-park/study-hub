// file_encrypt_v2.c
// Improved file encryption tool (header + key verification)
// Learning objective: Structs, file headers, hash functions, improved error handling

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Constants
#define MAGIC "XENC"          // Magic number for file identification
#define VERSION 1             // File format version
#define BUFFER_SIZE 4096      // 4KB buffer
#define HEADER_SIZE 17        // Header size (bytes)

/**
 * Encrypted file header struct
 *
 * Layout:
 * - magic[4]: "XENC" magic number (file identification)
 * - version: File format version (1 byte)
 * - key_hash: Hash of the key (4 bytes, for key verification)
 * - original_size: Original file size (8 bytes)
 */
typedef struct {
    char magic[4];          // Magic number: "XENC"
    uint8_t version;        // Version: 1
    uint32_t key_hash;      // Key hash (djb2)
    uint64_t original_size; // Original file size
} FileHeader;

// Function declarations
void print_usage(const char *name);
uint32_t hash_key(const char *key);
void xor_buffer(unsigned char *buf, size_t len, const char *key, size_t key_len, size_t *pos);
int encrypt_file(const char *input, const char *output, const char *key);
int decrypt_file(const char *input, const char *output, const char *key);
int show_info(const char *filename);

/**
 * Simple hash function (djb2 algorithm)
 *
 * @param key String to hash
 * @return 32-bit hash value
 *
 * djb2 is a simple yet effective hash function created by Daniel J. Bernstein
 * Initial value: 5381
 * Formula: hash = hash * 33 + c
 */
uint32_t hash_key(const char *key) {
    uint32_t hash = 5381;
    int c;

    while ((c = *key++)) {
        // hash * 33 + c = (hash << 5) + hash + c
        hash = ((hash << 5) + hash) + c;
    }

    return hash;
}

/**
 * Print usage
 */
void print_usage(const char *name) {
    printf("Improved File Encryption Tool v2\n\n");
    printf("Usage:\n");
    printf("  %s encrypt <input> <output> <password>\n", name);
    printf("  %s decrypt <input> <output> <password>\n", name);
    printf("  %s info <encrypted_file>\n", name);
    printf("\nExamples:\n");
    printf("  %s encrypt secret.txt secret.enc mypassword\n", name);
    printf("  %s decrypt secret.enc decrypted.txt mypassword\n", name);
    printf("  %s info secret.enc\n", name);
}

/**
 * XOR encrypt/decrypt buffer (tracks key position)
 *
 * @param buf Buffer to process
 * @param len Buffer length
 * @param key Encryption key
 * @param key_len Key length
 * @param pos Current key position (pointer, maintains state)
 *
 * The pos parameter tracks key position continuously across multiple buffers.
 * This ensures the key pattern remains consistent even when reading the file in blocks.
 */
void xor_buffer(unsigned char *buf, size_t len, const char *key, size_t key_len, size_t *pos) {
    for (size_t i = 0; i < len; i++) {
        buf[i] ^= key[*pos % key_len];
        (*pos)++;  // Advance key position
    }
}

/**
 * Encrypt file (with header)
 *
 * @param input Input file path
 * @param output Output file path
 * @param key Encryption key
 * @return 0 on success, 1 on failure
 */
int encrypt_file(const char *input, const char *output, const char *key) {
    // Open input file
    FILE *fin = fopen(input, "rb");
    if (!fin) {
        perror("Failed to open input file");
        return 1;
    }

    // Get original file size
    fseek(fin, 0, SEEK_END);
    uint64_t file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);

    // Open output file
    FILE *fout = fopen(output, "wb");
    if (!fout) {
        perror("Failed to open output file");
        fclose(fin);
        return 1;
    }

    // Initialize header struct
    FileHeader header;
    memcpy(header.magic, MAGIC, 4);     // Copy "XENC"
    header.version = VERSION;            // Version 1
    header.key_hash = hash_key(key);     // Store key hash
    header.original_size = file_size;    // Store original size

    // Write header to file
    fwrite(&header, sizeof(FileHeader), 1, fout);

    // Encrypt data
    unsigned char buffer[BUFFER_SIZE];
    size_t bytes_read;
    size_t key_len = strlen(key);
    size_t key_pos = 0;  // Track key position

    printf("Encrypting...\n");
    while ((bytes_read = fread(buffer, 1, BUFFER_SIZE, fin)) > 0) {
        xor_buffer(buffer, bytes_read, key, key_len, &key_pos);
        fwrite(buffer, 1, bytes_read, fout);

        // Simple progress indicator
        printf(".");
        fflush(stdout);
    }
    printf("\n");

    fclose(fin);
    fclose(fout);

    printf("Encryption complete: %s -> %s\n", input, output);
    printf("Original size: %llu bytes\n", (unsigned long long)file_size);
    printf("Key hash: 0x%08X\n", header.key_hash);

    return 0;
}

/**
 * Decrypt file (with header verification)
 *
 * @param input Input file path (encrypted file)
 * @param output Output file path (decrypted file)
 * @param key Decryption key
 * @return 0 on success, 1 on failure
 */
int decrypt_file(const char *input, const char *output, const char *key) {
    // Open encrypted file
    FILE *fin = fopen(input, "rb");
    if (!fin) {
        perror("Failed to open input file");
        return 1;
    }

    // Read header
    FileHeader header;
    if (fread(&header, sizeof(FileHeader), 1, fin) != 1) {
        fprintf(stderr, "Error: Invalid encrypted file (failed to read header)\n");
        fclose(fin);
        return 1;
    }

    // Verify magic number (check if file is XENC format)
    if (memcmp(header.magic, MAGIC, 4) != 0) {
        fprintf(stderr, "Error: Not a valid encrypted file (magic number mismatch)\n");
        fprintf(stderr, "Expected: %.4s, Actual: %.4s\n", MAGIC, header.magic);
        fclose(fin);
        return 1;
    }

    // Check version
    if (header.version != VERSION) {
        fprintf(stderr, "Warning: File version mismatch (expected: %d, actual: %d)\n",
                VERSION, header.version);
    }

    // Verify key (hash comparison)
    uint32_t input_key_hash = hash_key(key);
    if (header.key_hash != input_key_hash) {
        fprintf(stderr, "Error: Wrong password\n");
        fprintf(stderr, "Expected hash: 0x%08X, Input hash: 0x%08X\n",
                header.key_hash, input_key_hash);
        fclose(fin);
        return 1;
    }

    // Open output file
    FILE *fout = fopen(output, "wb");
    if (!fout) {
        perror("Failed to open output file");
        fclose(fin);
        return 1;
    }

    // Decrypt data
    unsigned char buffer[BUFFER_SIZE];
    size_t bytes_read;
    size_t key_len = strlen(key);
    size_t key_pos = 0;  // Track key position

    printf("Decrypting...\n");
    while ((bytes_read = fread(buffer, 1, BUFFER_SIZE, fin)) > 0) {
        xor_buffer(buffer, bytes_read, key, key_len, &key_pos);
        fwrite(buffer, 1, bytes_read, fout);

        // Progress indicator
        printf(".");
        fflush(stdout);
    }
    printf("\n");

    fclose(fin);
    fclose(fout);

    printf("Decryption complete: %s -> %s\n", input, output);
    printf("Original size: %llu bytes\n", (unsigned long long)header.original_size);

    return 0;
}

/**
 * Display encrypted file info
 *
 * @param filename Encrypted file path
 * @return 0 on success, 1 on failure
 */
int show_info(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Failed to open file");
        return 1;
    }

    FileHeader header;
    if (fread(&header, sizeof(FileHeader), 1, fp) != 1) {
        fprintf(stderr, "Error: Cannot read header\n");
        fclose(fp);
        return 1;
    }

    // Get total file size
    fseek(fp, 0, SEEK_END);
    long total_size = ftell(fp);
    fclose(fp);

    // Verify magic number
    if (memcmp(header.magic, MAGIC, 4) != 0) {
        printf("Not an encrypted file (no XENC magic number)\n");
        return 1;
    }

    // Print info
    printf("=== Encrypted File Info ===\n");
    printf("Magic number:   %.4s\n", header.magic);
    printf("Version:        %d\n", header.version);
    printf("Key hash:       0x%08X\n", header.key_hash);
    printf("Original size:  %llu bytes\n", (unsigned long long)header.original_size);
    printf("File size:      %ld bytes\n", total_size);
    printf("Header size:    %lu bytes\n", sizeof(FileHeader));
    printf("Encrypted data: %ld bytes\n", total_size - (long)sizeof(FileHeader));

    return 0;
}

/**
 * Main function
 */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Command dispatch
    if (strcmp(argv[1], "encrypt") == 0) {
        if (argc < 5) {
            print_usage(argv[0]);
            return 1;
        }
        return encrypt_file(argv[2], argv[3], argv[4]);
    }
    else if (strcmp(argv[1], "decrypt") == 0) {
        if (argc < 5) {
            print_usage(argv[0]);
            return 1;
        }
        return decrypt_file(argv[2], argv[3], argv[4]);
    }
    else if (strcmp(argv[1], "info") == 0) {
        if (argc < 3) {
            print_usage(argv[0]);
            return 1;
        }
        return show_info(argv[2]);
    }
    else {
        fprintf(stderr, "Error: Unknown command '%s'\n", argv[1]);
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
