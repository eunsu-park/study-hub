// file_encrypt.c
// File encryption tool (XOR-based)
// Learning objective: Byte-level file I/O, command-line argument handling

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 4096  // 4KB buffer size

// Function declarations
void print_usage(const char *program_name);
int encrypt_file(const char *input_file, const char *output_file, const char *key);
int decrypt_file(const char *input_file, const char *output_file, const char *key);
void xor_buffer(unsigned char *buffer, int len, const char *key, int key_len);

/**
 * Main function - Command-line argument parsing
 *
 * @param argc Argument count
 * @param argv Argument array
 *
 * Usage:
 *   ./file_encrypt -e input.txt output.enc mypassword
 *   ./file_encrypt -d output.enc decrypted.txt mypassword
 */
int main(int argc, char *argv[]) {
    // Check argument count (program + mode + input + output + key = 5)
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    const char *mode = argv[1];         // -e or -d
    const char *input_file = argv[2];   // Input filename
    const char *output_file = argv[3];  // Output filename
    const char *key = argv[4];          // Encryption key

    // Validate non-empty key
    if (strlen(key) == 0) {
        fprintf(stderr, "Error: Key cannot be empty\n");
        return 1;
    }

    int result;
    if (strcmp(mode, "-e") == 0 || strcmp(mode, "--encrypt") == 0) {
        // Encryption mode
        result = encrypt_file(input_file, output_file, key);
        if (result == 0) {
            printf("Encryption successful: %s -> %s\n", input_file, output_file);
        }
    } else if (strcmp(mode, "-d") == 0 || strcmp(mode, "--decrypt") == 0) {
        // Decryption mode
        result = decrypt_file(input_file, output_file, key);
        if (result == 0) {
            printf("Decryption successful: %s -> %s\n", input_file, output_file);
        }
    } else {
        fprintf(stderr, "Error: Unknown mode '%s'\n", mode);
        print_usage(argv[0]);
        return 1;
    }

    return result;
}

/**
 * Print usage
 *
 * @param program_name Program name (argv[0])
 */
void print_usage(const char *program_name) {
    printf("File Encryption Tool (XOR)\n\n");
    printf("Usage:\n");
    printf("  %s -e <input> <output> <key>  Encrypt file\n", program_name);
    printf("  %s -d <input> <output> <key>  Decrypt file\n", program_name);
    printf("\nOptions:\n");
    printf("  -e, --encrypt  Encryption mode\n");
    printf("  -d, --decrypt  Decryption mode\n");
    printf("\nExamples:\n");
    printf("  %s -e secret.txt secret.enc mypassword\n", program_name);
    printf("  %s -d secret.enc secret.txt mypassword\n", program_name);
}

/**
 * XOR encrypt/decrypt buffer
 *
 * @param buffer Buffer to process
 * @param len Buffer length
 * @param key Encryption key (string)
 * @param key_len Key length
 *
 * If the key is shorter than the data, it is reused cyclically (modulo operation)
 */
void xor_buffer(unsigned char *buffer, int len, const char *key, int key_len) {
    for (int i = 0; i < len; i++) {
        // Apply XOR cyclically with the key
        buffer[i] ^= key[i % key_len];
    }
}

/**
 * File encryption function
 *
 * @param input_file Input file path
 * @param output_file Output file path
 * @param key Encryption key
 * @return 0 on success, 1 on failure
 */
int encrypt_file(const char *input_file, const char *output_file, const char *key) {
    // Open input file (binary read mode)
    FILE *fin = fopen(input_file, "rb");
    if (fin == NULL) {
        perror("Failed to open input file");
        return 1;
    }

    // Open output file (binary write mode)
    FILE *fout = fopen(output_file, "wb");
    if (fout == NULL) {
        perror("Failed to open output file");
        fclose(fin);
        return 1;
    }

    // Prepare buffer
    unsigned char buffer[BUFFER_SIZE];
    int key_len = strlen(key);
    size_t bytes_read;

    // Read and process in buffer-sized chunks
    while ((bytes_read = fread(buffer, 1, BUFFER_SIZE, fin)) > 0) {
        // Apply XOR encryption
        xor_buffer(buffer, bytes_read, key, key_len);

        // Write encrypted data
        fwrite(buffer, 1, bytes_read, fout);
    }

    // Close files
    fclose(fin);
    fclose(fout);

    return 0;
}

/**
 * File decryption function
 *
 * @param input_file Input file path
 * @param output_file Output file path
 * @param key Encryption key
 * @return 0 on success, 1 on failure
 *
 * Due to the nature of XOR encryption, encryption and decryption are the same operation
 * (A ^ K = B, B ^ K = A)
 */
int decrypt_file(const char *input_file, const char *output_file, const char *key) {
    // XOR encryption uses the same operation for both encryption and decryption
    return encrypt_file(input_file, output_file, key);
}
