# Project 6: File Encryption Tool

**Previous**: [Project 5: Linked List](./07_Project_Linked_List.md) | **Next**: [Project 7: Stack and Queue](./09_Project_Stack_Queue.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply bitwise operators (AND, OR, XOR, NOT, shift) and explain their truth tables
2. Explain the self-inverse property of XOR (`A ^ B ^ B == A`) and why it enables symmetric encryption
3. Implement byte-level file processing using `fread`, `fwrite`, `fgetc`, and `fputc` with binary mode
4. Design a command-line tool that parses `argc` and `argv` to accept mode, filenames, and a key
5. Build an encrypted file format with a magic number, version, key hash, and original-size header
6. Implement a simple hash function (djb2) for key verification during decryption
7. Identify the security limitations of XOR encryption and explain why it is unsuitable for production use

---

Bit operations are the lowest level of computation you can do in C -- flipping individual ones and zeros inside a byte. They may seem abstract until you realize that encryption, compression, network protocols, and hardware drivers all depend on them. This project puts bitwise operators to practical use by building a file encryption tool that can scramble and restore any file with a password.

## XOR Encryption Principle

### XOR (Exclusive OR) Operation

```
A XOR B = C
C XOR B = A  <- XOR again with same key restores original!

Example:
  01100001 (a = 97)
^ 00110000 (key = 48)
-----------
  01010001 (Q = 81)  Encrypted

  01010001 (Q = 81)
^ 00110000 (key = 48)
-----------
  01100001 (a = 97)  Decrypted!
```

### Properties

- `A ^ A = 0` (XOR with self = 0)
- `A ^ 0 = A` (XOR with 0 = self)
- `(A ^ B) ^ B = A` (XOR twice = original)

---

## Step 1: Understanding Bit Operations

### C Bit Operators

```c
#include <stdio.h>

int main(void) {
    unsigned char a = 0b11001010;  // 202
    unsigned char b = 0b10110100;  // 180

    printf("a     = %d (0b", a);
    for (int i = 7; i >= 0; i--) printf("%d", (a >> i) & 1);
    printf(")\n");

    printf("b     = %d (0b", b);
    for (int i = 7; i >= 0; i--) printf("%d", (b >> i) & 1);
    printf(")\n\n");

    // AND: 1 if both are 1
    printf("a & b = %d\n", a & b);   // 128

    // OR: 1 if either is 1
    printf("a | b = %d\n", a | b);   // 254

    // XOR: 1 if different
    printf("a ^ b = %d\n", a ^ b);   // 126

    // NOT: bit inversion
    printf("~a    = %d\n", (unsigned char)~a);  // 53

    // Left shift: multiply by 2
    printf("a << 1 = %d\n", a << 1);  // 148 (overflow)

    // Right shift: divide by 2
    printf("a >> 1 = %d\n", a >> 1);  // 101

    return 0;
}
```

### Bit Operation Truth Table

| A | B | AND | OR | XOR |
|---|---|-----|----|----|
| 0 | 0 |  0  | 0  | 0  |
| 0 | 1 |  0  | 1  | 1  |
| 1 | 0 |  0  | 1  | 1  |
| 1 | 1 |  1  | 1  | 0  |

---

## Step 2: Simple XOR Encryption

```c
// simple_xor.c
#include <stdio.h>
#include <string.h>

void xor_encrypt(char *data, int len, char key) {
    for (int i = 0; i < len; i++) {
        data[i] ^= key;
    }
}

int main(void) {
    char message[] = "Hello, World!";
    char key = 'K';  // Simple single character key

    printf("Original: %s\n", message);

    // Encrypt
    xor_encrypt(message, strlen(message), key);
    printf("Encrypted: ");
    for (int i = 0; message[i]; i++) {
        printf("%02X ", (unsigned char)message[i]);
    }
    printf("\n");

    // Decrypt (XOR again with same key)
    xor_encrypt(message, strlen(message), key);
    printf("Decrypted: %s\n", message);

    return 0;
}
```

### Example Output

```
Original: Hello, World!
Encrypted: 03 2E 27 27 24 67 52 18 24 31 27 2F 48
Decrypted: Hello, World!
```

---

## Step 3: File Encryption Tool

### Core Syntax: Byte-Level File Processing

```c
// Byte-level read/write
FILE *fp = fopen("file.bin", "rb");

int byte;
while ((byte = fgetc(fp)) != EOF) {
    // process byte
}

fclose(fp);

// Byte write
FILE *fp = fopen("file.bin", "wb");
fputc(encrypted_byte, fp);
fclose(fp);
```

### Core Syntax: Command-Line Arguments

```c
// ./program arg1 arg2
// argc = 3
// argv[0] = "./program"
// argv[1] = "arg1"
// argv[2] = "arg2"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <argument>\n", argv[0]);
        return 1;
    }

    printf("First argument: %s\n", argv[1]);
    return 0;
}
```

### File Encryption Program

```c
// file_encrypt.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 4096

// Function declarations
void print_usage(const char *program_name);
int encrypt_file(const char *input_file, const char *output_file, const char *key);
int decrypt_file(const char *input_file, const char *output_file, const char *key);
void xor_buffer(unsigned char *buffer, int len, const char *key, int key_len);

int main(int argc, char *argv[]) {
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    const char *mode = argv[1];
    const char *input_file = argv[2];
    const char *output_file = argv[3];
    const char *key = argv[4];

    if (strlen(key) == 0) {
        fprintf(stderr, "Error: Key cannot be empty\n");
        return 1;
    }

    int result;
    if (strcmp(mode, "-e") == 0 || strcmp(mode, "--encrypt") == 0) {
        result = encrypt_file(input_file, output_file, key);
        if (result == 0) {
            printf("Encryption successful: %s -> %s\n", input_file, output_file);
        }
    } else if (strcmp(mode, "-d") == 0 || strcmp(mode, "--decrypt") == 0) {
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

void print_usage(const char *program_name) {
    printf("File Encryption Tool (XOR)\n\n");
    printf("Usage:\n");
    printf("  %s -e <input> <output> <key>  Encrypt file\n", program_name);
    printf("  %s -d <input> <output> <key>  Decrypt file\n", program_name);
    printf("\nOptions:\n");
    printf("  -e, --encrypt  Encrypt mode\n");
    printf("  -d, --decrypt  Decrypt mode\n");
    printf("\nExample:\n");
    printf("  %s -e secret.txt secret.enc mypassword\n", program_name);
    printf("  %s -d secret.enc secret.txt mypassword\n", program_name);
}

void xor_buffer(unsigned char *buffer, int len, const char *key, int key_len) {
    for (int i = 0; i < len; i++) {
        buffer[i] ^= key[i % key_len];
    }
}

int encrypt_file(const char *input_file, const char *output_file, const char *key) {
    FILE *fin = fopen(input_file, "rb");
    if (fin == NULL) {
        perror("Error opening input file");
        return 1;
    }

    FILE *fout = fopen(output_file, "wb");
    if (fout == NULL) {
        perror("Error opening output file");
        fclose(fin);
        return 1;
    }

    unsigned char buffer[BUFFER_SIZE];
    int key_len = strlen(key);
    size_t bytes_read;

    while ((bytes_read = fread(buffer, 1, BUFFER_SIZE, fin)) > 0) {
        xor_buffer(buffer, bytes_read, key, key_len);
        fwrite(buffer, 1, bytes_read, fout);
    }

    fclose(fin);
    fclose(fout);
    return 0;
}

int decrypt_file(const char *input_file, const char *output_file, const char *key) {
    // XOR encryption and decryption are identical
    return encrypt_file(input_file, output_file, key);
}
```

---

## Step 4: Improved Version (With Header)

### Encrypted File Format

```
+-----------------------------------------+
|              File Header                |
+-----------------------------------------+
|  Magic Number (4 bytes): "XENC"         |
|  Version (1 byte): 1                    |
|  Key Hash (4 bytes): for verification   |
|  Original Size (8 bytes): original size |
+-----------------------------------------+
|              Encrypted Data             |
+-----------------------------------------+
```

### Improved Code

```c
// file_encrypt_v2.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define MAGIC "XENC"
#define VERSION 1
#define BUFFER_SIZE 4096
#define HEADER_SIZE 17

// File header struct
typedef struct {
    char magic[4];
    uint8_t version;
    uint32_t key_hash;
    uint64_t original_size;
} FileHeader;

// Simple hash function (djb2)
uint32_t hash_key(const char *key) {
    uint32_t hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

void print_usage(const char *name) {
    printf("Enhanced File Encryption Tool v2\n\n");
    printf("Usage:\n");
    printf("  %s encrypt <input> <output> <password>\n", name);
    printf("  %s decrypt <input> <output> <password>\n", name);
    printf("  %s info <encrypted_file>\n", name);
}

void xor_buffer(unsigned char *buf, size_t len, const char *key, size_t key_len, size_t *pos) {
    for (size_t i = 0; i < len; i++) {
        buf[i] ^= key[*pos % key_len];
        (*pos)++;
    }
}

int encrypt_file(const char *input, const char *output, const char *key) {
    FILE *fin = fopen(input, "rb");
    if (!fin) {
        perror("Error opening input file");
        return 1;
    }

    // Get original file size
    fseek(fin, 0, SEEK_END);
    uint64_t file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);

    FILE *fout = fopen(output, "wb");
    if (!fout) {
        perror("Error opening output file");
        fclose(fin);
        return 1;
    }

    // Write header
    FileHeader header;
    memcpy(header.magic, MAGIC, 4);
    header.version = VERSION;
    header.key_hash = hash_key(key);
    header.original_size = file_size;
    fwrite(&header, sizeof(FileHeader), 1, fout);

    // Encrypt data
    unsigned char buffer[BUFFER_SIZE];
    size_t bytes_read;
    size_t key_len = strlen(key);
    size_t key_pos = 0;

    while ((bytes_read = fread(buffer, 1, BUFFER_SIZE, fin)) > 0) {
        xor_buffer(buffer, bytes_read, key, key_len, &key_pos);
        fwrite(buffer, 1, bytes_read, fout);
    }

    fclose(fin);
    fclose(fout);

    printf("Encrypted: %s -> %s\n", input, output);
    printf("Original size: %llu bytes\n", (unsigned long long)file_size);
    return 0;
}

int decrypt_file(const char *input, const char *output, const char *key) {
    FILE *fin = fopen(input, "rb");
    if (!fin) {
        perror("Error opening input file");
        return 1;
    }

    // Read header
    FileHeader header;
    if (fread(&header, sizeof(FileHeader), 1, fin) != 1) {
        fprintf(stderr, "Error: Invalid encrypted file\n");
        fclose(fin);
        return 1;
    }

    // Verify magic number
    if (memcmp(header.magic, MAGIC, 4) != 0) {
        fprintf(stderr, "Error: Not a valid encrypted file\n");
        fclose(fin);
        return 1;
    }

    // Verify key
    if (header.key_hash != hash_key(key)) {
        fprintf(stderr, "Error: Wrong password\n");
        fclose(fin);
        return 1;
    }

    FILE *fout = fopen(output, "wb");
    if (!fout) {
        perror("Error opening output file");
        fclose(fin);
        return 1;
    }

    // Decrypt data
    unsigned char buffer[BUFFER_SIZE];
    size_t bytes_read;
    size_t key_len = strlen(key);
    size_t key_pos = 0;

    while ((bytes_read = fread(buffer, 1, BUFFER_SIZE, fin)) > 0) {
        xor_buffer(buffer, bytes_read, key, key_len, &key_pos);
        fwrite(buffer, 1, bytes_read, fout);
    }

    fclose(fin);
    fclose(fout);

    printf("Decrypted: %s -> %s\n", input, output);
    printf("Original size: %llu bytes\n", (unsigned long long)header.original_size);
    return 0;
}

int show_info(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("Error opening file");
        return 1;
    }

    FileHeader header;
    if (fread(&header, sizeof(FileHeader), 1, fp) != 1) {
        fprintf(stderr, "Error: Cannot read header\n");
        fclose(fp);
        return 1;
    }

    fclose(fp);

    if (memcmp(header.magic, MAGIC, 4) != 0) {
        printf("Not an encrypted file (no XENC magic)\n");
        return 1;
    }

    printf("=== Encrypted File Info ===\n");
    printf("Magic: %.4s\n", header.magic);
    printf("Version: %d\n", header.version);
    printf("Key Hash: 0x%08X\n", header.key_hash);
    printf("Original Size: %llu bytes\n", (unsigned long long)header.original_size);

    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

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
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
```

---

## Compile and Run

```bash
# Compile
gcc -Wall -Wextra -std=c11 file_encrypt_v2.c -o encrypt

# Create test file
echo "This is a secret message!" > secret.txt

# Encrypt
./encrypt encrypt secret.txt secret.enc mypassword

# View file info
./encrypt info secret.enc

# Decrypt
./encrypt decrypt secret.enc decrypted.txt mypassword

# Verify
cat decrypted.txt

# Try with wrong password
./encrypt decrypt secret.enc fail.txt wrongpassword
# Error: Wrong password
```

---

## Example Output

```
$ ./encrypt encrypt secret.txt secret.enc mypassword
Encrypted: secret.txt -> secret.enc
Original size: 27 bytes

$ ./encrypt info secret.enc
=== Encrypted File Info ===
Magic: XENC
Version: 1
Key Hash: 0x7C9E6D5A
Original Size: 27 bytes

$ ./encrypt decrypt secret.enc decrypted.txt mypassword
Decrypted: secret.enc -> decrypted.txt
Original size: 27 bytes

$ cat decrypted.txt
This is a secret message!
```

---

## Summary

| Concept | Description |
|---------|-------------|
| `^` (XOR) | Bit XOR operation |
| `&` (AND) | Bit AND operation |
| `\|` (OR) | Bit OR operation |
| `~` (NOT) | Bit inversion |
| `<<`, `>>` | Bit shift |
| `fgetc`, `fputc` | Byte-level file I/O |
| `argc`, `argv` | Command-line arguments |

---

## Warning

> **Security Warning**: Use XOR encryption for learning purposes only!
> - Pattern exposure when same key is reused
> - Vulnerable to known plaintext attacks
> - Use AES, RSA, etc. for real security

---

## Exercises

1. **Progress display**: Show progress bar for large file processing

2. **Compress then encrypt**: Compress with zlib before encrypting

3. **Directory processing**: Batch encrypt all files in a folder

4. **Encryption algorithm selection**: Add other simple encryption options besides XOR

---

## Learning Complete!

You have completed the project-based C language learning.

### Projects Summary

| Project | Core Concepts |
|---------|---------------|
| Calculator | Functions, switch, input handling |
| Number Guessing | Loops, random numbers, game logic |
| Address Book | Structs, file I/O |
| Dynamic Array | malloc, realloc, free |
| Linked List | Pointers, data structures |
| File Encryption | Bit operations, byte processing |

### Next Steps

1. **Advanced Data Structures**: Trees, hash tables, graphs
2. **Algorithms**: Sorting, searching, recursion
3. **Systems Programming**: Processes, threads, sockets
4. **Embedded C**: Microcontroller programming
