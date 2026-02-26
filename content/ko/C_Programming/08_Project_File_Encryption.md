# 프로젝트 6: 파일 암호화 도구

**이전**: [프로젝트 5: 연결 리스트](./07_Project_Linked_List.md) | **다음**: [프로젝트 7: 스택과 큐](./09_Project_Stack_Queue.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 비트 연산자(AND, OR, XOR, NOT, 시프트)를 적용하고 진리표(truth table)를 설명할 수 있습니다
2. XOR의 자기 역원 속성(`A ^ B ^ B == A`)과 그것이 대칭 암호화(symmetric encryption)를 가능하게 하는 이유를 설명할 수 있습니다
3. 바이너리 모드에서 `fread`, `fwrite`, `fgetc`, `fputc`를 사용하여 바이트 단위 파일 처리(byte-level file processing)를 구현할 수 있습니다
4. 모드, 파일명, 키를 받기 위해 `argc`와 `argv`를 파싱하는 명령줄 도구(command-line tool)를 설계할 수 있습니다
5. 매직 넘버(magic number), 버전, 키 해시, 원본 크기 헤더를 포함한 암호화 파일 형식을 구현할 수 있습니다
6. 복호화 시 키 검증을 위한 간단한 해시 함수(djb2)를 구현할 수 있습니다
7. XOR 암호화의 보안 한계를 파악하고 실제 운영 환경에 적합하지 않은 이유를 설명할 수 있습니다

---

비트 연산(bit operations)은 C에서 수행할 수 있는 가장 낮은 수준의 연산으로, 바이트 안의 개별 0과 1을 뒤집습니다. 암호화, 압축, 네트워크 프로토콜, 하드웨어 드라이버가 모두 비트 연산에 의존하고 있다는 것을 알기 전까지는 추상적으로 느껴질 수 있습니다. 이 프로젝트는 비트 연산자를 실용적으로 활용하여, 비밀번호로 모든 파일을 암호화하고 복원할 수 있는 파일 암호화 도구를 구현합니다.

## XOR 암호화 원리

### XOR (Exclusive OR) 연산

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

### 특성

- `A ^ A = 0` (자기 자신과 XOR = 0)
- `A ^ 0 = A` (0과 XOR = 자신)
- `(A ^ B) ^ B = A` (두 번 XOR = 원본)

---

## 1단계: 비트 연산 이해

### C의 비트 연산자

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

### 비트 연산 진리표(Truth Table)

| A | B | AND | OR | XOR |
|---|---|-----|----|----|
| 0 | 0 |  0  | 0  | 0  |
| 0 | 1 |  0  | 1  | 1  |
| 1 | 0 |  0  | 1  | 1  |
| 1 | 1 |  1  | 1  | 0  |

---

## 2단계: 간단한 XOR 암호화

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

### 실행 결과

```
Original: Hello, World!
Encrypted: 03 2E 27 27 24 67 52 18 24 31 27 2F 48
Decrypted: Hello, World!
```

---

## 3단계: 파일 암호화 도구

### 핵심 문법: 바이트 단위 파일 처리

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

### 핵심 문법: 명령줄 인자(Command-Line Arguments)

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

### 파일 암호화 프로그램

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

## 4단계: 개선된 버전 (헤더 추가)

### 암호화 파일 형식

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

### 개선된 코드

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

## 컴파일 및 실행

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

## 실행 결과

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

## 배운 내용 정리

| 개념 | 설명 |
|------|------|
| `^` (XOR) | 비트 XOR 연산 |
| `&` (AND) | 비트 AND 연산 |
| `\|` (OR) | 비트 OR 연산 |
| `~` (NOT) | 비트 반전 |
| `<<`, `>>` | 비트 시프트 |
| `fgetc`, `fputc` | 바이트 단위 파일 I/O |
| `argc`, `argv` | 명령줄 인자 |

---

## 주의사항

> **보안 경고**: XOR 암호화는 학습 목적으로만 사용하세요!
> - 같은 키 반복 사용 시 패턴 노출
> - 알려진 평문 공격(known plaintext attack)에 취약
> - 실제 보안에는 AES, RSA 등 사용

---

## 연습 문제

1. **진행률 표시**: 대용량 파일 처리 시 진행률 바 표시

2. **압축 후 암호화**: zlib으로 압축 후 암호화

3. **디렉토리 처리**: 폴더 내 모든 파일 일괄 암호화

4. **암호화 알고리즘 선택**: XOR 외에 다른 간단한 암호화 옵션 추가

---

## 학습 완료!

C 언어 프로젝트 기반 학습을 완료했습니다.

### 학습한 프로젝트 정리

| 프로젝트 | 핵심 개념 |
|---------|----------|
| 계산기 | 함수, switch, 입력 처리 |
| 숫자 맞추기 | 반복문, 난수, 게임 로직 |
| 주소록 | 구조체, 파일 I/O |
| 동적 배열 | malloc, realloc, free |
| 연결 리스트 | 포인터, 자료구조 |
| 파일 암호화 | 비트 연산, 바이트 처리 |

### 다음 학습 추천

1. **고급 자료구조**: 트리, 해시 테이블, 그래프
2. **알고리즘**: 정렬, 탐색, 재귀
3. **시스템 프로그래밍**: 프로세스, 스레드, 소켓
4. **임베디드 C**: 마이크로컨트롤러 프로그래밍

---

**이전**: [프로젝트 5: 연결 리스트](./07_Project_Linked_List.md) | **다음**: [프로젝트 7: 스택과 큐](./09_Project_Stack_Queue.md)
