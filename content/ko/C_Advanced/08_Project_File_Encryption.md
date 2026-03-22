# 프로젝트: 파일 암호화 도구

**이전**: [프로젝트: 해시 테이블](./07_Project_Hash_Table.md) | **다음**: [프로세스 관리](./09_Process_Management.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 비트 연산자(AND, OR, XOR, NOT, 시프트)를 적용하고 진리표를 설명할 수 있다
2. XOR의 자기 역원 속성(`A ^ B ^ B == A`)과 이것이 대칭 암호화를 가능하게 하는 이유를 설명할 수 있다
3. `fread`, `fwrite`, `fgetc`, `fputc`를 바이너리 모드로 사용한 바이트 레벨 파일 처리를 구현할 수 있다
4. `argc`와 `argv`를 파싱하여 모드, 파일명, 키를 받는 명령줄 도구를 설계할 수 있다
5. 매직 넘버, 버전, 키 해시, 원본 크기 헤더를 갖춘 암호화 파일 형식을 구축할 수 있다
6. 복호화 시 키 검증을 위한 간단한 해시 함수(djb2)를 구현할 수 있다
7. XOR 암호화의 보안 한계를 식별하고 프로덕션 용도에 부적합한 이유를 설명할 수 있다

---

비트 연산은 C에서 수행할 수 있는 가장 낮은 수준의 계산입니다 -- 바이트 내의 개별 1과 0을 뒤집는 것입니다. 암호화, 압축, 네트워크 프로토콜, 하드웨어 드라이버가 모두 이에 의존한다는 것을 깨달을 때까지는 추상적으로 보일 수 있습니다. 이 프로젝트는 비밀번호로 모든 파일을 스크램블하고 복원할 수 있는 파일 암호화 도구를 구축하여 비트 연산자를 실용적으로 활용합니다.

## XOR 암호화 원리

### XOR (배타적 OR) 연산

```
A XOR B = C
C XOR B = A  <- 같은 키로 다시 XOR하면 원본 복원!

예시:
  01100001 (a = 97)
^ 00110000 (key = 48)
-----------
  01010001 (Q = 81)  암호화됨

  01010001 (Q = 81)
^ 00110000 (key = 48)
-----------
  01100001 (a = 97)  복호화됨!
```

### 속성

- `A ^ A = 0` (자기 자신과 XOR = 0)
- `A ^ 0 = A` (0과 XOR = 자기 자신)
- `(A ^ B) ^ B = A` (두 번 XOR = 원본)

---

## 단계 1: 비트 연산 이해

### C 비트 연산자

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

    // AND: 둘 다 1일 때 1
    printf("a & b = %d\n", a & b);   // 128

    // OR: 하나라도 1이면 1
    printf("a | b = %d\n", a | b);   // 254

    // XOR: 다르면 1
    printf("a ^ b = %d\n", a ^ b);   // 126

    // NOT: 비트 반전
    printf("~a    = %d\n", (unsigned char)~a);  // 53

    // 왼쪽 시프트: 2를 곱함
    printf("a << 1 = %d\n", a << 1);  // 148 (오버플로우)

    // 오른쪽 시프트: 2로 나눔
    printf("a >> 1 = %d\n", a >> 1);  // 101

    return 0;
}
```

### 비트 연산 진리표

| A | B | AND | OR | XOR |
|---|---|-----|----|----|
| 0 | 0 |  0  | 0  | 0  |
| 0 | 1 |  0  | 1  | 1  |
| 1 | 0 |  0  | 1  | 1  |
| 1 | 1 |  1  | 1  | 0  |

---

## 단계 2: 간단한 XOR 암호화

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
    char key = 'K';  // 간단한 단일 문자 키

    printf("원본: %s\n", message);

    // 암호화
    xor_encrypt(message, strlen(message), key);
    printf("암호화: ");
    for (int i = 0; message[i]; i++) {
        printf("%02X ", (unsigned char)message[i]);
    }
    printf("\n");

    // 복호화 (같은 키로 다시 XOR)
    xor_encrypt(message, strlen(message), key);
    printf("복호화: %s\n", message);

    return 0;
}
```

### 예제 출력

```
원본: Hello, World!
암호화: 03 2E 27 27 24 67 52 18 24 31 27 2F 48
복호화: Hello, World!
```

---

## 단계 3: 파일 암호화 도구

### 핵심 문법: 바이트 레벨 파일 처리

```c
// 바이트 단위 읽기/쓰기
FILE *fp = fopen("file.bin", "rb");

int byte;
while ((byte = fgetc(fp)) != EOF) {
    // 바이트 처리
}

fclose(fp);

// 바이트 쓰기
FILE *fp = fopen("file.bin", "wb");
fputc(encrypted_byte, fp);
fclose(fp);
```

### 핵심 문법: 명령줄 인자

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

// 함수 선언
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
        fprintf(stderr, "오류: 키는 비어있을 수 없습니다\n");
        return 1;
    }

    int result;
    if (strcmp(mode, "-e") == 0 || strcmp(mode, "--encrypt") == 0) {
        result = encrypt_file(input_file, output_file, key);
        if (result == 0) {
            printf("암호화 성공: %s -> %s\n", input_file, output_file);
        }
    } else if (strcmp(mode, "-d") == 0 || strcmp(mode, "--decrypt") == 0) {
        result = decrypt_file(input_file, output_file, key);
        if (result == 0) {
            printf("복호화 성공: %s -> %s\n", input_file, output_file);
        }
    } else {
        fprintf(stderr, "오류: 알 수 없는 모드 '%s'\n", mode);
        print_usage(argv[0]);
        return 1;
    }

    return result;
}

void print_usage(const char *program_name) {
    printf("파일 암호화 도구 (XOR)\n\n");
    printf("사용법:\n");
    printf("  %s -e <입력> <출력> <키>  파일 암호화\n", program_name);
    printf("  %s -d <입력> <출력> <키>  파일 복호화\n", program_name);
    printf("\n옵션:\n");
    printf("  -e, --encrypt  암호화 모드\n");
    printf("  -d, --decrypt  복호화 모드\n");
    printf("\n예시:\n");
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
        perror("입력 파일 열기 오류");
        return 1;
    }

    FILE *fout = fopen(output_file, "wb");
    if (fout == NULL) {
        perror("출력 파일 열기 오류");
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
    // XOR 암호화와 복호화는 동일
    return encrypt_file(input_file, output_file, key);
}
```

---

## 단계 4: 개선된 버전 (헤더 포함)

### 암호화 파일 형식

```
+-----------------------------------------+
|              파일 헤더                   |
+-----------------------------------------+
|  매직 넘버 (4 bytes): "XENC"            |
|  버전 (1 byte): 1                       |
|  키 해시 (4 bytes): 검증용              |
|  원본 크기 (8 bytes): 원래 파일 크기    |
+-----------------------------------------+
|              암호화된 데이터             |
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

// 파일 헤더 구조체
typedef struct {
    char magic[4];
    uint8_t version;
    uint32_t key_hash;
    uint64_t original_size;
} FileHeader;

// 간단한 해시 함수 (djb2)
uint32_t hash_key(const char *key) {
    uint32_t hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

void print_usage(const char *name) {
    printf("향상된 파일 암호화 도구 v2\n\n");
    printf("사용법:\n");
    printf("  %s encrypt <입력> <출력> <비밀번호>\n", name);
    printf("  %s decrypt <입력> <출력> <비밀번호>\n", name);
    printf("  %s info <암호화된_파일>\n", name);
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
        perror("입력 파일 열기 오류");
        return 1;
    }

    // 원본 파일 크기 가져오기
    fseek(fin, 0, SEEK_END);
    uint64_t file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);

    FILE *fout = fopen(output, "wb");
    if (!fout) {
        perror("출력 파일 열기 오류");
        fclose(fin);
        return 1;
    }

    // 헤더 쓰기
    FileHeader header;
    memcpy(header.magic, MAGIC, 4);
    header.version = VERSION;
    header.key_hash = hash_key(key);
    header.original_size = file_size;
    fwrite(&header, sizeof(FileHeader), 1, fout);

    // 데이터 암호화
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

    printf("암호화됨: %s -> %s\n", input, output);
    printf("원본 크기: %llu bytes\n", (unsigned long long)file_size);
    return 0;
}

int decrypt_file(const char *input, const char *output, const char *key) {
    FILE *fin = fopen(input, "rb");
    if (!fin) {
        perror("입력 파일 열기 오류");
        return 1;
    }

    // 헤더 읽기
    FileHeader header;
    if (fread(&header, sizeof(FileHeader), 1, fin) != 1) {
        fprintf(stderr, "오류: 잘못된 암호화 파일\n");
        fclose(fin);
        return 1;
    }

    // 매직 넘버 확인
    if (memcmp(header.magic, MAGIC, 4) != 0) {
        fprintf(stderr, "오류: 유효한 암호화 파일이 아님\n");
        fclose(fin);
        return 1;
    }

    // 키 확인
    if (header.key_hash != hash_key(key)) {
        fprintf(stderr, "오류: 잘못된 비밀번호\n");
        fclose(fin);
        return 1;
    }

    FILE *fout = fopen(output, "wb");
    if (!fout) {
        perror("출력 파일 열기 오류");
        fclose(fin);
        return 1;
    }

    // 데이터 복호화
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

    printf("복호화됨: %s -> %s\n", input, output);
    printf("원본 크기: %llu bytes\n", (unsigned long long)header.original_size);
    return 0;
}

int show_info(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("파일 열기 오류");
        return 1;
    }

    FileHeader header;
    if (fread(&header, sizeof(FileHeader), 1, fp) != 1) {
        fprintf(stderr, "오류: 헤더를 읽을 수 없음\n");
        fclose(fp);
        return 1;
    }

    fclose(fp);

    if (memcmp(header.magic, MAGIC, 4) != 0) {
        printf("암호화된 파일이 아님 (XENC 매직 없음)\n");
        return 1;
    }

    printf("=== 암호화 파일 정보 ===\n");
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
# 컴파일
gcc -Wall -Wextra -std=c11 file_encrypt_v2.c -o encrypt

# 테스트 파일 생성
echo "This is a secret message!" > secret.txt

# 암호화
./encrypt encrypt secret.txt secret.enc mypassword

# 파일 정보 확인
./encrypt info secret.enc

# 복호화
./encrypt decrypt secret.enc decrypted.txt mypassword

# 검증
cat decrypted.txt

# 잘못된 비밀번호로 시도
./encrypt decrypt secret.enc fail.txt wrongpassword
# 오류: 잘못된 비밀번호
```

---

## 예제 출력

```
$ ./encrypt encrypt secret.txt secret.enc mypassword
암호화됨: secret.txt -> secret.enc
원본 크기: 27 bytes

$ ./encrypt info secret.enc
=== 암호화 파일 정보 ===
Magic: XENC
Version: 1
Key Hash: 0x7C9E6D5A
Original Size: 27 bytes

$ ./encrypt decrypt secret.enc decrypted.txt mypassword
복호화됨: secret.enc -> decrypted.txt
원본 크기: 27 bytes

$ cat decrypted.txt
This is a secret message!
```

---

## 요약

| 개념 | 설명 |
|------|------|
| `^` (XOR) | 비트 배타적 OR 연산 |
| `&` (AND) | 비트 AND 연산 |
| `\|` (OR) | 비트 OR 연산 |
| `~` (NOT) | 비트 반전 |
| `<<`, `>>` | 비트 시프트 |
| `fgetc`, `fputc` | 바이트 레벨 파일 I/O |
| `argc`, `argv` | 명령줄 인자 |

---

## 경고

> **보안 경고**: XOR 암호화는 학습 목적으로만 사용하세요!
> - 같은 키를 재사용하면 패턴이 노출됨
> - 알려진 평문 공격에 취약
> - 실제 보안에는 AES, RSA 등을 사용하세요

---

## 연습문제

1. **진행률 표시**: 대용량 파일 처리 시 진행률 바 표시

2. **압축 후 암호화**: zlib으로 압축 후 암호화

3. **디렉토리 처리**: 폴더 내 모든 파일을 일괄 암호화

4. **암호화 알고리즘 선택**: XOR 외에 다른 간단한 암호화 옵션 추가 (예: 시저 암호, 비제네르 암호)

5. **무결성 검사**: 파일 헤더에 CRC32 또는 SHA-256 체크섬을 추가하여 복호화 후 데이터 손상을 감지하세요

---

## 다음 단계

[프로세스 관리](./09_Process_Management.md) -> UNIX 프로세스 생성과 관리를 탐구해봅시다!
