# File Encryption Examples

XOR-based file encryption tool implementation examples

## File List

### 1. `simple_xor.c`
A simple demo showing the basic principles of XOR encryption

**Learning Content:**
- Reversibility of XOR operations (A ^ B ^ B = A)
- Bit operation basics
- Hexadecimal output
- Binary pattern output

**Compile and Run:**
```bash
gcc -Wall -Wextra -std=c11 simple_xor.c -o simple_xor
./simple_xor
```

**Sample Output:**
```
=== XOR Encryption Demo ===

Original message: Hello, World!
Original (hex):   48 65 6C 6C 6F 2C 20 57 6F 72 6C 64 21

XOR operation for first character 'H' ^ 'K':
  'H' = 72 (0b01001000)
  'K' = 75 (0b01001011)
  XOR = 3 (0b00000011)

Encryption complete!
Decrypted result: Hello, World!
```

---

### 2. `file_encrypt.c`
A practical file encryption tool (basic version)

**Learning Content:**
- Byte-level file I/O (`fread`, `fwrite`)
- Command-line argument parsing (`argc`, `argv`)
- Buffered file processing (4KB buffer)
- Error handling (`perror`)
- Key cycling (modulo operation)

**Compile:**
```bash
gcc -Wall -Wextra -std=c11 file_encrypt.c -o file_encrypt
```

**Usage:**
```bash
# Encrypt
./file_encrypt -e input.txt output.enc mypassword

# Decrypt
./file_encrypt -d output.enc decrypted.txt mypassword
```

**Features:**
- Simple and intuitive interface
- Encryption/decryption use the same operation due to XOR properties
- Supports all file types (text, binary)

---

### 3. `file_encrypt_v2.c`
Enhanced file encryption tool (header + validation)

**Learning Content:**
- Struct usage (`FileHeader`)
- File magic number validation
- Hash function implementation (djb2 algorithm)
- Key validation (password verification)
- File metadata storage
- Progress display
- Fixed-width integer types (`uint8_t`, `uint32_t`, `uint64_t`)

**Compile:**
```bash
gcc -Wall -Wextra -std=c11 file_encrypt_v2.c -o file_encrypt_v2
```

**Usage:**
```bash
# Encrypt
./file_encrypt_v2 encrypt secret.txt secret.enc mypassword

# Check file info
./file_encrypt_v2 info secret.enc

# Decrypt
./file_encrypt_v2 decrypt secret.enc decrypted.txt mypassword
```

**File Header Structure:**
```
┌─────────────────────────────────────────┐
│  Magic Number (4 bytes): "XENC"         │
│  Version (1 byte): 1                    │
│  Key Hash (4 bytes): for validation     │
│  Original Size (8 bytes): original size │
├─────────────────────────────────────────┤
│         Encrypted Data                  │
└─────────────────────────────────────────┘
```

**Sample Output:**
```bash
$ ./file_encrypt_v2 encrypt test.txt test.enc mypass
Encrypting...
.
Encryption complete: test.txt -> test.enc
Original size: 38 bytes
Key hash: 0x6CBFD0D9

$ ./file_encrypt_v2 info test.enc
=== Encrypted File Info ===
Magic number: XENC
Version: 1
Key hash: 0x6CBFD0D9
Original size: 38 bytes
File size: 62 bytes
Header size: 24 bytes
Encrypted data: 38 bytes

$ ./file_encrypt_v2 decrypt test.enc out.txt wrongpass
Error: Wrong password
Expected hash: 0x6CBFD0D9, Input hash: 0x289A5245
```

---

## Core Concepts

### XOR Encryption Principle
```
Principle: A ^ K = C, C ^ K = A

Example:
  Original: 'H' (72) = 0b01001000
  Key:      'K' (75) = 0b01001011
  Cipher:   XOR      = 0b00000011 (3)
  Decrypt:  3 ^ 75   = 0b01001000 (72 = 'H')
```

### djb2 Hash Algorithm
```c
uint32_t hash = 5381;
while ((c = *str++)) {
    hash = hash * 33 + c;
}
```
- Fast and effective string hash
- Low collision probability
- Suitable for key validation

---

## Security Notice

⚠️ **For Educational Purposes Only**

These examples are created for learning purposes. For actual security needs:

1. **XOR Encryption Weaknesses:**
   - Repeated key usage exposes patterns
   - Vulnerable to known-plaintext attacks
   - Short keys are susceptible to brute-force attacks

2. **Recommendations for Real-World Use:**
   - AES-256 (symmetric encryption)
   - RSA (asymmetric encryption)
   - Use OpenSSL library
   - Key stretching (PBKDF2, bcrypt)
   - Add salt

3. **Limitations of This Implementation:**
   - No key stretching
   - No salt used
   - No integrity verification (MAC)
   - No replay attack prevention

---

## Test Examples

```bash
# 1. Basic test
echo "Hello, World!" > test.txt
./file_encrypt -e test.txt test.enc mykey
./file_encrypt -d test.enc out.txt mykey
diff test.txt out.txt  # Should be identical

# 2. Binary file test
./file_encrypt_v2 encrypt /bin/ls ls.enc strongpass
./file_encrypt_v2 decrypt ls.enc ls.dec strongpass
diff /bin/ls ls.dec  # Should be identical

# 3. Wrong key test
./file_encrypt_v2 decrypt test.enc wrong.txt wrongkey
# Output: Error: Wrong password

# 4. Large file test (10MB)
dd if=/dev/urandom of=large.bin bs=1M count=10
time ./file_encrypt_v2 encrypt large.bin large.enc mypass
./file_encrypt_v2 info large.enc
```

---

## Compiler Options Explained

```bash
gcc -Wall -Wextra -std=c11 file.c -o program
```

- `-Wall`: Enable basic warnings
- `-Wextra`: Enable additional warnings
- `-std=c11`: Use C11 standard
- `-o program`: Specify output file name

---

## Learning Checklist

- [ ] Understand reversibility of XOR operations
- [ ] Bit operator usage (`^`, `&`, `|`, `~`, `<<`, `>>`)
- [ ] Byte-level file I/O (`fread`, `fwrite`)
- [ ] Command-line argument parsing (`argc`, `argv`)
- [ ] File header design using structs
- [ ] Hash function implementation (djb2)
- [ ] Error handling and validation logic
- [ ] Efficient file processing through buffering

---

## References

- [C11 Standard](https://en.cppreference.com/w/c/11)
- [XOR Cipher - Wikipedia](https://en.wikipedia.org/wiki/XOR_cipher)
- [djb2 Hash Function](http://www.cse.yorku.ca/~oz/hash.html)
- [OpenSSL Documentation](https://www.openssl.org/docs/)

---

## Related Study Materials

- `/opt/projects/01_Personal/03_Study/content/ko/C_Programming/08_Project_File_Encryption.md`
- `/opt/projects/01_Personal/03_Study/content/ko/C_Programming/14_Bit_Operations.md`
