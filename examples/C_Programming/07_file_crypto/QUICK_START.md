# Quick Start Guide - File Encryption Examples

## Build & Run (Quick Start)

```bash
# Compile all programs
make

# Run all tests
make test

# Clean up
make clean
```

## Individual Execution Examples

### 1. View XOR Demo
```bash
gcc -Wall -Wextra -std=c11 simple_xor.c -o simple_xor
./simple_xor
```

### 2. File Encryption (Basic Version)
```bash
gcc -Wall -Wextra -std=c11 file_encrypt.c -o file_encrypt

# Usage
./file_encrypt -e <input> <output> <key>    # Encrypt
./file_encrypt -d <input> <output> <key>    # Decrypt

# Example
echo "Secret data" > secret.txt
./file_encrypt -e secret.txt secret.enc mypassword
./file_encrypt -d secret.enc decrypted.txt mypassword
cat decrypted.txt
```

### 3. File Encryption v2 (Header + Validation)
```bash
gcc -Wall -Wextra -std=c11 file_encrypt_v2.c -o file_encrypt_v2

# Usage
./file_encrypt_v2 encrypt <input> <output> <password>
./file_encrypt_v2 decrypt <input> <output> <password>
./file_encrypt_v2 info <encrypted_file>

# Example
echo "Top secret!" > data.txt
./file_encrypt_v2 encrypt data.txt data.enc strongpass
./file_encrypt_v2 info data.enc
./file_encrypt_v2 decrypt data.enc restored.txt strongpass
diff data.txt restored.txt  # Should be identical
```

## Key Differences

| Feature | file_encrypt | file_encrypt_v2 |
|---------|--------------|-----------------|
| Interface | `-e` / `-d` options | `encrypt` / `decrypt` commands |
| File Header | ✗ | ✓ (XENC magic number) |
| Key Validation | ✗ | ✓ (hash comparison) |
| Metadata | ✗ | ✓ (original size, version) |
| File Info | ✗ | ✓ (`info` command) |
| Progress Display | ✗ | ✓ |

## Recommended Learning Order

1. `simple_xor.c` - Understand basic XOR principles
2. `file_encrypt.c` - File I/O and basic structure
3. `file_encrypt_v2.c` - Headers, validation, and advanced features

## Security Warning

⚠️ **For Learning Only - Do Not Use for Actual Security**

For real encryption needs:
- `openssl enc -aes-256-cbc -in file -out file.enc`
- GPG (GNU Privacy Guard)
- libsodium library
