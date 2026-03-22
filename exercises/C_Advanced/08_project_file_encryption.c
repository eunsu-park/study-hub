/*
 * Exercises for Lesson 08: Project File Encryption
 * Topic: C_Advanced
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex08 08_project_file_encryption.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* === Exercise 1: Caesar Cipher === */
/* Problem: Implement encryption and decryption with configurable shift. */

void caesar_encrypt(char *text, int shift) {
    /*
     * Caesar cipher: shift each letter by 'shift' positions.
     * - Only affects alphabetic characters
     * - Wraps around: Z + 1 = A
     * - Preserves case
     *
     * Security note: Caesar cipher has only 25 possible keys and is
     * trivially broken by brute force or frequency analysis. It is
     * a toy cipher for learning, NOT for actual security.
     */
    shift = ((shift % 26) + 26) % 26; /* Normalize to [0, 25] */
    for (int i = 0; text[i]; i++) {
        if (isupper((unsigned char)text[i])) {
            text[i] = (char)('A' + (text[i] - 'A' + shift) % 26);
        } else if (islower((unsigned char)text[i])) {
            text[i] = (char)('a' + (text[i] - 'a' + shift) % 26);
        }
        /* Non-alpha characters pass through unchanged */
    }
}

void caesar_decrypt(char *text, int shift) {
    /* Decryption is encryption with the negative shift */
    caesar_encrypt(text, -shift);
}

void exercise_1(void) {
    printf("=== Exercise 1: Caesar Cipher ===\n");

    /* Test with various shifts */
    struct {
        const char *plaintext;
        int shift;
    } tests[] = {
        {"Hello, World!", 3},
        {"Attack at dawn", 13},  /* ROT13 */
        {"ZEBRA", 1},
        {"abc XYZ 123", 25},
        {"Test", 0},
        {"Negative shift", -5},
    };
    int n_tests = (int)(sizeof(tests) / sizeof(tests[0]));

    for (int i = 0; i < n_tests; i++) {
        char buf[128];
        strncpy(buf, tests[i].plaintext, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';

        printf("\nPlaintext:  \"%s\" (shift=%d)\n", buf, tests[i].shift);
        caesar_encrypt(buf, tests[i].shift);
        printf("Encrypted:  \"%s\"\n", buf);
        caesar_decrypt(buf, tests[i].shift);
        printf("Decrypted:  \"%s\"\n", buf);
    }

    /* Brute force attack demonstration */
    printf("\nBrute-force attack on 'Khoor':\n");
    for (int shift = 1; shift <= 25; shift++) {
        char attack[] = "Khoor";
        caesar_decrypt(attack, shift);
        printf("  shift=%2d: %s%s\n", shift, attack,
               strcmp(attack, "Hello") == 0 ? "  <-- found!" : "");
    }
}

/* === Exercise 2: XOR Encryption === */
/* Problem: Implement XOR cipher with a multi-byte key. */

void xor_encrypt(unsigned char *data, size_t len,
                 const unsigned char *key, size_t key_len) {
    /*
     * XOR cipher properties:
     * - Symmetric: encrypt and decrypt are the same operation
     * - data ^ key ^ key = data (self-inverse)
     * - With a truly random key as long as the message (one-time pad),
     *   XOR encryption is theoretically unbreakable
     * - With a short repeating key, vulnerable to Kasiski examination
     *
     * Key reuse is the critical weakness: if same key encrypts two
     * messages, XORing the ciphertexts cancels the key:
     *   (m1 ^ k) ^ (m2 ^ k) = m1 ^ m2
     */
    for (size_t i = 0; i < len; i++) {
        data[i] ^= key[i % key_len];
    }
}

void exercise_2(void) {
    printf("\n=== Exercise 2: XOR Encryption ===\n");

    /* Single-byte key */
    printf("Single-byte XOR (key=0x42):\n");
    unsigned char msg1[] = "Hello, World!";
    size_t len1 = strlen((char *)msg1);
    printf("  Original:  %s\n", msg1);

    xor_encrypt(msg1, len1, (unsigned char *)"\x42", 1);
    printf("  Encrypted: ");
    for (size_t i = 0; i < len1; i++) printf("%02X ", msg1[i]);
    printf("\n");

    xor_encrypt(msg1, len1, (unsigned char *)"\x42", 1);
    printf("  Decrypted: %s\n", msg1);

    /* Multi-byte key */
    printf("\nMulti-byte XOR (key='SECRET'):\n");
    unsigned char msg2[] = "This is a secret message!";
    size_t len2 = strlen((char *)msg2);
    const unsigned char key[] = "SECRET";
    size_t key_len = strlen((char *)key);

    printf("  Original:  %s\n", msg2);

    xor_encrypt(msg2, len2, key, key_len);
    printf("  Encrypted: ");
    for (size_t i = 0; i < len2; i++) printf("%02X ", msg2[i]);
    printf("\n");

    xor_encrypt(msg2, len2, key, key_len);
    printf("  Decrypted: %s\n", msg2);

    /* Demonstrate XOR properties */
    printf("\nXOR truth table:\n");
    printf("  A  B  A^B\n");
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            printf("  %d  %d   %d\n", a, b, a ^ b);
        }
    }
    printf("Key property: A ^ B ^ B = A (self-inverse)\n");
}

/* === Exercise 3: File Statistics === */
/* Problem: Analyze a file's byte frequency distribution. */

typedef struct {
    size_t total_bytes;
    size_t byte_freq[256];
    size_t printable;
    size_t whitespace;
    size_t control;
    double entropy;
} FileStats;

FileStats analyze_buffer(const unsigned char *data, size_t len) {
    FileStats stats = {0};
    stats.total_bytes = len;

    for (size_t i = 0; i < len; i++) {
        stats.byte_freq[data[i]]++;
        if (isprint(data[i])) stats.printable++;
        else if (isspace(data[i])) stats.whitespace++;
        else stats.control++;
    }

    /*
     * Shannon entropy: measures information content / randomness.
     * H = -sum(p_i * log2(p_i)) for each byte value i
     *
     * - English text: ~4.0-5.0 bits/byte
     * - Compressed/encrypted data: ~7.5-8.0 bits/byte
     * - Random data: ~8.0 bits/byte (maximum)
     * - All same byte: 0 bits/byte (minimum)
     */
    stats.entropy = 0;
    for (int i = 0; i < 256; i++) {
        if (stats.byte_freq[i] > 0) {
            double p = (double)stats.byte_freq[i] / (double)len;
            stats.entropy -= p * (log(p) / log(2.0));
        }
    }

    return stats;
}

void exercise_3(void) {
    printf("\n=== Exercise 3: File Statistics ===\n");

    /* Analyze English text */
    const char *text = "The quick brown fox jumps over the lazy dog. "
                       "This is a sample text for frequency analysis. "
                       "English text has characteristic letter frequencies.";
    FileStats ts = analyze_buffer((const unsigned char *)text, strlen(text));

    printf("Text analysis:\n");
    printf("  Total bytes:  %zu\n", ts.total_bytes);
    printf("  Printable:    %zu (%.1f%%)\n", ts.printable,
           100.0 * (double)ts.printable / (double)ts.total_bytes);
    printf("  Whitespace:   %zu\n", ts.whitespace);
    printf("  Entropy:      %.2f bits/byte\n", ts.entropy);

    /* Show top 5 most frequent bytes */
    printf("  Top 5 bytes:  ");
    for (int rank = 0; rank < 5; rank++) {
        size_t max_freq = 0;
        int max_byte = 0;
        for (int b = 0; b < 256; b++) {
            if (ts.byte_freq[b] > max_freq) {
                /* Skip already-printed bytes */
                int skip = 0;
                for (int r2 = 0; r2 < rank; r2++) {
                    (void)r2; /* Will check differently */
                }
                if (!skip) { max_freq = ts.byte_freq[b]; max_byte = b; }
            }
        }
        if (max_freq > 0) {
            printf("'%c'(%zu) ", isprint(max_byte) ? max_byte : '?', max_freq);
            ts.byte_freq[max_byte] = 0; /* Mark as printed */
        }
    }
    printf("\n");

    /* Analyze encrypted data (should have higher entropy) */
    unsigned char encrypted[150];
    memcpy(encrypted, text, strlen(text));
    xor_encrypt(encrypted, strlen(text), (unsigned char *)"KEY", 3);
    FileStats es = analyze_buffer(encrypted, strlen(text));

    printf("\nEncrypted analysis:\n");
    printf("  Entropy:      %.2f bits/byte (higher = more random)\n", es.entropy);
    printf("  Printable:    %zu (%.1f%%)\n", es.printable,
           100.0 * (double)es.printable / (double)es.total_bytes);
    printf("\nEntropy comparison: text=%.2f, encrypted=%.2f\n",
           analyze_buffer((const unsigned char *)text, strlen(text)).entropy,
           es.entropy);
}

/* === Exercise 4: Binary File Handling === */
/* Problem: Read and write binary files, handle endianness. */
void exercise_4(void) {
    printf("\n=== Exercise 4: Binary File Handling ===\n");

    /*
     * Binary vs text mode:
     * - Text mode: OS may translate newlines (\n <-> \r\n on Windows)
     * - Binary mode ("rb"/"wb"): no translation, byte-for-byte
     * - Always use binary mode for non-text data
     */

    /* Write binary data */
    const char *filename = "/tmp/binary_test.bin";
    FILE *fp = fopen(filename, "wb");
    if (!fp) { printf("Cannot create test file\n"); return; }

    /* Write a simple header + data structure */
    unsigned char magic[] = {0x89, 'T', 'E', 'S', 'T'};  /* Magic number */
    fwrite(magic, 1, sizeof(magic), fp);

    /* Write integers in known byte order */
    int values[] = {1, 256, 65536, 16777216};
    int n_values = (int)(sizeof(values) / sizeof(values[0]));
    fwrite(&n_values, sizeof(int), 1, fp);
    fwrite(values, sizeof(int), (size_t)n_values, fp);

    fclose(fp);
    printf("Wrote binary file: %s\n", filename);

    /* Read it back */
    fp = fopen(filename, "rb");
    if (!fp) { printf("Cannot read test file\n"); return; }

    /* Verify magic number */
    unsigned char read_magic[5];
    fread(read_magic, 1, 5, fp);
    printf("Magic: %02X %c%c%c%c\n", read_magic[0],
           read_magic[1], read_magic[2], read_magic[3], read_magic[4]);

    /* Read count and values */
    int count;
    fread(&count, sizeof(int), 1, fp);
    printf("Count: %d\n", count);

    int read_values[4];
    fread(read_values, sizeof(int), (size_t)count, fp);
    printf("Values: ");
    for (int i = 0; i < count; i++) {
        printf("%d ", read_values[i]);
    }
    printf("\n");

    fclose(fp);

    /* Show byte-level representation */
    printf("\nByte-level view of int 256 (0x00000100):\n");
    int val = 256;
    unsigned char *bytes = (unsigned char *)&val;
    printf("  Bytes: ");
    for (size_t i = 0; i < sizeof(int); i++) {
        printf("%02X ", bytes[i]);
    }
    printf("(%s endian)\n", bytes[0] == 0 ? "big" : "little");

    remove(filename);
}

/* === Exercise 5: Simple Checksum === */
/* Problem: Implement checksums for data integrity verification. */

unsigned char checksum_xor(const unsigned char *data, size_t len) {
    /* XOR all bytes together. Simple but weak -- can't detect swapped bytes. */
    unsigned char sum = 0;
    for (size_t i = 0; i < len; i++) sum ^= data[i];
    return sum;
}

unsigned short checksum_fletcher16(const unsigned char *data, size_t len) {
    /*
     * Fletcher-16: A position-dependent checksum.
     * Unlike simple XOR or sum, it detects byte reordering.
     *
     * sum1 accumulates the running sum of bytes.
     * sum2 accumulates the running sum of sum1 values.
     * This makes the checksum sensitive to byte order.
     *
     * Used in: TCP (similar concept), some file formats.
     * Stronger than simple sum but weaker than CRC.
     */
    unsigned short sum1 = 0, sum2 = 0;
    for (size_t i = 0; i < len; i++) {
        sum1 = (sum1 + data[i]) % 255;
        sum2 = (sum2 + sum1) % 255;
    }
    return (unsigned short)((sum2 << 8) | sum1);
}

unsigned int crc32_simple(const unsigned char *data, size_t len) {
    /*
     * Simplified CRC-32 implementation.
     * CRC is a polynomial division in GF(2) (binary arithmetic without carry).
     * It's much stronger than checksums at detecting burst errors.
     *
     * The polynomial 0xEDB88320 is the bit-reversed form of the
     * standard CRC-32 polynomial used in Ethernet, ZIP, PNG, etc.
     */
    unsigned int crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; bit++) {
            if (crc & 1) crc = (crc >> 1) ^ 0xEDB88320;
            else crc >>= 1;
        }
    }
    return ~crc;
}

void exercise_5(void) {
    printf("\n=== Exercise 5: Simple Checksum ===\n");

    struct {
        const char *label;
        const char *data;
    } tests[] = {
        {"Hello",          "Hello"},
        {"Hello (again)",  "Hello"},
        {"Hellp (1 bit)",  "Hellp"},   /* One bit different from Hello */
        {"olleH (reversed)", "olleH"},
        {"Empty",          ""},
        {"Single byte",   "A"},
    };
    int n_tests = (int)(sizeof(tests) / sizeof(tests[0]));

    printf("%-20s  %-6s  %-10s  %-12s\n",
           "Data", "XOR", "Fletcher16", "CRC-32");
    printf("--------------------  ------  ----------  ------------\n");

    for (int i = 0; i < n_tests; i++) {
        size_t len = strlen(tests[i].data);
        const unsigned char *d = (const unsigned char *)tests[i].data;

        printf("%-20s  0x%02X    0x%04X      0x%08X\n",
               tests[i].label,
               checksum_xor(d, len),
               checksum_fletcher16(d, len),
               crc32_simple(d, len));
    }

    printf("\nNote: XOR gives same result for 'Hello' and 'olleH'\n");
    printf("(position-insensitive), but Fletcher-16 and CRC-32 differ.\n");
    printf("This shows why position-dependent checks are superior.\n");
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();

    printf("\nAll exercises completed!\n");
    return 0;
}
