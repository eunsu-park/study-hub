// simple_xor.c
// Simple XOR encryption demo
// Learning objective: Understanding the reversibility of XOR operations

#include <stdio.h>
#include <string.h>

/**
 * XOR encryption/decryption function
 *
 * @param data Data to encrypt (modified in-place)
 * @param len Data length
 * @param key Encryption key (single character)
 *
 * Key property of XOR:
 * - A ^ B = C
 * - C ^ B = A (XORing again with the same key restores the original)
 */
void xor_encrypt(char *data, int len, char key) {
    for (int i = 0; i < len; i++) {
        data[i] ^= key;  // Encrypt/decrypt with XOR operation
    }
}

/**
 * Print binary data in hexadecimal
 *
 * @param data Data to print
 * @param len Data length
 */
void print_hex(const char *data, int len) {
    for (int i = 0; i < len; i++) {
        printf("%02X ", (unsigned char)data[i]);
    }
    printf("\n");
}

/**
 * Print bit pattern (8 bits)
 *
 * @param byte Byte to print
 */
void print_binary(unsigned char byte) {
    for (int i = 7; i >= 0; i--) {
        printf("%d", (byte >> i) & 1);
    }
}

int main(void) {
    char message[] = "Hello, World!";
    char key = 'K';  // Simple single-character key (ASCII 75)

    printf("=== XOR Encryption Demo ===\n\n");

    // Print original message
    printf("Original message: %s\n", message);
    printf("Original (hex):   ");
    print_hex(message, strlen(message));
    printf("\n");

    // Detailed XOR operation for the first character
    printf("XOR operation for first char 'H' XOR 'K':\n");
    printf("  'H' = %d (0b", (unsigned char)message[0]);
    print_binary((unsigned char)message[0]);
    printf(")\n");
    printf("  'K' = %d (0b", (unsigned char)key);
    print_binary((unsigned char)key);
    printf(")\n");
    printf("  XOR = %d (0b", (unsigned char)(message[0] ^ key));
    print_binary((unsigned char)(message[0] ^ key));
    printf(")\n\n");

    // Encrypt
    xor_encrypt(message, strlen(message), key);
    printf("Encryption done!\n");
    printf("Encrypted (hex): ");
    print_hex(message, strlen(message));

    // Encrypted text may contain control characters that cannot be printed
    printf("Encrypted text: ");
    for (int i = 0; message[i]; i++) {
        // Display only printable characters
        if (message[i] >= 32 && message[i] <= 126) {
            printf("%c", message[i]);
        } else {
            printf("?");  // Show control characters as ?
        }
    }
    printf("\n\n");

    // Decrypt (XOR again with the same key)
    xor_encrypt(message, strlen(message), key);
    printf("Decryption done! (XOR again with the same key)\n");
    printf("Decrypted result: %s\n", message);
    printf("Decrypted (hex):  ");
    print_hex(message, strlen(message));

    // Verify XOR reversibility
    printf("\n=== XOR Reversibility Verification ===\n");
    char test = 'A';
    printf("Original:  %c (%d)\n", test, (unsigned char)test);

    test ^= key;
    printf("Encrypted: %c (%d)\n", test, (unsigned char)test);

    test ^= key;
    printf("Decrypted: %c (%d)\n", test, (unsigned char)test);
    printf("Success: Original and decrypted results are identical!\n");

    return 0;
}
