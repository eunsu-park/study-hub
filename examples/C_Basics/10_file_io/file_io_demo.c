/*
 * file_io_demo.c — Text and binary file read/write operations.
 *
 * Compile: gcc -Wall -Wextra -std=c11 -o file_io_demo file_io_demo.c
 * Run:     ./file_io_demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEXT_FILE   "demo_output.txt"
#define BINARY_FILE "demo_data.bin"

typedef struct {
    int    id;
    char   name[32];
    double score;
} Record;

int main(void)
{
    /* === Text file: write === */
    printf("=== Text File Write ===\n");
    FILE *fp = fopen(TEXT_FILE, "w");
    if (!fp) { perror("fopen"); return 1; }

    fprintf(fp, "Name,Score\n");
    fprintf(fp, "Alice,95.5\n");
    fprintf(fp, "Bob,87.3\n");
    fprintf(fp, "Carol,92.1\n");
    fclose(fp);
    printf("Wrote %s\n", TEXT_FILE);

    /* === Text file: read line-by-line === */
    printf("\n=== Text File Read ===\n");
    fp = fopen(TEXT_FILE, "r");
    if (!fp) { perror("fopen"); return 1; }

    char line[256];
    int line_num = 0;
    while (fgets(line, sizeof(line), fp)) {
        /* Remove trailing newline */
        line[strcspn(line, "\n")] = '\0';
        printf("  line %d: %s\n", ++line_num, line);
    }
    fclose(fp);

    /* === Text file: append === */
    printf("\n=== Text File Append ===\n");
    fp = fopen(TEXT_FILE, "a");
    if (!fp) { perror("fopen"); return 1; }
    fprintf(fp, "Dave,88.0\n");
    fclose(fp);
    printf("Appended one line to %s\n", TEXT_FILE);

    /* === Binary file: write structs === */
    printf("\n=== Binary File Write ===\n");
    Record records[] = {
        {1, "Alice",  95.5},
        {2, "Bob",    87.3},
        {3, "Carol",  92.1}
    };
    int count = (int)(sizeof(records) / sizeof(records[0]));

    fp = fopen(BINARY_FILE, "wb");
    if (!fp) { perror("fopen"); return 1; }

    fwrite(&count, sizeof(int), 1, fp);          /* write record count */
    fwrite(records, sizeof(Record), (size_t)count, fp); /* write all records */
    fclose(fp);
    printf("Wrote %d records to %s (%zu bytes each)\n",
           count, BINARY_FILE, sizeof(Record));

    /* === Binary file: read structs === */
    printf("\n=== Binary File Read ===\n");
    fp = fopen(BINARY_FILE, "rb");
    if (!fp) { perror("fopen"); return 1; }

    int n;
    fread(&n, sizeof(int), 1, fp);
    printf("Reading %d records:\n", n);

    for (int i = 0; i < n; i++) {
        Record r;
        fread(&r, sizeof(Record), 1, fp);
        printf("  [%d] %s — %.1f\n", r.id, r.name, r.score);
    }
    fclose(fp);

    /* Clean up generated files */
    remove(TEXT_FILE);
    remove(BINARY_FILE);
    printf("\nCleaned up temporary files.\n");

    return 0;
}
