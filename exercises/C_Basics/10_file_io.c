/*
 * Exercises for Lesson 10: File I/O
 * Topic: C_Basics
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex10 10_file_io.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* === Exercise 1: Word Counter from File === */
/* Problem: Count lines, words, and characters in a text file (like wc). */

typedef struct {
    int lines;
    int words;
    int chars;
} FileStats;

FileStats count_file(const char *filename) {
    FileStats stats = {0, 0, 0};
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror(filename);
        return stats;
    }

    int c;
    int in_word = 0;

    while ((c = fgetc(fp)) != EOF) {
        stats.chars++;
        if (c == '\n') stats.lines++;
        if (isspace(c)) {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            stats.words++;
        }
    }

    fclose(fp);
    return stats;
}

void exercise_1(void) {
    printf("=== Exercise 1: Word Counter from File ===\n");

    /* Create a test file */
    const char *testfile = "/tmp/c_basics_ex10_test.txt";
    FILE *fp = fopen(testfile, "w");
    if (!fp) { perror("create test file"); return; }
    fprintf(fp, "Hello World\n");
    fprintf(fp, "This is a test file.\n");
    fprintf(fp, "It has multiple lines\n");
    fprintf(fp, "and various words.\n");
    fclose(fp);

    FileStats stats = count_file(testfile);
    printf("File: %s\n", testfile);
    printf("  Lines: %d\n", stats.lines);
    printf("  Words: %d\n", stats.words);
    printf("  Chars: %d\n", stats.chars);

    /* Clean up */
    remove(testfile);
}

/* === Exercise 2: Config File Parser === */
/* Problem: Parse a simple key=value config file, ignoring comments and
 *          blank lines. */

#define MAX_KEY 64
#define MAX_VAL 256
#define MAX_ENTRIES 32

typedef struct {
    char key[MAX_KEY];
    char value[MAX_VAL];
} ConfigEntry;

typedef struct {
    ConfigEntry entries[MAX_ENTRIES];
    int count;
} Config;

/* Trim leading/trailing whitespace in-place */
static char *trim(char *str) {
    while (isspace((unsigned char)*str)) str++;
    if (*str == '\0') return str;
    char *end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    *(end + 1) = '\0';
    return str;
}

int config_load(Config *cfg, const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) { perror(filename); return 0; }

    cfg->count = 0;
    char line[512];

    while (fgets(line, sizeof(line), fp) && cfg->count < MAX_ENTRIES) {
        char *trimmed = trim(line);

        /* Skip empty lines and comments */
        if (*trimmed == '\0' || *trimmed == '#' || *trimmed == ';') {
            continue;
        }

        /* Find the '=' delimiter */
        char *eq = strchr(trimmed, '=');
        if (!eq) continue;

        *eq = '\0';
        char *key = trim(trimmed);
        char *val = trim(eq + 1);

        strncpy(cfg->entries[cfg->count].key, key, MAX_KEY - 1);
        strncpy(cfg->entries[cfg->count].value, val, MAX_VAL - 1);
        cfg->count++;
    }

    fclose(fp);
    return 1;
}

const char *config_get(const Config *cfg, const char *key) {
    for (int i = 0; i < cfg->count; i++) {
        if (strcmp(cfg->entries[i].key, key) == 0) {
            return cfg->entries[i].value;
        }
    }
    return NULL;
}

void config_print(const Config *cfg) {
    for (int i = 0; i < cfg->count; i++) {
        printf("  [%s] = [%s]\n", cfg->entries[i].key, cfg->entries[i].value);
    }
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Config File Parser ===\n");

    /* Create a test config file */
    const char *cfgfile = "/tmp/c_basics_ex10_config.ini";
    FILE *fp = fopen(cfgfile, "w");
    if (!fp) { perror("create config file"); return; }
    fprintf(fp, "# Database configuration\n");
    fprintf(fp, "host = localhost\n");
    fprintf(fp, "port = 5432\n");
    fprintf(fp, "database = myapp\n");
    fprintf(fp, "\n");
    fprintf(fp, "; Authentication\n");
    fprintf(fp, "username = admin\n");
    fprintf(fp, "password = secret123\n");
    fprintf(fp, "max_connections = 100\n");
    fclose(fp);

    Config cfg;
    if (config_load(&cfg, cfgfile)) {
        printf("Loaded %d entries from %s:\n", cfg.count, cfgfile);
        config_print(&cfg);

        /* Look up specific keys */
        const char *host = config_get(&cfg, "host");
        const char *port = config_get(&cfg, "port");
        printf("\nLookup: host=%s, port=%s\n",
               host ? host : "(not found)",
               port ? port : "(not found)");

        const char *missing = config_get(&cfg, "nonexistent");
        printf("Lookup: nonexistent=%s\n",
               missing ? missing : "(not found)");
    }

    /* Clean up */
    remove(cfgfile);
}

int main(void) {
    exercise_1();
    exercise_2();

    printf("\nAll exercises completed!\n");
    return 0;
}
