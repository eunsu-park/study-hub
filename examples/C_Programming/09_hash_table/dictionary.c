/*
 * dictionary.c
 * Practical dictionary program using a hash table
 *
 * Features:
 * - Add/search/delete words
 * - Display full word list
 * - Save/load from file
 * - Word statistics and search suggestions
 * - Case-insensitive search
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

#define TABLE_SIZE 1000
#define KEY_SIZE 100
#define VALUE_SIZE 500
#define FILENAME "dictionary.txt"

// Node struct (chaining method)
typedef struct Node {
    char word[KEY_SIZE];
    char meaning[VALUE_SIZE];
    int search_count;       // Number of times searched
    struct Node *next;
} Node;

// Dictionary struct
typedef struct {
    Node *buckets[TABLE_SIZE];
    int count;
    int total_searches;
} Dictionary;

// Statistics struct
typedef struct {
    char word[KEY_SIZE];
    int count;
} WordStat;

// Case-insensitive djb2 hash function
unsigned int hash(const char *key) {
    unsigned int hash = 5381;
    while (*key) {
        hash = ((hash << 5) + hash) + tolower((unsigned char)*key++);
    }
    return hash % TABLE_SIZE;
}

// Create dictionary
Dictionary* dict_create(void) {
    Dictionary *dict = calloc(1, sizeof(Dictionary));
    if (!dict) {
        fprintf(stderr, "Memory allocation failed\n");
    }
    return dict;
}

// Destroy dictionary
void dict_destroy(Dictionary *dict) {
    if (!dict) return;

    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = dict->buckets[i];
        while (current) {
            Node *next = current->next;
            free(current);
            current = next;
        }
    }
    free(dict);
}

// Add or update a word
void dict_add(Dictionary *dict, const char *word, const char *meaning) {
    if (!dict || !word || !meaning) return;

    unsigned int index = hash(word);

    // Check for existing word
    Node *current = dict->buckets[index];
    while (current) {
        if (strcasecmp(current->word, word) == 0) {
            // Update existing word
            strncpy(current->meaning, meaning, VALUE_SIZE - 1);
            current->meaning[VALUE_SIZE - 1] = '\0';
            printf("'%s' updated\n", word);
            return;
        }
        current = current->next;
    }

    // Add new word
    Node *node = malloc(sizeof(Node));
    if (!node) {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }

    strncpy(node->word, word, KEY_SIZE - 1);
    node->word[KEY_SIZE - 1] = '\0';
    strncpy(node->meaning, meaning, VALUE_SIZE - 1);
    node->meaning[VALUE_SIZE - 1] = '\0';
    node->search_count = 0;

    node->next = dict->buckets[index];
    dict->buckets[index] = node;
    dict->count++;

    printf("'%s' added\n", word);
}

// Search for a word
char* dict_search(Dictionary *dict, const char *word) {
    if (!dict || !word) return NULL;

    unsigned int index = hash(word);

    Node *current = dict->buckets[index];
    while (current) {
        if (strcasecmp(current->word, word) == 0) {
            current->search_count++;
            dict->total_searches++;
            return current->meaning;
        }
        current = current->next;
    }

    return NULL;
}

// Delete a word
bool dict_delete(Dictionary *dict, const char *word) {
    if (!dict || !word) return false;

    unsigned int index = hash(word);

    Node *current = dict->buckets[index];
    Node *prev = NULL;

    while (current) {
        if (strcasecmp(current->word, word) == 0) {
            if (prev) {
                prev->next = current->next;
            } else {
                dict->buckets[index] = current->next;
            }
            free(current);
            dict->count--;
            printf("'%s' deleted\n", word);
            return true;
        }
        prev = current;
        current = current->next;
    }

    printf("'%s' not found\n", word);
    return false;
}

// Display all words
void dict_list(Dictionary *dict) {
    if (!dict) return;

    printf("\n+============================================+\n");
    printf("|       Dictionary List (%d total)            |\n", dict->count);
    printf("+============================================+\n\n");

    if (dict->count == 0) {
        printf("  (empty)\n");
        return;
    }

    int num = 0;
    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = dict->buckets[i];
        while (current) {
            printf("  %3d. %-20s : %s\n",
                   ++num, current->word, current->meaning);
            current = current->next;
        }
    }
}

// Save to file
bool dict_save(Dictionary *dict, const char *filename) {
    if (!dict || !filename) return false;

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return false;
    }

    // Write header
    fprintf(fp, "# Dictionary File\n");
    fprintf(fp, "# Count: %d\n\n", dict->count);

    // Save all words
    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = dict->buckets[i];
        while (current) {
            fprintf(fp, "%s|%s|%d\n",
                   current->word, current->meaning, current->search_count);
            current = current->next;
        }
    }

    fclose(fp);
    printf("Saved %d words to '%s'\n", dict->count, filename);
    return true;
}

// Load from file
bool dict_load(Dictionary *dict, const char *filename) {
    if (!dict || !filename) return false;

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return false;
    }

    char line[KEY_SIZE + VALUE_SIZE + 50];
    int loaded = 0;

    while (fgets(line, sizeof(line), fp)) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n') continue;

        // Remove newline
        line[strcspn(line, "\n")] = '\0';

        // Parse: word|meaning|search_count
        char word[KEY_SIZE], meaning[VALUE_SIZE];
        int search_count = 0;

        char *token = strtok(line, "|");
        if (token) strncpy(word, token, KEY_SIZE - 1);

        token = strtok(NULL, "|");
        if (token) strncpy(meaning, token, VALUE_SIZE - 1);

        token = strtok(NULL, "|");
        if (token) search_count = atoi(token);

        // Add to dictionary (without output)
        unsigned int index = hash(word);
        Node *node = malloc(sizeof(Node));
        if (!node) continue;

        strncpy(node->word, word, KEY_SIZE - 1);
        node->word[KEY_SIZE - 1] = '\0';
        strncpy(node->meaning, meaning, VALUE_SIZE - 1);
        node->meaning[VALUE_SIZE - 1] = '\0';
        node->search_count = search_count;

        node->next = dict->buckets[index];
        dict->buckets[index] = node;
        dict->count++;
        loaded++;
    }

    fclose(fp);
    printf("Loaded %d words from '%s'\n", loaded, filename);
    return true;
}

// Search suggestions (partial match)
void dict_suggest(Dictionary *dict, const char *prefix) {
    if (!dict || !prefix) return;

    printf("\nWords starting with '%s':\n", prefix);

    int found = 0;
    int len = strlen(prefix);

    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = dict->buckets[i];
        while (current) {
            if (strncasecmp(current->word, prefix, len) == 0) {
                printf("  - %s\n", current->word);
                found++;
            }
            current = current->next;
        }
    }

    if (found == 0) {
        printf("  (none)\n");
    } else {
        printf("%d found\n", found);
    }
}

// Popular word statistics
void dict_statistics(Dictionary *dict) {
    if (!dict) return;

    printf("\n+============================================+\n");
    printf("|          Dictionary Statistics              |\n");
    printf("+============================================+\n\n");

    printf("Total words:      %d\n", dict->count);
    printf("Total searches:   %d\n", dict->total_searches);

    // Sort by search count (Top 10)
    WordStat *stats = malloc(sizeof(WordStat) * dict->count);
    if (!stats) return;

    int idx = 0;
    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = dict->buckets[i];
        while (current) {
            strncpy(stats[idx].word, current->word, KEY_SIZE - 1);
            stats[idx].count = current->search_count;
            idx++;
            current = current->next;
        }
    }

    // Bubble sort (simple)
    for (int i = 0; i < dict->count - 1; i++) {
        for (int j = 0; j < dict->count - i - 1; j++) {
            if (stats[j].count < stats[j + 1].count) {
                WordStat temp = stats[j];
                stats[j] = stats[j + 1];
                stats[j + 1] = temp;
            }
        }
    }

    // Print Top 10
    printf("\nTop 10 Most Searched Words:\n");
    int limit = dict->count < 10 ? dict->count : 10;
    for (int i = 0; i < limit; i++) {
        if (stats[i].count > 0) {
            printf("  %2d. %-20s (%d times)\n",
                   i + 1, stats[i].word, stats[i].count);
        }
    }

    free(stats);
}

// Print menu
void print_menu(void) {
    printf("\n+============================================+\n");
    printf("|        Simple Dictionary Program           |\n");
    printf("+============================================+\n");
    printf("|  1. Add word                               |\n");
    printf("|  2. Search word                            |\n");
    printf("|  3. Delete word                            |\n");
    printf("|  4. List all                               |\n");
    printf("|  5. Search suggestions                     |\n");
    printf("|  6. View statistics                        |\n");
    printf("|  7. Save to file                           |\n");
    printf("|  8. Load from file                         |\n");
    printf("|  0. Quit                                   |\n");
    printf("+============================================+\n");
}

// Clear input buffer
void clear_input(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}

// Load sample data
void load_sample_data(Dictionary *dict) {
    dict_add(dict, "apple", "A round fruit with red or green skin");
    dict_add(dict, "book", "A written or printed work consisting of pages");
    dict_add(dict, "computer", "An electronic device for processing data");
    dict_add(dict, "dictionary", "A reference book listing words with their definitions");
    dict_add(dict, "education", "The process of teaching and learning knowledge and skills");
    dict_add(dict, "friend", "A person with whom one has a close relationship");
    dict_add(dict, "galaxy", "A system of millions or billions of stars");
    dict_add(dict, "happiness", "A state of well-being and contentment");
    dict_add(dict, "internet", "A global network connecting computers worldwide");
    dict_add(dict, "javascript", "A programming language for web development");
}

// Main function
int main(void) {
    Dictionary *dict = dict_create();
    if (!dict) return 1;

    // Load sample data
    printf("Loading sample data...\n");
    load_sample_data(dict);

    // Load existing file if present
    FILE *test = fopen(FILENAME, "r");
    if (test) {
        fclose(test);
        printf("\nExisting dictionary file found.\n");
        printf("Would you like to load it? (y/n): ");
        char choice;
        scanf(" %c", &choice);
        clear_input();

        if (choice == 'y' || choice == 'Y') {
            // Delete existing data then load
            dict_destroy(dict);
            dict = dict_create();
            dict_load(dict, FILENAME);
        }
    }

    int choice;
    char word[KEY_SIZE];
    char meaning[VALUE_SIZE];

    while (1) {
        print_menu();
        printf("Choice: ");

        if (scanf("%d", &choice) != 1) {
            clear_input();
            printf("Invalid input\n");
            continue;
        }
        clear_input();

        switch (choice) {
            case 1:  // Add
                printf("\nWord: ");
                fgets(word, KEY_SIZE, stdin);
                word[strcspn(word, "\n")] = '\0';

                if (strlen(word) == 0) {
                    printf("Please enter a word\n");
                    break;
                }

                printf("Definition: ");
                fgets(meaning, VALUE_SIZE, stdin);
                meaning[strcspn(meaning, "\n")] = '\0';

                if (strlen(meaning) == 0) {
                    printf("Please enter a definition\n");
                    break;
                }

                dict_add(dict, word, meaning);
                break;

            case 2:  // Search
                printf("\nWord to search: ");
                fgets(word, KEY_SIZE, stdin);
                word[strcspn(word, "\n")] = '\0';

                char *result = dict_search(dict, word);
                if (result) {
                    printf("\n+----------------------------------------+\n");
                    printf("| %s\n", word);
                    printf("+----------------------------------------+\n");
                    printf("| %s\n", result);
                    printf("+----------------------------------------+\n");
                } else {
                    printf("\n'%s' not found\n", word);
                    dict_suggest(dict, word);
                }
                break;

            case 3:  // Delete
                printf("\nWord to delete: ");
                fgets(word, KEY_SIZE, stdin);
                word[strcspn(word, "\n")] = '\0';

                dict_delete(dict, word);
                break;

            case 4:  // List
                dict_list(dict);
                break;

            case 5:  // Suggestions
                printf("\nPrefix to search: ");
                fgets(word, KEY_SIZE, stdin);
                word[strcspn(word, "\n")] = '\0';

                dict_suggest(dict, word);
                break;

            case 6:  // Statistics
                dict_statistics(dict);
                break;

            case 7:  // Save
                dict_save(dict, FILENAME);
                break;

            case 8:  // Load
                printf("\nCurrent data will be deleted. Continue? (y/n): ");
                char confirm;
                scanf(" %c", &confirm);
                clear_input();

                if (confirm == 'y' || confirm == 'Y') {
                    dict_destroy(dict);
                    dict = dict_create();
                    dict_load(dict, FILENAME);
                }
                break;

            case 0:  // Quit
                printf("\nWould you like to save? (y/n): ");
                char save_choice;
                scanf(" %c", &save_choice);
                clear_input();

                if (save_choice == 'y' || save_choice == 'Y') {
                    dict_save(dict, FILENAME);
                }

                printf("Exiting the dictionary.\n");
                dict_destroy(dict);
                return 0;

            default:
                printf("Invalid choice\n");
        }
    }

    return 0;
}
