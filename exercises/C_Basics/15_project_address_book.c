/*
 * Exercises for Lesson 05: Project Address Book
 * Topic: C_Basics
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex05 05_project_address_book.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_CONTACTS 100
#define NAME_LEN 64
#define PHONE_LEN 20
#define EMAIL_LEN 64

/* === Exercise 1: Struct Design === */
/* Problem: Design a contact struct with proper field sizing and initialization. */

typedef struct {
    char first_name[NAME_LEN];
    char last_name[NAME_LEN];
    char phone[PHONE_LEN];
    char email[EMAIL_LEN];
    int  active;  /* Soft delete flag: 1 = active, 0 = deleted */
} Contact;

typedef struct {
    Contact entries[MAX_CONTACTS];
    int count;
} AddressBook;

void addressbook_init(AddressBook *ab) {
    ab->count = 0;
    memset(ab->entries, 0, sizeof(ab->entries));
}

int addressbook_add(AddressBook *ab, const char *first, const char *last,
                    const char *phone, const char *email) {
    /*
     * Defensive programming: always check capacity before adding.
     * strncpy with explicit null-termination prevents buffer overflow.
     * A common bug: strncpy does NOT null-terminate if src >= n chars.
     */
    if (ab->count >= MAX_CONTACTS) return -1;

    Contact *c = &ab->entries[ab->count];
    strncpy(c->first_name, first, NAME_LEN - 1);
    c->first_name[NAME_LEN - 1] = '\0';
    strncpy(c->last_name, last, NAME_LEN - 1);
    c->last_name[NAME_LEN - 1] = '\0';
    strncpy(c->phone, phone, PHONE_LEN - 1);
    c->phone[PHONE_LEN - 1] = '\0';
    strncpy(c->email, email, EMAIL_LEN - 1);
    c->email[EMAIL_LEN - 1] = '\0';
    c->active = 1;

    ab->count++;
    return ab->count - 1; /* Return index of new contact */
}

void exercise_1(void) {
    printf("=== Exercise 1: Struct Design ===\n");

    AddressBook ab;
    addressbook_init(&ab);

    printf("Contact struct size:     %zu bytes\n", sizeof(Contact));
    printf("AddressBook struct size: %zu bytes\n", sizeof(AddressBook));
    printf("Max contacts:            %d\n", MAX_CONTACTS);

    /*
     * Design decisions:
     * - Fixed-size arrays vs dynamic strings: simpler memory management,
     *   predictable struct size, but wastes space for short names.
     * - Soft delete (active flag) vs shifting array: O(1) delete vs O(n).
     * - Separate first/last name: enables sorting by either field.
     */

    addressbook_add(&ab, "Alice", "Smith", "555-0101", "alice@example.com");
    addressbook_add(&ab, "Bob", "Jones", "555-0102", "bob@example.com");
    addressbook_add(&ab, "Charlie", "Brown", "555-0103", "charlie@example.com");

    printf("\nAddress book (%d contacts):\n", ab.count);
    printf("%-12s %-12s %-12s %-20s\n", "First", "Last", "Phone", "Email");
    printf("------------ ------------ ------------ --------------------\n");

    for (int i = 0; i < ab.count; i++) {
        if (ab.entries[i].active) {
            printf("%-12s %-12s %-12s %-20s\n",
                   ab.entries[i].first_name, ab.entries[i].last_name,
                   ab.entries[i].phone, ab.entries[i].email);
        }
    }
}

/* === Exercise 2: Search Functions === */
/* Problem: Implement case-insensitive search by name, phone, or email. */

/* Case-insensitive substring search */
static int strcasestr_custom(const char *haystack, const char *needle) {
    /*
     * Standard strstr is case-sensitive. We implement our own
     * case-insensitive version for portability (strcasestr is POSIX,
     * not C standard).
     *
     * Time complexity: O(n * m) where n = haystack length, m = needle length.
     * Could use Boyer-Moore for better average case, but overkill here.
     */
    if (!*needle) return 1;

    for (const char *h = haystack; *h; h++) {
        const char *hp = h;
        const char *np = needle;
        while (*hp && *np && tolower((unsigned char)*hp) == tolower((unsigned char)*np)) {
            hp++;
            np++;
        }
        if (!*np) return 1; /* Found */
    }
    return 0;
}

typedef struct {
    int indices[MAX_CONTACTS];
    int count;
} SearchResult;

SearchResult search_contacts(const AddressBook *ab, const char *query) {
    SearchResult result = { .count = 0 };

    for (int i = 0; i < ab->count; i++) {
        if (!ab->entries[i].active) continue;

        const Contact *c = &ab->entries[i];
        if (strcasestr_custom(c->first_name, query) ||
            strcasestr_custom(c->last_name, query) ||
            strcasestr_custom(c->phone, query) ||
            strcasestr_custom(c->email, query)) {
            result.indices[result.count++] = i;
        }
    }
    return result;
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Search Functions ===\n");

    AddressBook ab;
    addressbook_init(&ab);
    addressbook_add(&ab, "Alice", "Smith", "555-0101", "alice@example.com");
    addressbook_add(&ab, "Bob", "Jones", "555-0102", "bob@work.com");
    addressbook_add(&ab, "Alice", "Johnson", "555-0103", "aj@example.com");
    addressbook_add(&ab, "Charlie", "Smith", "555-0104", "charlie@work.com");
    addressbook_add(&ab, "Diana", "Prince", "555-0105", "diana@example.com");

    const char *queries[] = {"alice", "Smith", "555-01", "work.com", "xyz", "DIANA"};
    int n_queries = (int)(sizeof(queries) / sizeof(queries[0]));

    for (int q = 0; q < n_queries; q++) {
        SearchResult sr = search_contacts(&ab, queries[q]);
        printf("Search '%s' -> %d result(s):", queries[q], sr.count);
        for (int i = 0; i < sr.count; i++) {
            int idx = sr.indices[i];
            printf(" %s %s", ab.entries[idx].first_name, ab.entries[idx].last_name);
            if (i < sr.count - 1) printf(",");
        }
        printf("\n");
    }
}

/* === Exercise 3: File Save/Load === */
/* Problem: Serialize address book to CSV and deserialize back. */

int addressbook_save(const AddressBook *ab, const char *filename) {
    /*
     * CSV format: simple, human-readable, widely compatible.
     * Caveat: Fields containing commas or quotes need escaping.
     * For a production system, use a proper CSV library or JSON.
     *
     * We write a header line followed by one line per active contact.
     */
    FILE *fp = fopen(filename, "w");
    if (!fp) return -1;

    fprintf(fp, "first_name,last_name,phone,email\n");
    for (int i = 0; i < ab->count; i++) {
        if (!ab->entries[i].active) continue;
        const Contact *c = &ab->entries[i];
        fprintf(fp, "%s,%s,%s,%s\n", c->first_name, c->last_name,
                c->phone, c->email);
    }

    fclose(fp);
    return 0;
}

int addressbook_load(AddressBook *ab, const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;

    addressbook_init(ab);

    char line[256];
    int header_skipped = 0;

    while (fgets(line, sizeof(line), fp)) {
        /* Remove trailing newline */
        line[strcspn(line, "\n")] = '\0';

        if (!header_skipped) { header_skipped = 1; continue; }

        /* Parse CSV fields (simple: no escaping support) */
        char first[NAME_LEN], last[NAME_LEN], phone[PHONE_LEN], email[EMAIL_LEN];
        if (sscanf(line, "%63[^,],%63[^,],%19[^,],%63[^,\n]",
                   first, last, phone, email) == 4) {
            addressbook_add(ab, first, last, phone, email);
        }
    }

    fclose(fp);
    return 0;
}

void exercise_3(void) {
    printf("\n=== Exercise 3: File Save/Load ===\n");

    AddressBook ab;
    addressbook_init(&ab);
    addressbook_add(&ab, "Alice", "Smith", "555-0101", "alice@example.com");
    addressbook_add(&ab, "Bob", "Jones", "555-0102", "bob@work.com");
    addressbook_add(&ab, "Charlie", "Brown", "555-0103", "charlie@mail.com");

    const char *filename = "/tmp/addressbook_test.csv";

    /* Save */
    if (addressbook_save(&ab, filename) == 0) {
        printf("Saved %d contacts to %s\n", ab.count, filename);
    }

    /* Load into new address book */
    AddressBook loaded;
    if (addressbook_load(&loaded, filename) == 0) {
        printf("Loaded %d contacts from %s\n", loaded.count, filename);

        /* Verify data integrity */
        int match = 1;
        for (int i = 0; i < ab.count && i < loaded.count; i++) {
            if (strcmp(ab.entries[i].first_name, loaded.entries[i].first_name) != 0 ||
                strcmp(ab.entries[i].last_name, loaded.entries[i].last_name) != 0) {
                match = 0;
                break;
            }
        }
        printf("Data integrity check: %s\n", match ? "PASS" : "FAIL");
    }

    /* Clean up */
    remove(filename);
    printf("Cleaned up test file.\n");
}

/* === Exercise 4: Sorting Contacts === */
/* Problem: Sort contacts by last name, then first name using qsort. */

int compare_by_name(const void *a, const void *b) {
    /*
     * qsort comparator for contacts:
     * 1. Primary sort: last name (ascending, case-insensitive)
     * 2. Secondary sort: first name (ascending, case-insensitive)
     *
     * strcasecmp is POSIX; for pure C11, we'd need a manual comparison.
     * Using strcmp here for simplicity (case-sensitive).
     */
    const Contact *ca = (const Contact *)a;
    const Contact *cb = (const Contact *)b;

    int cmp = strcmp(ca->last_name, cb->last_name);
    if (cmp != 0) return cmp;
    return strcmp(ca->first_name, cb->first_name);
}

void exercise_4(void) {
    printf("\n=== Exercise 4: Sorting Contacts ===\n");

    AddressBook ab;
    addressbook_init(&ab);
    addressbook_add(&ab, "Charlie", "Brown", "555-0103", "charlie@mail.com");
    addressbook_add(&ab, "Alice", "Smith", "555-0101", "alice@example.com");
    addressbook_add(&ab, "Bob", "Jones", "555-0102", "bob@work.com");
    addressbook_add(&ab, "Alice", "Johnson", "555-0104", "aj@example.com");
    addressbook_add(&ab, "Diana", "Brown", "555-0105", "diana@mail.com");

    printf("Before sorting:\n");
    for (int i = 0; i < ab.count; i++) {
        printf("  %s %s\n", ab.entries[i].first_name, ab.entries[i].last_name);
    }

    /* qsort: O(n log n) average, in-place */
    qsort(ab.entries, (size_t)ab.count, sizeof(Contact), compare_by_name);

    printf("\nAfter sorting (by last name, then first name):\n");
    for (int i = 0; i < ab.count; i++) {
        printf("  %s %s\n", ab.entries[i].first_name, ab.entries[i].last_name);
    }

    /*
     * qsort is not guaranteed to be stable (equal elements may be
     * reordered). If stability matters, use merge sort or add a
     * sequence number as a tiebreaker in the comparator.
     */
}

/* === Exercise 5: Duplicate Detection === */
/* Problem: Detect and report duplicate contacts based on name or email. */

void exercise_5(void) {
    printf("\n=== Exercise 5: Duplicate Detection ===\n");

    AddressBook ab;
    addressbook_init(&ab);
    addressbook_add(&ab, "Alice", "Smith", "555-0101", "alice@example.com");
    addressbook_add(&ab, "Bob", "Jones", "555-0102", "bob@work.com");
    addressbook_add(&ab, "Alice", "Smith", "555-0199", "asmith@other.com");
    addressbook_add(&ab, "Charlie", "Brown", "555-0103", "alice@example.com");
    addressbook_add(&ab, "Diana", "Prince", "555-0104", "diana@mail.com");

    /*
     * Duplicate detection strategies:
     * 1. Exact name match: same first + last name
     * 2. Email match: same email address
     * 3. Fuzzy match: edit distance < threshold (more complex)
     *
     * Time complexity: O(n^2) brute force comparison.
     * For large datasets, use a hash set for O(n) average case.
     */

    /* Check for name duplicates */
    printf("Name duplicates:\n");
    int name_dups = 0;
    for (int i = 0; i < ab.count; i++) {
        for (int j = i + 1; j < ab.count; j++) {
            if (strcmp(ab.entries[i].first_name, ab.entries[j].first_name) == 0 &&
                strcmp(ab.entries[i].last_name, ab.entries[j].last_name) == 0) {
                printf("  [%d] %s %s <%s> == [%d] %s %s <%s>\n",
                       i, ab.entries[i].first_name, ab.entries[i].last_name,
                       ab.entries[i].email,
                       j, ab.entries[j].first_name, ab.entries[j].last_name,
                       ab.entries[j].email);
                name_dups++;
            }
        }
    }
    if (name_dups == 0) printf("  (none)\n");

    /* Check for email duplicates */
    printf("\nEmail duplicates:\n");
    int email_dups = 0;
    for (int i = 0; i < ab.count; i++) {
        for (int j = i + 1; j < ab.count; j++) {
            if (strcmp(ab.entries[i].email, ab.entries[j].email) == 0) {
                printf("  [%d] %s %s <%s> == [%d] %s %s <%s>\n",
                       i, ab.entries[i].first_name, ab.entries[i].last_name,
                       ab.entries[i].email,
                       j, ab.entries[j].first_name, ab.entries[j].last_name,
                       ab.entries[j].email);
                email_dups++;
            }
        }
    }
    if (email_dups == 0) printf("  (none)\n");

    printf("\nTotal: %d name duplicate(s), %d email duplicate(s)\n",
           name_dups, email_dups);
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
