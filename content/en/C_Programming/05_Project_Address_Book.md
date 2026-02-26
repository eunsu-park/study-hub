# Project 3: Address Book Program

**Previous**: [Project 2: Number Guessing Game](./04_Project_Number_Guessing.md) | **Next**: [Project 4: Dynamic Array](./06_Project_Dynamic_Array.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design a `typedef struct` to group related fields (name, phone, email) into a single record type
2. Build a fixed-size array of structs and manage a count variable to track active entries
3. Implement full CRUD operations (Create, Read, Update, Delete) using struct pointers and array shifting
4. Apply file I/O functions (`fopen`, `fread`, `fwrite`, `fclose`) to persist binary data across sessions
5. Distinguish between text-mode (`fprintf`/`fscanf`) and binary-mode (`fread`/`fwrite`) file operations
6. Use `fgets` with `strcspn` to safely read string input and strip trailing newlines
7. Search records with `strstr` for substring matching across multiple fields

---

An address book is the classic exercise for learning structs and file I/O because it mirrors real database applications: you create records, list them, search by keyword, edit fields, and delete entries -- the full CRUD cycle. By saving contacts to a binary file you will also experience the difference between volatile in-memory data and durable on-disk storage, a distinction that matters in every real-world program.

## Program Requirements

```
1. Add contact
2. View contact list
3. Search contacts
4. Edit contact
5. Delete contact
6. Save/Load to file
```

---

## Step 1: Struct Definition

### Core Syntax: Structs

```c
#include <stdio.h>
#include <string.h>

// Contact struct definition
typedef struct {
    int id;
    char name[50];
    char phone[20];
    char email[50];
} Contact;

int main(void) {
    // Struct variable declaration
    Contact c1;

    // Assign values to members
    c1.id = 1;
    strcpy(c1.name, "John Doe");
    strcpy(c1.phone, "010-1234-5678");
    strcpy(c1.email, "john@email.com");

    // Output
    printf("ID: %d\n", c1.id);
    printf("Name: %s\n", c1.name);
    printf("Phone: %s\n", c1.phone);
    printf("Email: %s\n", c1.email);

    // Declaration with initialization
    Contact c2 = {2, "Jane Smith", "010-9876-5432", "jane@email.com"};
    printf("\n%s: %s\n", c2.name, c2.phone);

    return 0;
}
```

### Struct Arrays

```c
#define MAX_CONTACTS 100

Contact contacts[MAX_CONTACTS];
int contact_count = 0;

// Add
contacts[contact_count].id = contact_count + 1;
strcpy(contacts[contact_count].name, "New Contact");
contact_count++;

// Iterate
for (int i = 0; i < contact_count; i++) {
    printf("%s\n", contacts[i].name);
}
```

---

## Step 2: Basic CRUD Implementation

```c
// addressbook_v1.c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_CONTACTS 100
#define NAME_LEN 50
#define PHONE_LEN 20
#define EMAIL_LEN 50

// Contact struct
typedef struct {
    int id;
    char name[NAME_LEN];
    char phone[PHONE_LEN];
    char email[EMAIL_LEN];
} Contact;

// Global variables
Contact contacts[MAX_CONTACTS];
int contact_count = 0;
int next_id = 1;

// Function declarations
void print_menu(void);
void add_contact(void);
void list_contacts(void);
void search_contact(void);
void delete_contact(void);
void clear_input_buffer(void);

int main(void) {
    int choice;

    printf("=== Address Book v1 ===\n");

    while (1) {
        print_menu();
        printf("Choice: ");

        if (scanf("%d", &choice) != 1) {
            clear_input_buffer();
            continue;
        }
        clear_input_buffer();

        switch (choice) {
            case 1:
                add_contact();
                break;
            case 2:
                list_contacts();
                break;
            case 3:
                search_contact();
                break;
            case 4:
                delete_contact();
                break;
            case 0:
                printf("Exiting program.\n");
                return 0;
            default:
                printf("Invalid choice.\n");
        }
        printf("\n");
    }

    return 0;
}

void print_menu(void) {
    printf("\n");
    printf("--------------------\n");
    printf("|  1. Add Contact  |\n");
    printf("|  2. View List    |\n");
    printf("|  3. Search       |\n");
    printf("|  4. Delete       |\n");
    printf("|  0. Exit         |\n");
    printf("--------------------\n");
}

void add_contact(void) {
    if (contact_count >= MAX_CONTACTS) {
        printf("Address book is full.\n");
        return;
    }

    Contact *c = &contacts[contact_count];
    c->id = next_id++;

    printf("\n[ Add New Contact ]\n");

    printf("Name: ");
    fgets(c->name, NAME_LEN, stdin);
    c->name[strcspn(c->name, "\n")] = '\0';  // Remove newline

    printf("Phone: ");
    fgets(c->phone, PHONE_LEN, stdin);
    c->phone[strcspn(c->phone, "\n")] = '\0';

    printf("Email: ");
    fgets(c->email, EMAIL_LEN, stdin);
    c->email[strcspn(c->email, "\n")] = '\0';

    contact_count++;
    printf("\nContact added. (ID: %d)\n", c->id);
}

void list_contacts(void) {
    printf("\n[ Contact List ] (Total: %d)\n", contact_count);
    printf("----------------------------------------\n");

    if (contact_count == 0) {
        printf("No saved contacts.\n");
        return;
    }

    for (int i = 0; i < contact_count; i++) {
        Contact *c = &contacts[i];
        printf("ID: %d\n", c->id);
        printf("  Name: %s\n", c->name);
        printf("  Phone: %s\n", c->phone);
        printf("  Email: %s\n", c->email);
        printf("----------------------------------------\n");
    }
}

void search_contact(void) {
    char keyword[NAME_LEN];
    int found = 0;

    printf("\n[ Search Contact ]\n");
    printf("Search term (name): ");
    fgets(keyword, NAME_LEN, stdin);
    keyword[strcspn(keyword, "\n")] = '\0';

    printf("\nSearch Results:\n");
    printf("----------------------------------------\n");

    for (int i = 0; i < contact_count; i++) {
        // Substring search (strstr)
        if (strstr(contacts[i].name, keyword) != NULL) {
            Contact *c = &contacts[i];
            printf("ID: %d | %s | %s\n", c->id, c->name, c->phone);
            found++;
        }
    }

    if (found == 0) {
        printf("No results found.\n");
    } else {
        printf("\n%d result(s) found\n", found);
    }
}

void delete_contact(void) {
    int id, found = -1;

    printf("\n[ Delete Contact ]\n");
    printf("ID to delete: ");
    scanf("%d", &id);
    clear_input_buffer();

    // Find by ID
    for (int i = 0; i < contact_count; i++) {
        if (contacts[i].id == id) {
            found = i;
            break;
        }
    }

    if (found == -1) {
        printf("Contact with that ID not found.\n");
        return;
    }

    // Confirmation
    printf("Delete '%s'? (y/n): ", contacts[found].name);
    char confirm;
    scanf(" %c", &confirm);
    clear_input_buffer();

    if (confirm != 'y' && confirm != 'Y') {
        printf("Deletion cancelled.\n");
        return;
    }

    // Delete: shift elements forward
    for (int i = found; i < contact_count - 1; i++) {
        contacts[i] = contacts[i + 1];
    }
    contact_count--;

    printf("Contact deleted.\n");
}

void clear_input_buffer(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}
```

---

## Step 3: Adding File Save/Load

### Core Syntax: File I/O

```c
#include <stdio.h>

// Text file writing
FILE *fp = fopen("data.txt", "w");
if (fp == NULL) {
    printf("Failed to open file\n");
    return;
}
fprintf(fp, "Hello, File!\n");
fprintf(fp, "Number: %d\n", 42);
fclose(fp);

// Text file reading
FILE *fp = fopen("data.txt", "r");
char line[100];
while (fgets(line, sizeof(line), fp) != NULL) {
    printf("%s", line);
}
fclose(fp);

// Binary file writing (useful for struct storage)
FILE *fp = fopen("data.bin", "wb");
Contact c = {1, "John Doe", "010-1234", "john@test.com"};
fwrite(&c, sizeof(Contact), 1, fp);
fclose(fp);

// Binary file reading
FILE *fp = fopen("data.bin", "rb");
Contact c;
fread(&c, sizeof(Contact), 1, fp);
fclose(fp);
```

### File Modes

| Mode | Description |
|------|-------------|
| `"r"` | Read (fails if file doesn't exist) |
| `"w"` | Write (overwrites file) |
| `"a"` | Append (adds to end of file) |
| `"rb"` | Binary read |
| `"wb"` | Binary write |

---

## Step 4: Final Version

```c
// addressbook.c (Final)
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_CONTACTS 100
#define NAME_LEN 50
#define PHONE_LEN 20
#define EMAIL_LEN 50
#define FILENAME "contacts.dat"

// Contact struct
typedef struct {
    int id;
    char name[NAME_LEN];
    char phone[PHONE_LEN];
    char email[EMAIL_LEN];
} Contact;

// Address book struct
typedef struct {
    Contact contacts[MAX_CONTACTS];
    int count;
    int next_id;
} AddressBook;

// Function declarations
void init_addressbook(AddressBook *ab);
void print_menu(void);
void add_contact(AddressBook *ab);
void list_contacts(AddressBook *ab);
void search_contact(AddressBook *ab);
void edit_contact(AddressBook *ab);
void delete_contact(AddressBook *ab);
int save_to_file(AddressBook *ab);
int load_from_file(AddressBook *ab);
void clear_input_buffer(void);
int find_by_id(AddressBook *ab, int id);

int main(void) {
    AddressBook ab;
    int choice;

    init_addressbook(&ab);

    // Load data from file
    if (load_from_file(&ab) == 0) {
        printf("Loaded existing data. (%d contacts)\n", ab.count);
    }

    printf("\n===============================\n");
    printf("|      Address Book           |\n");
    printf("===============================\n");

    while (1) {
        print_menu();
        printf("Choice: ");

        if (scanf("%d", &choice) != 1) {
            clear_input_buffer();
            printf("Please enter a number.\n");
            continue;
        }
        clear_input_buffer();

        switch (choice) {
            case 1:
                add_contact(&ab);
                break;
            case 2:
                list_contacts(&ab);
                break;
            case 3:
                search_contact(&ab);
                break;
            case 4:
                edit_contact(&ab);
                break;
            case 5:
                delete_contact(&ab);
                break;
            case 6:
                if (save_to_file(&ab) == 0) {
                    printf("Saved to file.\n");
                }
                break;
            case 0:
                // Save confirmation before exit
                printf("Save changes? (y/n): ");
                char save_confirm;
                scanf(" %c", &save_confirm);
                if (save_confirm == 'y' || save_confirm == 'Y') {
                    save_to_file(&ab);
                    printf("Save complete.\n");
                }
                printf("Exiting program.\n");
                return 0;
            default:
                printf("Invalid choice.\n");
        }
        printf("\n");
    }

    return 0;
}

void init_addressbook(AddressBook *ab) {
    ab->count = 0;
    ab->next_id = 1;
    memset(ab->contacts, 0, sizeof(ab->contacts));
}

void print_menu(void) {
    printf("\n-------------------------\n");
    printf("|  1. Add Contact       |\n");
    printf("|  2. View List         |\n");
    printf("|  3. Search            |\n");
    printf("|  4. Edit              |\n");
    printf("|  5. Delete            |\n");
    printf("|  6. Save to File      |\n");
    printf("|  0. Exit              |\n");
    printf("-------------------------\n");
}

void add_contact(AddressBook *ab) {
    if (ab->count >= MAX_CONTACTS) {
        printf("Address book is full. (Max %d)\n", MAX_CONTACTS);
        return;
    }

    Contact *c = &ab->contacts[ab->count];
    c->id = ab->next_id++;

    printf("\n=== Add New Contact ===\n\n");

    printf("Name: ");
    fgets(c->name, NAME_LEN, stdin);
    c->name[strcspn(c->name, "\n")] = '\0';

    if (strlen(c->name) == 0) {
        printf("Name is required. Addition cancelled.\n");
        return;
    }

    printf("Phone: ");
    fgets(c->phone, PHONE_LEN, stdin);
    c->phone[strcspn(c->phone, "\n")] = '\0';

    printf("Email: ");
    fgets(c->email, EMAIL_LEN, stdin);
    c->email[strcspn(c->email, "\n")] = '\0';

    ab->count++;
    printf("\nContact '%s' added. (ID: %d)\n", c->name, c->id);
}

void list_contacts(AddressBook *ab) {
    printf("\n=== Contact List === (Total: %d)\n", ab->count);

    if (ab->count == 0) {
        printf("\nNo saved contacts.\n");
        return;
    }

    printf("\n%-4s | %-15s | %-15s | %-20s\n", "ID", "Name", "Phone", "Email");
    printf("-----|-----------------|-----------------|---------------------\n");

    for (int i = 0; i < ab->count; i++) {
        Contact *c = &ab->contacts[i];
        printf("%-4d | %-15s | %-15s | %-20s\n",
               c->id, c->name, c->phone, c->email);
    }
}

void search_contact(AddressBook *ab) {
    char keyword[NAME_LEN];
    int found = 0;

    printf("\n=== Search Contact ===\n\n");
    printf("Search term: ");
    fgets(keyword, NAME_LEN, stdin);
    keyword[strcspn(keyword, "\n")] = '\0';

    if (strlen(keyword) == 0) {
        printf("Please enter a search term.\n");
        return;
    }

    printf("\nSearch Results:\n");
    printf("-----------------------------------------------------\n");

    for (int i = 0; i < ab->count; i++) {
        Contact *c = &ab->contacts[i];
        // Search in name, phone, email
        if (strstr(c->name, keyword) != NULL ||
            strstr(c->phone, keyword) != NULL ||
            strstr(c->email, keyword) != NULL) {

            printf("ID: %d\n", c->id);
            printf("  Name: %s\n", c->name);
            printf("  Phone: %s\n", c->phone);
            printf("  Email: %s\n", c->email);
            printf("-----------------------------------------------------\n");
            found++;
        }
    }

    if (found == 0) {
        printf("No results for '%s'.\n", keyword);
    } else {
        printf("%d result(s) found\n", found);
    }
}

void edit_contact(AddressBook *ab) {
    int id;
    char input[EMAIL_LEN];

    printf("\n=== Edit Contact ===\n\n");
    printf("Contact ID to edit: ");
    scanf("%d", &id);
    clear_input_buffer();

    int idx = find_by_id(ab, id);
    if (idx == -1) {
        printf("Contact with that ID not found.\n");
        return;
    }

    Contact *c = &ab->contacts[idx];

    printf("\nCurrent info:\n");
    printf("  Name: %s\n", c->name);
    printf("  Phone: %s\n", c->phone);
    printf("  Email: %s\n", c->email);

    printf("\nEnter new info (leave blank to keep current):\n");

    printf("Name [%s]: ", c->name);
    fgets(input, NAME_LEN, stdin);
    input[strcspn(input, "\n")] = '\0';
    if (strlen(input) > 0) {
        strcpy(c->name, input);
    }

    printf("Phone [%s]: ", c->phone);
    fgets(input, PHONE_LEN, stdin);
    input[strcspn(input, "\n")] = '\0';
    if (strlen(input) > 0) {
        strcpy(c->phone, input);
    }

    printf("Email [%s]: ", c->email);
    fgets(input, EMAIL_LEN, stdin);
    input[strcspn(input, "\n")] = '\0';
    if (strlen(input) > 0) {
        strcpy(c->email, input);
    }

    printf("\nContact updated.\n");
}

void delete_contact(AddressBook *ab) {
    int id;

    printf("\n=== Delete Contact ===\n\n");
    printf("Contact ID to delete: ");
    scanf("%d", &id);
    clear_input_buffer();

    int idx = find_by_id(ab, id);
    if (idx == -1) {
        printf("Contact with that ID not found.\n");
        return;
    }

    printf("Delete '%s'? (y/n): ", ab->contacts[idx].name);
    char confirm;
    scanf(" %c", &confirm);
    clear_input_buffer();

    if (confirm != 'y' && confirm != 'Y') {
        printf("Deletion cancelled.\n");
        return;
    }

    // Delete: shift elements forward
    for (int i = idx; i < ab->count - 1; i++) {
        ab->contacts[i] = ab->contacts[i + 1];
    }
    ab->count--;

    printf("Contact deleted.\n");
}

int save_to_file(AddressBook *ab) {
    FILE *fp = fopen(FILENAME, "wb");
    if (fp == NULL) {
        printf("Save failed: Could not open file.\n");
        return -1;
    }

    // Save metadata
    fwrite(&ab->count, sizeof(int), 1, fp);
    fwrite(&ab->next_id, sizeof(int), 1, fp);

    // Save contacts
    fwrite(ab->contacts, sizeof(Contact), ab->count, fp);

    fclose(fp);
    return 0;
}

int load_from_file(AddressBook *ab) {
    FILE *fp = fopen(FILENAME, "rb");
    if (fp == NULL) {
        // No file means fresh start
        return -1;
    }

    // Read metadata
    fread(&ab->count, sizeof(int), 1, fp);
    fread(&ab->next_id, sizeof(int), 1, fp);

    // Read contacts
    fread(ab->contacts, sizeof(Contact), ab->count, fp);

    fclose(fp);
    return 0;
}

int find_by_id(AddressBook *ab, int id) {
    for (int i = 0; i < ab->count; i++) {
        if (ab->contacts[i].id == id) {
            return i;
        }
    }
    return -1;
}

void clear_input_buffer(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}
```

---

## Compile and Run

```bash
gcc -Wall -Wextra -std=c11 addressbook.c -o addressbook
./addressbook
```

---

## Example Output

```
Loaded existing data. (3 contacts)

===============================
|      Address Book           |
===============================

-------------------------
|  1. Add Contact       |
|  2. View List         |
|  0. Exit              |
-------------------------
Choice: 2

=== Contact List === (Total: 3)

ID   | Name            | Phone           | Email
-----|-----------------|-----------------|---------------------
1    | John Doe        | 010-1234-5678   | john@email.com
2    | Jane Smith      | 010-9876-5432   | jane@email.com
3    | Bob Wilson      | 010-5555-5555   | bob@email.com
```

---

## Summary

| Concept | Description |
|---------|-------------|
| `typedef struct` | Struct type definition |
| `strcpy`, `strstr` | String copy, search |
| `fopen`, `fclose` | File open/close |
| `fread`, `fwrite` | Binary read/write |
| `fprintf`, `fscanf` | Text read/write |
| Struct pointer | Access members with `->` operator |

---

## Exercises

1. **Sort feature**: Add sorting by name or ID

2. **CSV export**: Export contacts to CSV file
   ```c
   // contacts.csv
   // id,name,phone,email
   // 1,John Doe,010-1234-5678,john@email.com
   ```

3. **Group feature**: Add group property to contacts (Family, Friends, Work)

---

## Next Step

[Project 4: Dynamic Array](./06_Project_Dynamic_Array.md) -> Let's learn about dynamic memory allocation!
