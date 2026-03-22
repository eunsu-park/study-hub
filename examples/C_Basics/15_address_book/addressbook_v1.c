/*
 * addressbook_v1.c
 *
 * Address Book Program - Full CRUD functionality
 *
 * Features:
 *   1. Add contact (Create)
 *   2. List contacts (Read)
 *   3. Search contacts (Read)
 *   4. Edit contact (Update)
 *   5. Delete contact (Delete)
 *   6. File save/load (Persistence)
 *
 * Compile: gcc -Wall -Wextra -std=c11 addressbook_v1.c -o addressbook
 * Run: ./addressbook
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Constants */
#define MAX_CONTACTS 100
#define NAME_LEN 50
#define PHONE_LEN 20
#define EMAIL_LEN 50
#define FILENAME "contacts.dat"

/* Contact struct */
typedef struct {
    int id;
    char name[NAME_LEN];
    char phone[PHONE_LEN];
    char email[EMAIL_LEN];
} Contact;

/* Address book struct */
typedef struct {
    Contact contacts[MAX_CONTACTS];
    int count;      // Number of currently stored contacts
    int next_id;    // Next ID to assign
} AddressBook;

/* Function declarations */
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

/* Main function */
int main(void) {
    AddressBook ab;
    int choice;

    /* Initialize address book */
    init_addressbook(&ab);

    /* Load existing data from file */
    if (load_from_file(&ab) == 0) {
        printf("Loaded existing data. (%d contacts)\n", ab.count);
    }

    /* Program start message */
    printf("\n╔═══════════════════════════════╗\n");
    printf("║      Address Book Program     ║\n");
    printf("╚═══════════════════════════════╝\n");

    /* Main loop */
    while (1) {
        print_menu();
        printf("Choice: ");

        /* Menu choice input */
        if (scanf("%d", &choice) != 1) {
            clear_input_buffer();
            printf("Please enter a number.\n");
            continue;
        }
        clear_input_buffer();

        /* Handle menu */
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
                /* Confirm save before exit */
                printf("Save changes? (y/n): ");
                char save_confirm;
                scanf(" %c", &save_confirm);
                if (save_confirm == 'y' || save_confirm == 'Y') {
                    save_to_file(&ab);
                    printf("Save complete.\n");
                }
                printf("Exiting the program.\n");
                return 0;
            default:
                printf("Invalid choice.\n");
        }
        printf("\n");
    }

    return 0;
}

/* Initialize address book */
void init_addressbook(AddressBook *ab) {
    ab->count = 0;
    ab->next_id = 1;
    memset(ab->contacts, 0, sizeof(ab->contacts));
}

/* Print menu */
void print_menu(void) {
    printf("\n┌─────────────────────────┐\n");
    printf("│  1. Add contact         │\n");
    printf("│  2. List contacts       │\n");
    printf("│  3. Search              │\n");
    printf("│  4. Edit                │\n");
    printf("│  5. Delete              │\n");
    printf("│  6. Save to file        │\n");
    printf("│  0. Quit                │\n");
    printf("└─────────────────────────┘\n");
}

/* Add contact */
void add_contact(AddressBook *ab) {
    /* Check if address book is full */
    if (ab->count >= MAX_CONTACTS) {
        printf("Address book is full. (max %d contacts)\n", MAX_CONTACTS);
        return;
    }

    /* Pointer for new contact */
    Contact *c = &ab->contacts[ab->count];
    c->id = ab->next_id++;

    printf("\n=== Add New Contact ===\n\n");

    /* Name input (required) */
    printf("Name: ");
    fgets(c->name, NAME_LEN, stdin);
    c->name[strcspn(c->name, "\n")] = '\0';  // Remove newline character

    if (strlen(c->name) == 0) {
        printf("Name is required. Addition cancelled.\n");
        return;
    }

    /* Phone number input */
    printf("Phone: ");
    fgets(c->phone, PHONE_LEN, stdin);
    c->phone[strcspn(c->phone, "\n")] = '\0';

    /* Email input */
    printf("Email: ");
    fgets(c->email, EMAIL_LEN, stdin);
    c->email[strcspn(c->email, "\n")] = '\0';

    /* Increment contact count */
    ab->count++;
    printf("\nContact '%s' added. (ID: %d)\n", c->name, c->id);
}

/* List contacts */
void list_contacts(AddressBook *ab) {
    printf("\n=== Contact List === (%d total)\n", ab->count);

    if (ab->count == 0) {
        printf("\nNo contacts stored.\n");
        return;
    }

    /* Table header */
    printf("\n%-4s | %-15s | %-15s | %-20s\n", "ID", "Name", "Phone", "Email");
    printf("─────┼─────────────────┼─────────────────┼─────────────────────\n");

    /* Print all contacts */
    for (int i = 0; i < ab->count; i++) {
        Contact *c = &ab->contacts[i];
        printf("%-4d | %-15s | %-15s | %-20s\n",
               c->id, c->name, c->phone, c->email);
    }
}

/* Search contacts */
void search_contact(AddressBook *ab) {
    char keyword[NAME_LEN];
    int found = 0;

    printf("\n=== Search Contacts ===\n\n");
    printf("Keyword: ");
    fgets(keyword, NAME_LEN, stdin);
    keyword[strcspn(keyword, "\n")] = '\0';

    if (strlen(keyword) == 0) {
        printf("Please enter a keyword.\n");
        return;
    }

    printf("\nSearch results:\n");
    printf("─────────────────────────────────────────────────────\n");

    /* Search all contacts */
    for (int i = 0; i < ab->count; i++) {
        Contact *c = &ab->contacts[i];
        /* Substring search in name, phone, and email */
        if (strstr(c->name, keyword) != NULL ||
            strstr(c->phone, keyword) != NULL ||
            strstr(c->email, keyword) != NULL) {

            printf("ID: %d\n", c->id);
            printf("  Name:  %s\n", c->name);
            printf("  Phone: %s\n", c->phone);
            printf("  Email: %s\n", c->email);
            printf("─────────────────────────────────────────────────────\n");
            found++;
        }
    }

    if (found == 0) {
        printf("No results found for '%s'.\n", keyword);
    } else {
        printf("%d result(s) found\n", found);
    }
}

/* Edit contact */
void edit_contact(AddressBook *ab) {
    int id;
    char input[EMAIL_LEN];

    printf("\n=== Edit Contact ===\n\n");
    printf("Contact ID to edit: ");
    scanf("%d", &id);
    clear_input_buffer();

    /* Find contact by ID */
    int idx = find_by_id(ab, id);
    if (idx == -1) {
        printf("Contact with that ID not found.\n");
        return;
    }

    Contact *c = &ab->contacts[idx];

    /* Display current info */
    printf("\nCurrent info:\n");
    printf("  Name:  %s\n", c->name);
    printf("  Phone: %s\n", c->phone);
    printf("  Email: %s\n", c->email);

    printf("\nEnter new info (leave blank to keep current):\n");

    /* Edit name */
    printf("Name [%s]: ", c->name);
    fgets(input, NAME_LEN, stdin);
    input[strcspn(input, "\n")] = '\0';
    if (strlen(input) > 0) {
        strcpy(c->name, input);
    }

    /* Edit phone number */
    printf("Phone [%s]: ", c->phone);
    fgets(input, PHONE_LEN, stdin);
    input[strcspn(input, "\n")] = '\0';
    if (strlen(input) > 0) {
        strcpy(c->phone, input);
    }

    /* Edit email */
    printf("Email [%s]: ", c->email);
    fgets(input, EMAIL_LEN, stdin);
    input[strcspn(input, "\n")] = '\0';
    if (strlen(input) > 0) {
        strcpy(c->email, input);
    }

    printf("\nContact updated.\n");
}

/* Delete contact */
void delete_contact(AddressBook *ab) {
    int id;

    printf("\n=== Delete Contact ===\n\n");
    printf("Contact ID to delete: ");
    scanf("%d", &id);
    clear_input_buffer();

    /* Find contact by ID */
    int idx = find_by_id(ab, id);
    if (idx == -1) {
        printf("Contact with that ID not found.\n");
        return;
    }

    /* Confirm deletion */
    printf("Delete contact '%s'? (y/n): ", ab->contacts[idx].name);
    char confirm;
    scanf(" %c", &confirm);
    clear_input_buffer();

    if (confirm != 'y' && confirm != 'Y') {
        printf("Deletion cancelled.\n");
        return;
    }

    /* Delete: shift elements forward */
    for (int i = idx; i < ab->count - 1; i++) {
        ab->contacts[i] = ab->contacts[i + 1];
    }
    ab->count--;

    printf("Contact deleted.\n");
}

/* Save to file (binary mode) */
int save_to_file(AddressBook *ab) {
    FILE *fp = fopen(FILENAME, "wb");
    if (fp == NULL) {
        printf("Failed to save: Cannot open file.\n");
        return -1;
    }

    /* Save metadata (count, next_id) */
    fwrite(&ab->count, sizeof(int), 1, fp);
    fwrite(&ab->next_id, sizeof(int), 1, fp);

    /* Save contacts array */
    fwrite(ab->contacts, sizeof(Contact), ab->count, fp);

    fclose(fp);
    return 0;
}

/* Load from file (binary mode) */
int load_from_file(AddressBook *ab) {
    FILE *fp = fopen(FILENAME, "rb");
    if (fp == NULL) {
        /* If file doesn't exist, start fresh */
        return -1;
    }

    /* Read metadata */
    fread(&ab->count, sizeof(int), 1, fp);
    fread(&ab->next_id, sizeof(int), 1, fp);

    /* Read contacts array */
    fread(ab->contacts, sizeof(Contact), ab->count, fp);

    fclose(fp);
    return 0;
}

/* Find contact by ID (returns index) */
int find_by_id(AddressBook *ab, int id) {
    for (int i = 0; i < ab->count; i++) {
        if (ab->contacts[i].id == id) {
            return i;
        }
    }
    return -1;  /* Not found */
}

/* Clear input buffer */
void clear_input_buffer(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}
