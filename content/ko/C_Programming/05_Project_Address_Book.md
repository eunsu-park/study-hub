# 프로젝트 3: 주소록 프로그램

**이전**: [프로젝트 2: 숫자 맞추기 게임](./04_Project_Number_Guessing.md) | **다음**: [프로젝트 4: 동적 배열](./06_Project_Dynamic_Array.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `typedef struct`를 사용하여 관련 필드(이름, 전화번호, 이메일)를 단일 레코드 타입으로 묶는 설계를 할 수 있습니다
2. 구조체 배열과 카운트 변수를 이용해 활성 항목을 추적하는 고정 크기 배열을 구축할 수 있습니다
3. 구조체 포인터와 배열 요소 이동을 활용하여 완전한 CRUD 연산(생성(Create), 읽기(Read), 수정(Update), 삭제(Delete))을 구현할 수 있습니다
4. 파일 I/O 함수(`fopen`, `fread`, `fwrite`, `fclose`)를 적용하여 세션 간 바이너리 데이터를 영속화할 수 있습니다
5. 텍스트 모드(`fprintf`/`fscanf`)와 바이너리 모드(`fread`/`fwrite`) 파일 연산의 차이를 구별할 수 있습니다
6. `fgets`와 `strcspn`을 함께 사용하여 문자열 입력을 안전하게 읽고 후행 개행 문자를 제거할 수 있습니다
7. `strstr`을 이용한 부분 문자열 검색으로 여러 필드에 걸쳐 레코드를 검색할 수 있습니다

---

주소록은 구조체와 파일 I/O를 익히기 위한 고전적인 실습 과제입니다. 실제 데이터베이스 애플리케이션과 동일한 패턴을 따르기 때문입니다. 레코드를 생성하고, 목록을 조회하고, 키워드로 검색하고, 필드를 수정하고, 항목을 삭제하는 — 완전한 CRUD 사이클을 모두 경험합니다. 연락처를 바이너리 파일에 저장함으로써 휘발성 메모리 내 데이터와 영속적인 디스크 저장 데이터의 차이도 직접 체험하게 됩니다. 이 차이는 모든 실제 프로그램에서 중요한 개념입니다.

## 프로그램 요구사항

```
1. 연락처 추가
2. 연락처 목록 보기
3. 연락처 검색
4. 연락처 수정
5. 연락처 삭제
6. 파일로 저장/불러오기
```

---

## 1단계: 구조체 정의

### 핵심 문법: 구조체

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

### 구조체 배열

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

## 2단계: 기본 CRUD 구현

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

## 3단계: 파일 저장/불러오기 추가

### 핵심 문법: 파일 I/O

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

### 파일 모드

| 모드 | 설명 |
|------|------|
| `"r"` | 읽기 (파일 없으면 실패) |
| `"w"` | 쓰기 (파일 덮어씀) |
| `"a"` | 추가 (파일 끝에 추가) |
| `"rb"` | 바이너리 읽기 |
| `"wb"` | 바이너리 쓰기 |

---

## 4단계: 최종 버전

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

## 컴파일 및 실행

```bash
gcc -Wall -Wextra -std=c11 addressbook.c -o addressbook
./addressbook
```

---

## 실행 예시

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

## 배운 내용 정리

| 개념 | 설명 |
|------|------|
| `typedef struct` | 구조체 타입 정의 |
| `strcpy`, `strstr` | 문자열 복사, 검색 |
| `fopen`, `fclose` | 파일 열기/닫기 |
| `fread`, `fwrite` | 바이너리 읽기/쓰기 |
| `fprintf`, `fscanf` | 텍스트 읽기/쓰기 |
| 구조체 포인터 | `->` 연산자로 멤버 접근 |

---

## 연습 문제

1. **정렬 기능**: 이름순 또는 ID순 정렬 기능 추가

2. **CSV 내보내기**: 연락처를 CSV 파일로 내보내기
   ```c
   // contacts.csv
   // id,name,phone,email
   // 1,John Doe,010-1234-5678,john@email.com
   ```

3. **그룹 기능**: 연락처에 그룹(가족, 친구, 직장) 속성 추가

---

## 다음 단계

[프로젝트 4: 동적 배열 (Dynamic Array)](./06_Project_Dynamic_Array.md) → 동적 메모리 할당을 배워봅시다!

---

**이전**: [프로젝트 2: 숫자 맞추기 게임](./04_Project_Number_Guessing.md) | **다음**: [프로젝트 4: 동적 배열](./06_Project_Dynamic_Array.md)
