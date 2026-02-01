# 프로젝트 3: 주소록 프로그램

## 학습 목표

이 프로젝트를 통해 배우는 내용:
- 구조체 정의와 활용
- 구조체 배열
- 파일 입출력 (fopen, fwrite, fread, fprintf, fscanf)
- CRUD 기능 구현 (Create, Read, Update, Delete)

---

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

// 연락처 구조체 정의
typedef struct {
    int id;
    char name[50];
    char phone[20];
    char email[50];
} Contact;

int main(void) {
    // 구조체 변수 선언
    Contact c1;

    // 멤버에 값 할당
    c1.id = 1;
    strcpy(c1.name, "홍길동");
    strcpy(c1.phone, "010-1234-5678");
    strcpy(c1.email, "hong@email.com");

    // 출력
    printf("ID: %d\n", c1.id);
    printf("이름: %s\n", c1.name);
    printf("전화: %s\n", c1.phone);
    printf("이메일: %s\n", c1.email);

    // 초기화와 함께 선언
    Contact c2 = {2, "김철수", "010-9876-5432", "kim@email.com"};
    printf("\n%s: %s\n", c2.name, c2.phone);

    return 0;
}
```

### 구조체 배열

```c
#define MAX_CONTACTS 100

Contact contacts[MAX_CONTACTS];
int contact_count = 0;

// 추가
contacts[contact_count].id = contact_count + 1;
strcpy(contacts[contact_count].name, "새 연락처");
contact_count++;

// 순회
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

// 연락처 구조체
typedef struct {
    int id;
    char name[NAME_LEN];
    char phone[PHONE_LEN];
    char email[EMAIL_LEN];
} Contact;

// 전역 변수
Contact contacts[MAX_CONTACTS];
int contact_count = 0;
int next_id = 1;

// 함수 선언
void print_menu(void);
void add_contact(void);
void list_contacts(void);
void search_contact(void);
void delete_contact(void);
void clear_input_buffer(void);

int main(void) {
    int choice;

    printf("=== 주소록 프로그램 v1 ===\n");

    while (1) {
        print_menu();
        printf("선택: ");

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
                printf("프로그램을 종료합니다.\n");
                return 0;
            default:
                printf("잘못된 선택입니다.\n");
        }
        printf("\n");
    }

    return 0;
}

void print_menu(void) {
    printf("\n");
    printf("┌────────────────────┐\n");
    printf("│  1. 연락처 추가    │\n");
    printf("│  2. 목록 보기      │\n");
    printf("│  3. 검색           │\n");
    printf("│  4. 삭제           │\n");
    printf("│  0. 종료           │\n");
    printf("└────────────────────┘\n");
}

void add_contact(void) {
    if (contact_count >= MAX_CONTACTS) {
        printf("주소록이 가득 찼습니다.\n");
        return;
    }

    Contact *c = &contacts[contact_count];
    c->id = next_id++;

    printf("\n[ 새 연락처 추가 ]\n");

    printf("이름: ");
    fgets(c->name, NAME_LEN, stdin);
    c->name[strcspn(c->name, "\n")] = '\0';  // 개행 제거

    printf("전화번호: ");
    fgets(c->phone, PHONE_LEN, stdin);
    c->phone[strcspn(c->phone, "\n")] = '\0';

    printf("이메일: ");
    fgets(c->email, EMAIL_LEN, stdin);
    c->email[strcspn(c->email, "\n")] = '\0';

    contact_count++;
    printf("\n연락처가 추가되었습니다. (ID: %d)\n", c->id);
}

void list_contacts(void) {
    printf("\n[ 연락처 목록 ] (총 %d명)\n", contact_count);
    printf("────────────────────────────────────────\n");

    if (contact_count == 0) {
        printf("저장된 연락처가 없습니다.\n");
        return;
    }

    for (int i = 0; i < contact_count; i++) {
        Contact *c = &contacts[i];
        printf("ID: %d\n", c->id);
        printf("  이름: %s\n", c->name);
        printf("  전화: %s\n", c->phone);
        printf("  이메일: %s\n", c->email);
        printf("────────────────────────────────────────\n");
    }
}

void search_contact(void) {
    char keyword[NAME_LEN];
    int found = 0;

    printf("\n[ 연락처 검색 ]\n");
    printf("검색어 (이름): ");
    fgets(keyword, NAME_LEN, stdin);
    keyword[strcspn(keyword, "\n")] = '\0';

    printf("\n검색 결과:\n");
    printf("────────────────────────────────────────\n");

    for (int i = 0; i < contact_count; i++) {
        // 부분 문자열 검색 (strstr)
        if (strstr(contacts[i].name, keyword) != NULL) {
            Contact *c = &contacts[i];
            printf("ID: %d | %s | %s\n", c->id, c->name, c->phone);
            found++;
        }
    }

    if (found == 0) {
        printf("검색 결과가 없습니다.\n");
    } else {
        printf("\n총 %d건 검색됨\n", found);
    }
}

void delete_contact(void) {
    int id, found = -1;

    printf("\n[ 연락처 삭제 ]\n");
    printf("삭제할 ID: ");
    scanf("%d", &id);
    clear_input_buffer();

    // ID로 찾기
    for (int i = 0; i < contact_count; i++) {
        if (contacts[i].id == id) {
            found = i;
            break;
        }
    }

    if (found == -1) {
        printf("해당 ID의 연락처를 찾을 수 없습니다.\n");
        return;
    }

    // 확인
    printf("'%s' 연락처를 삭제하시겠습니까? (y/n): ", contacts[found].name);
    char confirm;
    scanf(" %c", &confirm);
    clear_input_buffer();

    if (confirm != 'y' && confirm != 'Y') {
        printf("삭제가 취소되었습니다.\n");
        return;
    }

    // 삭제: 뒤의 요소들을 앞으로 이동
    for (int i = found; i < contact_count - 1; i++) {
        contacts[i] = contacts[i + 1];
    }
    contact_count--;

    printf("연락처가 삭제되었습니다.\n");
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

// 텍스트 파일 쓰기
FILE *fp = fopen("data.txt", "w");
if (fp == NULL) {
    printf("파일 열기 실패\n");
    return;
}
fprintf(fp, "Hello, File!\n");
fprintf(fp, "Number: %d\n", 42);
fclose(fp);

// 텍스트 파일 읽기
FILE *fp = fopen("data.txt", "r");
char line[100];
while (fgets(line, sizeof(line), fp) != NULL) {
    printf("%s", line);
}
fclose(fp);

// 바이너리 파일 쓰기 (구조체 저장에 유용)
FILE *fp = fopen("data.bin", "wb");
Contact c = {1, "홍길동", "010-1234", "hong@test.com"};
fwrite(&c, sizeof(Contact), 1, fp);
fclose(fp);

// 바이너리 파일 읽기
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
// addressbook.c (최종)
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_CONTACTS 100
#define NAME_LEN 50
#define PHONE_LEN 20
#define EMAIL_LEN 50
#define FILENAME "contacts.dat"

// 연락처 구조체
typedef struct {
    int id;
    char name[NAME_LEN];
    char phone[PHONE_LEN];
    char email[EMAIL_LEN];
} Contact;

// 주소록 구조체
typedef struct {
    Contact contacts[MAX_CONTACTS];
    int count;
    int next_id;
} AddressBook;

// 함수 선언
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

    // 파일에서 데이터 불러오기
    if (load_from_file(&ab) == 0) {
        printf("기존 데이터를 불러왔습니다. (%d명)\n", ab.count);
    }

    printf("\n╔═══════════════════════════════╗\n");
    printf("║      📒 주소록 프로그램       ║\n");
    printf("╚═══════════════════════════════╝\n");

    while (1) {
        print_menu();
        printf("선택: ");

        if (scanf("%d", &choice) != 1) {
            clear_input_buffer();
            printf("숫자를 입력해주세요.\n");
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
                    printf("파일에 저장되었습니다.\n");
                }
                break;
            case 0:
                // 종료 전 저장 확인
                printf("변경 사항을 저장하시겠습니까? (y/n): ");
                char save_confirm;
                scanf(" %c", &save_confirm);
                if (save_confirm == 'y' || save_confirm == 'Y') {
                    save_to_file(&ab);
                    printf("저장 완료.\n");
                }
                printf("프로그램을 종료합니다.\n");
                return 0;
            default:
                printf("잘못된 선택입니다.\n");
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
    printf("\n┌─────────────────────────┐\n");
    printf("│  1. 연락처 추가         │\n");
    printf("│  2. 목록 보기           │\n");
    printf("│  3. 검색                │\n");
    printf("│  4. 수정                │\n");
    printf("│  5. 삭제                │\n");
    printf("│  6. 파일 저장           │\n");
    printf("│  0. 종료                │\n");
    printf("└─────────────────────────┘\n");
}

void add_contact(AddressBook *ab) {
    if (ab->count >= MAX_CONTACTS) {
        printf("주소록이 가득 찼습니다. (최대 %d명)\n", MAX_CONTACTS);
        return;
    }

    Contact *c = &ab->contacts[ab->count];
    c->id = ab->next_id++;

    printf("\n═══ 새 연락처 추가 ═══\n\n");

    printf("이름: ");
    fgets(c->name, NAME_LEN, stdin);
    c->name[strcspn(c->name, "\n")] = '\0';

    if (strlen(c->name) == 0) {
        printf("이름은 필수입니다. 추가가 취소되었습니다.\n");
        return;
    }

    printf("전화번호: ");
    fgets(c->phone, PHONE_LEN, stdin);
    c->phone[strcspn(c->phone, "\n")] = '\0';

    printf("이메일: ");
    fgets(c->email, EMAIL_LEN, stdin);
    c->email[strcspn(c->email, "\n")] = '\0';

    ab->count++;
    printf("\n✓ '%s' 연락처가 추가되었습니다. (ID: %d)\n", c->name, c->id);
}

void list_contacts(AddressBook *ab) {
    printf("\n═══ 연락처 목록 ═══ (총 %d명)\n", ab->count);

    if (ab->count == 0) {
        printf("\n저장된 연락처가 없습니다.\n");
        return;
    }

    printf("\n%-4s │ %-15s │ %-15s │ %-20s\n", "ID", "이름", "전화번호", "이메일");
    printf("─────┼─────────────────┼─────────────────┼─────────────────────\n");

    for (int i = 0; i < ab->count; i++) {
        Contact *c = &ab->contacts[i];
        printf("%-4d │ %-15s │ %-15s │ %-20s\n",
               c->id, c->name, c->phone, c->email);
    }
}

void search_contact(AddressBook *ab) {
    char keyword[NAME_LEN];
    int found = 0;

    printf("\n═══ 연락처 검색 ═══\n\n");
    printf("검색어: ");
    fgets(keyword, NAME_LEN, stdin);
    keyword[strcspn(keyword, "\n")] = '\0';

    if (strlen(keyword) == 0) {
        printf("검색어를 입력해주세요.\n");
        return;
    }

    printf("\n검색 결과:\n");
    printf("─────────────────────────────────────────────────────\n");

    for (int i = 0; i < ab->count; i++) {
        Contact *c = &ab->contacts[i];
        // 이름, 전화번호, 이메일에서 검색
        if (strstr(c->name, keyword) != NULL ||
            strstr(c->phone, keyword) != NULL ||
            strstr(c->email, keyword) != NULL) {

            printf("ID: %d\n", c->id);
            printf("  이름: %s\n", c->name);
            printf("  전화: %s\n", c->phone);
            printf("  이메일: %s\n", c->email);
            printf("─────────────────────────────────────────────────────\n");
            found++;
        }
    }

    if (found == 0) {
        printf("'%s'에 대한 검색 결과가 없습니다.\n", keyword);
    } else {
        printf("총 %d건 검색됨\n", found);
    }
}

void edit_contact(AddressBook *ab) {
    int id;
    char input[EMAIL_LEN];

    printf("\n═══ 연락처 수정 ═══\n\n");
    printf("수정할 연락처 ID: ");
    scanf("%d", &id);
    clear_input_buffer();

    int idx = find_by_id(ab, id);
    if (idx == -1) {
        printf("해당 ID의 연락처를 찾을 수 없습니다.\n");
        return;
    }

    Contact *c = &ab->contacts[idx];

    printf("\n현재 정보:\n");
    printf("  이름: %s\n", c->name);
    printf("  전화: %s\n", c->phone);
    printf("  이메일: %s\n", c->email);

    printf("\n새 정보를 입력하세요 (빈 칸: 유지):\n");

    printf("이름 [%s]: ", c->name);
    fgets(input, NAME_LEN, stdin);
    input[strcspn(input, "\n")] = '\0';
    if (strlen(input) > 0) {
        strcpy(c->name, input);
    }

    printf("전화번호 [%s]: ", c->phone);
    fgets(input, PHONE_LEN, stdin);
    input[strcspn(input, "\n")] = '\0';
    if (strlen(input) > 0) {
        strcpy(c->phone, input);
    }

    printf("이메일 [%s]: ", c->email);
    fgets(input, EMAIL_LEN, stdin);
    input[strcspn(input, "\n")] = '\0';
    if (strlen(input) > 0) {
        strcpy(c->email, input);
    }

    printf("\n✓ 연락처가 수정되었습니다.\n");
}

void delete_contact(AddressBook *ab) {
    int id;

    printf("\n═══ 연락처 삭제 ═══\n\n");
    printf("삭제할 연락처 ID: ");
    scanf("%d", &id);
    clear_input_buffer();

    int idx = find_by_id(ab, id);
    if (idx == -1) {
        printf("해당 ID의 연락처를 찾을 수 없습니다.\n");
        return;
    }

    printf("'%s' 연락처를 삭제하시겠습니까? (y/n): ", ab->contacts[idx].name);
    char confirm;
    scanf(" %c", &confirm);
    clear_input_buffer();

    if (confirm != 'y' && confirm != 'Y') {
        printf("삭제가 취소되었습니다.\n");
        return;
    }

    // 삭제: 뒤의 요소들을 앞으로 이동
    for (int i = idx; i < ab->count - 1; i++) {
        ab->contacts[i] = ab->contacts[i + 1];
    }
    ab->count--;

    printf("✓ 연락처가 삭제되었습니다.\n");
}

int save_to_file(AddressBook *ab) {
    FILE *fp = fopen(FILENAME, "wb");
    if (fp == NULL) {
        printf("파일 저장 실패: 파일을 열 수 없습니다.\n");
        return -1;
    }

    // 메타데이터 저장
    fwrite(&ab->count, sizeof(int), 1, fp);
    fwrite(&ab->next_id, sizeof(int), 1, fp);

    // 연락처 저장
    fwrite(ab->contacts, sizeof(Contact), ab->count, fp);

    fclose(fp);
    return 0;
}

int load_from_file(AddressBook *ab) {
    FILE *fp = fopen(FILENAME, "rb");
    if (fp == NULL) {
        // 파일이 없으면 새로 시작
        return -1;
    }

    // 메타데이터 읽기
    fread(&ab->count, sizeof(int), 1, fp);
    fread(&ab->next_id, sizeof(int), 1, fp);

    // 연락처 읽기
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
기존 데이터를 불러왔습니다. (3명)

╔═══════════════════════════════╗
║      📒 주소록 프로그램       ║
╚═══════════════════════════════╝

┌─────────────────────────┐
│  1. 연락처 추가         │
│  2. 목록 보기           │
│  3. 검색                │
│  0. 종료                │
└─────────────────────────┘
선택: 2

═══ 연락처 목록 ═══ (총 3명)

ID   │ 이름            │ 전화번호        │ 이메일
─────┼─────────────────┼─────────────────┼─────────────────────
1    │ 홍길동          │ 010-1234-5678   │ hong@email.com
2    │ 김철수          │ 010-9876-5432   │ kim@email.com
3    │ 이영희          │ 010-5555-5555   │ lee@email.com
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

1. **정렬 기능**: 이름순, ID순 정렬 기능 추가

2. **CSV 내보내기**: 연락처를 CSV 파일로 내보내기
   ```c
   // contacts.csv
   // id,name,phone,email
   // 1,홍길동,010-1234-5678,hong@email.com
   ```

3. **그룹 기능**: 연락처에 그룹(가족, 친구, 직장) 속성 추가

---

## 다음 단계

[06_Project_Dynamic_Array.md](./06_Project_Dynamic_Array.md) → 동적 메모리 할당을 배워봅시다!
