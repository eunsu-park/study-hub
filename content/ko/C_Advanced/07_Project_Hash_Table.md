# 프로젝트: 해시 테이블

**이전**: [프로젝트: 스택과 큐](./06_Project_Stack_Queue.md) | **다음**: [프로젝트: 파일 암호화](./08_Project_File_Encryption.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 해시 함수가 키를 배열 인덱스로 변환하는 방법과 균일 분포가 중요한 이유를 설명할 수 있다
2. 인기 있는 문자열 해시 함수(단순 합산, djb2, sdbm, FNV-1a)를 비교하고 충돌 동작을 평가할 수 있다
3. 버킷별 연결 리스트로 충돌을 해결하는 체이닝 해시 테이블을 구현할 수 있다
4. 선형 탐사(linear probing)와 DELETED 센티널을 사용하는 개방 주소법 해시 테이블을 구현할 수 있다
5. 충돌과 톰스톤을 올바르게 처리하는 삽입, 검색, 삭제 연산을 설계할 수 있다
6. 추가, 검색, 삭제, 목록 명령을 갖춘 완전한 대화형 사전 프로그램을 구축할 수 있다
7. 적재율(load factor)에 기반하여 해시 테이블을 리사이즈해야 할 시점을 식별하고 재해싱 과정을 설명할 수 있다

---

해시 테이블(Hash Table)은 Python의 `dict`, JavaScript의 `Object`, 그리고 거의 모든 프로덕션 소프트웨어의 키-값 저장소 뒤에 있는 자료 구조입니다. 해시 함수를 통해 키를 배열 인덱스로 변환하여 평균 O(1) 조회를 달성합니다 -- 직접 구축하고 불가피한 충돌에 맞서기 전까지는 마법처럼 보이는 트릭입니다. 이 프로젝트는 체이닝과 개방 주소법 전략을 처음부터 구현하여 그 마법을 해체합니다.

## 해시 테이블이란?

### 개념

해시 테이블은 해시 함수를 사용하여 키(Key)를 **인덱스**로 변환한 후, 해당 위치에 값(Value)을 저장합니다.

```
Key: "apple"
        |
해시 함수: hash("apple") = 3
        |
+---+---+---+---+---+---+---+
|   |   |   | X |   |   |   |  -> 인덱스 3에 저장
+---+---+---+---+---+---+---+
  0   1   2   3   4   5   6
```

### 시간 복잡도

| 연산 | 평균 | 최악 |
|------|------|------|
| 삽입 | O(1) | O(n) |
| 검색 | O(1) | O(n) |
| 삭제 | O(1) | O(n) |

최악의 경우: 모든 키가 같은 인덱스로 충돌할 때

---

## 단계 1: 해시 함수 이해

### 좋은 해시 함수의 조건

1. **결정적(Deterministic)**: 같은 입력 -> 항상 같은 출력
2. **균일 분포**: 출력이 고르게 분포
3. **빠른 계산**: O(1) 시간

### 문자열 해시 함수

```c
// hash_functions.c
#include <stdio.h>
#include <string.h>

#define TABLE_SIZE 10

// 1. 단순 합산 (나쁜 예시)
unsigned int hash_simple(const char *key) {
    unsigned int hash = 0;
    while (*key) {
        hash += *key++;
    }
    return hash % TABLE_SIZE;
}

// 2. djb2 (Daniel J. Bernstein) - 권장
unsigned int hash_djb2(const char *key) {
    unsigned int hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;  // hash * 33 + c
    }
    return hash % TABLE_SIZE;
}

// 3. sdbm
unsigned int hash_sdbm(const char *key) {
    unsigned int hash = 0;
    int c;
    while ((c = *key++)) {
        hash = c + (hash << 6) + (hash << 16) - hash;
    }
    return hash % TABLE_SIZE;
}

// 4. FNV-1a
unsigned int hash_fnv1a(const char *key) {
    unsigned int hash = 2166136261u;
    while (*key) {
        hash ^= (unsigned char)*key++;
        hash *= 16777619;
    }
    return hash % TABLE_SIZE;
}

int main(void) {
    const char *keys[] = {"apple", "banana", "cherry", "date", "elderberry"};
    int n = sizeof(keys) / sizeof(keys[0]);

    printf("=== 해시 함수 비교 ===\n\n");
    printf("%-12s | simple | djb2 | sdbm | fnv1a\n", "Key");
    printf("-------------|--------|------|------|------\n");

    for (int i = 0; i < n; i++) {
        printf("%-12s | %6u | %4u | %4u | %5u\n",
               keys[i],
               hash_simple(keys[i]),
               hash_djb2(keys[i]),
               hash_sdbm(keys[i]),
               hash_fnv1a(keys[i]));
    }

    return 0;
}
```

---

## 단계 2: 체이닝 (분리 연쇄법, Separate Chaining)

같은 인덱스에서의 충돌을 연결 리스트로 저장합니다.

```
인덱스 3에서 충돌:

+---+
| 0 | -> NULL
+---+
| 1 | -> NULL
+---+
| 2 | -> NULL
+---+
| 3 | -> [apple] -> [apricot] -> NULL  (체인)
+---+
| 4 | -> NULL
+---+
```

### 구현

```c
// hash_chaining.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define TABLE_SIZE 10
#define KEY_SIZE 50
#define VALUE_SIZE 100

// 노드 (키-값 쌍)
typedef struct Node {
    char key[KEY_SIZE];
    char value[VALUE_SIZE];
    struct Node *next;
} Node;

// 해시 테이블
typedef struct {
    Node *buckets[TABLE_SIZE];
    int count;
} HashTable;

// 해시 함수 (djb2)
unsigned int hash(const char *key) {
    unsigned int hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % TABLE_SIZE;
}

// 생성
HashTable* ht_create(void) {
    HashTable *ht = malloc(sizeof(HashTable));
    if (ht) {
        for (int i = 0; i < TABLE_SIZE; i++) {
            ht->buckets[i] = NULL;
        }
        ht->count = 0;
    }
    return ht;
}

// 소멸
void ht_destroy(HashTable *ht) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = ht->buckets[i];
        while (current) {
            Node *next = current->next;
            free(current);
            current = next;
        }
    }
    free(ht);
}

// 삽입/갱신
bool ht_set(HashTable *ht, const char *key, const char *value) {
    unsigned int index = hash(key);

    // 기존 키 찾기
    Node *current = ht->buckets[index];
    while (current) {
        if (strcmp(current->key, key) == 0) {
            // 기존 키 -> 값 갱신
            strncpy(current->value, value, VALUE_SIZE - 1);
            return true;
        }
        current = current->next;
    }

    // 새 노드 생성
    Node *node = malloc(sizeof(Node));
    if (!node) return false;

    strncpy(node->key, key, KEY_SIZE - 1);
    strncpy(node->value, value, VALUE_SIZE - 1);
    node->next = ht->buckets[index];
    ht->buckets[index] = node;
    ht->count++;

    return true;
}

// 검색
char* ht_get(HashTable *ht, const char *key) {
    unsigned int index = hash(key);

    Node *current = ht->buckets[index];
    while (current) {
        if (strcmp(current->key, key) == 0) {
            return current->value;
        }
        current = current->next;
    }

    return NULL;  // 찾지 못함
}

// 삭제
bool ht_delete(HashTable *ht, const char *key) {
    unsigned int index = hash(key);

    Node *current = ht->buckets[index];
    Node *prev = NULL;

    while (current) {
        if (strcmp(current->key, key) == 0) {
            if (prev) {
                prev->next = current->next;
            } else {
                ht->buckets[index] = current->next;
            }
            free(current);
            ht->count--;
            return true;
        }
        prev = current;
        current = current->next;
    }

    return false;  // 찾지 못함
}

// 출력
void ht_print(HashTable *ht) {
    printf("\n=== 해시 테이블 (count=%d) ===\n", ht->count);
    for (int i = 0; i < TABLE_SIZE; i++) {
        printf("[%d]: ", i);
        Node *current = ht->buckets[i];
        if (!current) {
            printf("(비어있음)");
        }
        while (current) {
            printf("(\"%s\": \"%s\")", current->key, current->value);
            if (current->next) printf(" -> ");
            current = current->next;
        }
        printf("\n");
    }
}

// 테스트
int main(void) {
    HashTable *ht = ht_create();

    printf("=== 체이닝 해시 테이블 ===\n");

    // 삽입
    ht_set(ht, "apple", "a fruit");
    ht_set(ht, "banana", "a tropical fruit");
    ht_set(ht, "cherry", "a small red fruit");
    ht_set(ht, "date", "a sweet fruit");
    ht_set(ht, "elderberry", "a berry");

    ht_print(ht);

    // 검색
    printf("\n검색 테스트:\n");
    printf("apple: %s\n", ht_get(ht, "apple") ?: "(찾지 못함)");
    printf("grape: %s\n", ht_get(ht, "grape") ?: "(찾지 못함)");

    // 갱신
    printf("\n갱신 테스트:\n");
    ht_set(ht, "apple", "a delicious fruit");
    printf("apple: %s\n", ht_get(ht, "apple"));

    // 삭제
    printf("\n삭제 테스트:\n");
    ht_delete(ht, "banana");
    ht_print(ht);

    ht_destroy(ht);
    return 0;
}
```

---

## 단계 3: 개방 주소법(Open Addressing)

충돌 발생 시 다른 빈 슬롯을 찾아 저장합니다.

### 선형 탐사(Linear Probing)

```
hash("apple") = 3, hash("apricot") = 3 (충돌!)

"apple" 삽입:
+---+---+---+---+---+---+---+
|   |   |   | X |   |   |   |
+---+---+---+---+---+---+---+
  0   1   2   3   4   5   6

"apricot" 삽입 (충돌 -> 다음 슬롯):
+---+---+---+---+---+---+---+
|   |   |   | X | Y |   |   |  <- 인덱스 4에 저장
+---+---+---+---+---+---+---+
  0   1   2   3   4   5   6
```

### 구현

```c
// hash_linear_probing.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define TABLE_SIZE 10
#define KEY_SIZE 50
#define VALUE_SIZE 100

// 슬롯 상태
typedef enum {
    EMPTY,      // 비어있음
    OCCUPIED,   // 사용 중
    DELETED     // 삭제됨 (검색 시 계속 탐사)
} SlotStatus;

// 슬롯
typedef struct {
    char key[KEY_SIZE];
    char value[VALUE_SIZE];
    SlotStatus status;
} Slot;

// 해시 테이블
typedef struct {
    Slot slots[TABLE_SIZE];
    int count;
} HashTable;

unsigned int hash(const char *key) {
    unsigned int hash = 5381;
    int c;
    while ((c = *key++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % TABLE_SIZE;
}

HashTable* ht_create(void) {
    HashTable *ht = malloc(sizeof(HashTable));
    if (ht) {
        for (int i = 0; i < TABLE_SIZE; i++) {
            ht->slots[i].status = EMPTY;
        }
        ht->count = 0;
    }
    return ht;
}

void ht_destroy(HashTable *ht) {
    free(ht);
}

// 삽입
bool ht_set(HashTable *ht, const char *key, const char *value) {
    if (ht->count >= TABLE_SIZE) {
        printf("해시 테이블이 가득 참!\n");
        return false;
    }

    unsigned int index = hash(key);
    unsigned int original_index = index;

    // 선형 탐사
    do {
        // 빈 슬롯이거나 같은 키
        if (ht->slots[index].status != OCCUPIED ||
            strcmp(ht->slots[index].key, key) == 0) {

            if (ht->slots[index].status != OCCUPIED) {
                ht->count++;
            }

            strncpy(ht->slots[index].key, key, KEY_SIZE - 1);
            strncpy(ht->slots[index].value, value, VALUE_SIZE - 1);
            ht->slots[index].status = OCCUPIED;
            return true;
        }

        index = (index + 1) % TABLE_SIZE;  // 다음 슬롯
    } while (index != original_index);

    return false;
}

// 검색
char* ht_get(HashTable *ht, const char *key) {
    unsigned int index = hash(key);
    unsigned int original_index = index;

    do {
        if (ht->slots[index].status == EMPTY) {
            return NULL;  // 찾지 못함
        }

        if (ht->slots[index].status == OCCUPIED &&
            strcmp(ht->slots[index].key, key) == 0) {
            return ht->slots[index].value;
        }

        index = (index + 1) % TABLE_SIZE;
    } while (index != original_index);

    return NULL;
}

// 삭제
bool ht_delete(HashTable *ht, const char *key) {
    unsigned int index = hash(key);
    unsigned int original_index = index;

    do {
        if (ht->slots[index].status == EMPTY) {
            return false;
        }

        if (ht->slots[index].status == OCCUPIED &&
            strcmp(ht->slots[index].key, key) == 0) {
            ht->slots[index].status = DELETED;  // EMPTY가 아닌 DELETED
            ht->count--;
            return true;
        }

        index = (index + 1) % TABLE_SIZE;
    } while (index != original_index);

    return false;
}

void ht_print(HashTable *ht) {
    printf("\n=== 해시 테이블 (count=%d) ===\n", ht->count);
    for (int i = 0; i < TABLE_SIZE; i++) {
        printf("[%d]: ", i);
        switch (ht->slots[i].status) {
            case EMPTY:
                printf("(비어있음)\n");
                break;
            case DELETED:
                printf("(삭제됨)\n");
                break;
            case OCCUPIED:
                printf("\"%s\": \"%s\"\n",
                       ht->slots[i].key, ht->slots[i].value);
                break;
        }
    }
}

int main(void) {
    HashTable *ht = ht_create();

    printf("=== 선형 탐사 해시 테이블 ===\n");

    ht_set(ht, "apple", "a fruit");
    ht_set(ht, "banana", "a tropical fruit");
    ht_set(ht, "cherry", "a small red fruit");

    ht_print(ht);

    printf("\n검색: apple = %s\n", ht_get(ht, "apple") ?: "(찾지 못함)");

    printf("\n삭제: banana\n");
    ht_delete(ht, "banana");
    ht_print(ht);

    ht_destroy(ht);
    return 0;
}
```

---

## 단계 4: 실습 - 간단한 사전 프로그램

```c
// dictionary.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define TABLE_SIZE 1000
#define KEY_SIZE 100
#define VALUE_SIZE 500

typedef struct Node {
    char word[KEY_SIZE];
    char meaning[VALUE_SIZE];
    struct Node *next;
} Node;

typedef struct {
    Node *buckets[TABLE_SIZE];
    int count;
} Dictionary;

unsigned int hash(const char *key) {
    unsigned int hash = 5381;
    while (*key) {
        hash = ((hash << 5) + hash) + tolower(*key++);
    }
    return hash % TABLE_SIZE;
}

Dictionary* dict_create(void) {
    Dictionary *dict = calloc(1, sizeof(Dictionary));
    return dict;
}

void dict_destroy(Dictionary *dict) {
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

void dict_add(Dictionary *dict, const char *word, const char *meaning) {
    unsigned int index = hash(word);

    // 기존 단어 확인
    Node *current = dict->buckets[index];
    while (current) {
        if (strcasecmp(current->word, word) == 0) {
            strncpy(current->meaning, meaning, VALUE_SIZE - 1);
            printf("'%s' 갱신됨\n", word);
            return;
        }
        current = current->next;
    }

    // 새 단어 추가
    Node *node = malloc(sizeof(Node));
    strncpy(node->word, word, KEY_SIZE - 1);
    strncpy(node->meaning, meaning, VALUE_SIZE - 1);
    node->next = dict->buckets[index];
    dict->buckets[index] = node;
    dict->count++;
    printf("'%s' 추가됨\n", word);
}

char* dict_search(Dictionary *dict, const char *word) {
    unsigned int index = hash(word);

    Node *current = dict->buckets[index];
    while (current) {
        if (strcasecmp(current->word, word) == 0) {
            return current->meaning;
        }
        current = current->next;
    }
    return NULL;
}

void dict_delete(Dictionary *dict, const char *word) {
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
            printf("'%s' 삭제됨\n", word);
            return;
        }
        prev = current;
        current = current->next;
    }
    printf("'%s' 찾지 못함\n", word);
}

void dict_list(Dictionary *dict) {
    printf("\n=== 사전 목록 (총: %d) ===\n", dict->count);
    for (int i = 0; i < TABLE_SIZE; i++) {
        Node *current = dict->buckets[i];
        while (current) {
            printf("  %s: %s\n", current->word, current->meaning);
            current = current->next;
        }
    }
}

void print_menu(void) {
    printf("\n============================\n");
    printf("|      간단한 사전         |\n");
    printf("|==========================|\n");
    printf("|  1. 단어 추가            |\n");
    printf("|  2. 단어 검색            |\n");
    printf("|  3. 단어 삭제            |\n");
    printf("|  4. 전체 보기            |\n");
    printf("|  0. 종료                 |\n");
    printf("============================\n");
}

void clear_input(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}

int main(void) {
    Dictionary *dict = dict_create();
    int choice;
    char word[KEY_SIZE];
    char meaning[VALUE_SIZE];

    // 샘플 데이터
    dict_add(dict, "apple", "a fruit; round and sweet");
    dict_add(dict, "book", "printed pages bound together");
    dict_add(dict, "computer", "electronic computing device");

    while (1) {
        print_menu();
        printf("선택: ");
        scanf("%d", &choice);
        clear_input();

        switch (choice) {
            case 1:
                printf("단어: ");
                fgets(word, KEY_SIZE, stdin);
                word[strcspn(word, "\n")] = '\0';

                printf("의미: ");
                fgets(meaning, VALUE_SIZE, stdin);
                meaning[strcspn(meaning, "\n")] = '\0';

                dict_add(dict, word, meaning);
                break;

            case 2:
                printf("검색할 단어: ");
                fgets(word, KEY_SIZE, stdin);
                word[strcspn(word, "\n")] = '\0';

                char *result = dict_search(dict, word);
                if (result) {
                    printf("\n  %s: %s\n", word, result);
                } else {
                    printf("\n  '%s' 찾지 못함\n", word);
                }
                break;

            case 3:
                printf("삭제할 단어: ");
                fgets(word, KEY_SIZE, stdin);
                word[strcspn(word, "\n")] = '\0';

                dict_delete(dict, word);
                break;

            case 4:
                dict_list(dict);
                break;

            case 0:
                printf("사전을 종료합니다.\n");
                dict_destroy(dict);
                return 0;

            default:
                printf("잘못된 선택입니다.\n");
        }
    }

    return 0;
}
```

---

## 컴파일 및 실행

```bash
gcc -Wall -std=c11 hash_chaining.c -o hash_chaining
gcc -Wall -std=c11 dictionary.c -o dictionary
./dictionary
```

---

## 요약

| 개념 | 설명 |
|------|------|
| 해시 함수(Hash Function) | 키를 인덱스로 변환 |
| 충돌(Collision) | 다른 키가 같은 인덱스에 매핑 |
| 체이닝(Chaining) | 연결 리스트로 충돌 처리 |
| 개방 주소법(Open Addressing) | 빈 슬롯을 탐사하여 충돌 처리 |
| 적재율(Load Factor) | count / table_size (0.7 이하 권장) |

### 체이닝 vs 개방 주소법

| 비교 | 체이닝 | 개방 주소법 |
|------|--------|------------|
| 메모리 | 동적 할당 | 고정 크기 |
| 삭제 | 간단 | DELETED 마커 필요 |
| 캐시 | 불리 | 유리 |
| 적재율 | 1 초과 가능 | 1 미만이어야 함 |

---

## 연습문제

1. **리사이즈**: 적재율이 0.7을 초과하면 테이블 크기를 2배로 확장

2. **파일 저장**: 사전 데이터를 파일에 저장/불러오기

3. **이중 해싱(Double Hashing)**: 충돌 시 두 번째 해시 함수를 탐사 간격에 사용

4. **충돌 통계**: 버킷당 충돌 수와 최대 체인 길이를 출력하는 함수를 추가하세요

5. **제네릭 해시 테이블**: `void*`와 사용자 제공 해시/비교 함수를 사용하여 임의의 키와 값 타입을 받을 수 있도록 해시 테이블을 수정하세요

---

## 다음 단계

[프로젝트: 파일 암호화](./08_Project_File_Encryption.md) -> 비트 연산을 활용한 파일 암호화 도구를 만들어봅시다!
