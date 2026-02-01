# 프로젝트 2: 숫자 맞추기 게임

## 학습 목표

이 프로젝트를 통해 배우는 내용:
- 난수 생성 (`rand`, `srand`, `time`)
- 반복문 (`while`, `do-while`)
- 조건문 활용
- 게임 로직 구현

---

## 게임 규칙

```
1. 컴퓨터가 1~100 사이의 숫자를 선택
2. 플레이어가 숫자를 추측
3. "UP!" 또는 "DOWN!" 힌트 제공
4. 정답을 맞출 때까지 반복
5. 시도 횟수 표시
```

---

## 1단계: 난수 생성 이해

### 핵심 문법: rand()와 srand()

```c
#include <stdio.h>
#include <stdlib.h>  // rand, srand
#include <time.h>    // time

int main(void) {
    // 시드 설정 (한 번만 호출)
    // time(NULL): 현재 시간 (초)을 시드로 사용
    srand(time(NULL));

    // 난수 생성
    printf("%d\n", rand());  // 0 ~ RAND_MAX 사이 난수

    // 범위 지정: 1 ~ 100
    int num = rand() % 100 + 1;
    printf("1~100 사이 난수: %d\n", num);

    // 범위 공식: rand() % (최대 - 최소 + 1) + 최소
    // 예: 50~100 → rand() % 51 + 50

    return 0;
}
```

### 시드(Seed)가 필요한 이유

```c
// srand 없이 실행하면 매번 같은 순서의 난수가 생성됨!
// srand(time(NULL))로 현재 시간을 시드로 → 매 실행마다 다른 난수
```

---

## 2단계: 기본 게임 구현

```c
// guess_v1.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    // 난수 초기화
    srand(time(NULL));

    // 1~100 사이 정답 생성
    int answer = rand() % 100 + 1;
    int guess;
    int attempts = 0;

    printf("=== 숫자 맞추기 게임 ===\n");
    printf("1부터 100 사이의 숫자를 맞춰보세요!\n\n");

    // 게임 루프
    while (1) {
        printf("추측: ");
        scanf("%d", &guess);
        attempts++;

        if (guess < answer) {
            printf("UP! (더 높은 숫자입니다)\n\n");
        } else if (guess > answer) {
            printf("DOWN! (더 낮은 숫자입니다)\n\n");
        } else {
            printf("\n정답입니다!\n");
            printf("%d번 만에 맞추셨습니다!\n", attempts);
            break;
        }
    }

    return 0;
}
```

### 실행 예시

```
=== 숫자 맞추기 게임 ===
1부터 100 사이의 숫자를 맞춰보세요!

추측: 50
UP! (더 높은 숫자입니다)

추측: 75
DOWN! (더 낮은 숫자입니다)

추측: 62
UP! (더 높은 숫자입니다)

추측: 68
정답입니다!
4번 만에 맞추셨습니다!
```

---

## 3단계: 기능 추가

### 추가 기능

1. 시도 횟수 제한
2. 입력 검증
3. 재시작 기능
4. 난이도 선택

```c
// guess_v2.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 함수 선언
void print_title(void);
int get_difficulty(void);
int play_game(int max_num, int max_attempts);
void clear_input_buffer(void);

int main(void) {
    char play_again;

    srand(time(NULL));
    print_title();

    do {
        int difficulty = get_difficulty();
        int max_num, max_attempts;

        // 난이도 설정
        switch (difficulty) {
            case 1:  // 쉬움
                max_num = 50;
                max_attempts = 10;
                break;
            case 2:  // 보통
                max_num = 100;
                max_attempts = 7;
                break;
            case 3:  // 어려움
                max_num = 200;
                max_attempts = 8;
                break;
            default:
                max_num = 100;
                max_attempts = 7;
        }

        // 게임 실행
        int result = play_game(max_num, max_attempts);

        if (result) {
            printf("\n축하합니다! 승리!\n");
        } else {
            printf("\n아쉽습니다. 다음에 다시 도전하세요!\n");
        }

        // 재시작 확인
        printf("\n다시 하시겠습니까? (y/n): ");
        scanf(" %c", &play_again);
        clear_input_buffer();
        printf("\n");

    } while (play_again == 'y' || play_again == 'Y');

    printf("게임을 종료합니다. 안녕히 가세요!\n");
    return 0;
}

void print_title(void) {
    printf("\n");
    printf("╔════════════════════════════════╗\n");
    printf("║     숫자 맞추기 게임 v2        ║\n");
    printf("╚════════════════════════════════╝\n");
    printf("\n");
}

int get_difficulty(void) {
    int choice;

    printf("난이도를 선택하세요:\n");
    printf("  1. 쉬움   (1~50,  10번 기회)\n");
    printf("  2. 보통   (1~100, 7번 기회)\n");
    printf("  3. 어려움 (1~200, 8번 기회)\n");
    printf("\n선택: ");
    scanf("%d", &choice);
    clear_input_buffer();

    if (choice < 1 || choice > 3) {
        printf("잘못된 선택. 보통 난이도로 시작합니다.\n");
        choice = 2;
    }

    return choice;
}

int play_game(int max_num, int max_attempts) {
    int answer = rand() % max_num + 1;
    int guess;
    int attempts = 0;

    printf("\n1부터 %d 사이의 숫자를 맞춰보세요!\n", max_num);
    printf("기회: %d번\n\n", max_attempts);

    while (attempts < max_attempts) {
        printf("[%d/%d] 추측: ", attempts + 1, max_attempts);

        if (scanf("%d", &guess) != 1) {
            printf("숫자를 입력해주세요.\n\n");
            clear_input_buffer();
            continue;
        }

        // 범위 검증
        if (guess < 1 || guess > max_num) {
            printf("1~%d 사이의 숫자를 입력해주세요.\n\n", max_num);
            continue;
        }

        attempts++;

        if (guess < answer) {
            printf("UP! ↑\n");
            // 추가 힌트
            if (answer - guess > max_num / 4) {
                printf("(많이 차이납니다)\n");
            }
            printf("\n");
        } else if (guess > answer) {
            printf("DOWN! ↓\n");
            if (guess - answer > max_num / 4) {
                printf("(많이 차이납니다)\n");
            }
            printf("\n");
        } else {
            printf("\n🎉 정답입니다!\n");
            printf("%d번 만에 맞추셨습니다!\n", attempts);

            // 점수 계산
            int score = (max_attempts - attempts + 1) * 100;
            printf("점수: %d점\n", score);
            return 1;  // 승리
        }
    }

    printf("\n기회를 모두 사용했습니다.\n");
    printf("정답은 %d였습니다.\n", answer);
    return 0;  // 패배
}

void clear_input_buffer(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}
```

---

## 4단계: 최종 버전 (고급 기능)

### 추가 기능

- 최고 기록 저장 (세션 내)
- 통계 표시
- 더 나은 UI

```c
// guess_game.c (최종)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// 상수 정의
#define MAX_NAME_LEN 50

// 전역 변수 (게임 통계)
typedef struct {
    int games_played;
    int games_won;
    int best_score;
    int total_attempts;
    char best_player[MAX_NAME_LEN];
} GameStats;

// 함수 선언
void print_title(void);
void print_menu(void);
int get_difficulty(int *max_num, int *max_attempts);
int play_game(int max_num, int max_attempts, GameStats *stats);
void show_stats(GameStats *stats);
void clear_input_buffer(void);

int main(void) {
    int choice;
    GameStats stats = {0, 0, 0, 0, ""};

    srand(time(NULL));

    while (1) {
        print_title();
        print_menu();

        printf("선택: ");
        if (scanf("%d", &choice) != 1) {
            clear_input_buffer();
            continue;
        }
        clear_input_buffer();

        switch (choice) {
            case 1: {
                int max_num, max_attempts;
                get_difficulty(&max_num, &max_attempts);
                play_game(max_num, max_attempts, &stats);
                printf("\n계속하려면 Enter를 누르세요...");
                getchar();
                break;
            }
            case 2:
                show_stats(&stats);
                printf("\n계속하려면 Enter를 누르세요...");
                getchar();
                break;
            case 3:
                printf("\n게임을 종료합니다. 안녕히 가세요!\n\n");
                return 0;
            default:
                printf("\n잘못된 선택입니다.\n");
        }
    }

    return 0;
}

void print_title(void) {
    printf("\n");
    printf("  ╔═══════════════════════════════════╗\n");
    printf("  ║                                   ║\n");
    printf("  ║      🎯 숫자 맞추기 게임 🎯      ║\n");
    printf("  ║                                   ║\n");
    printf("  ╚═══════════════════════════════════╝\n");
    printf("\n");
}

void print_menu(void) {
    printf("  ┌─────────────────────────────────┐\n");
    printf("  │  1. 게임 시작                   │\n");
    printf("  │  2. 통계 보기                   │\n");
    printf("  │  3. 종료                        │\n");
    printf("  └─────────────────────────────────┘\n");
    printf("\n");
}

int get_difficulty(int *max_num, int *max_attempts) {
    int choice;

    printf("\n  난이도를 선택하세요:\n\n");
    printf("    1. 쉬움   │ 1~50   │ 10번 기회\n");
    printf("    2. 보통   │ 1~100  │ 7번 기회\n");
    printf("    3. 어려움 │ 1~200  │ 8번 기회\n");
    printf("    4. 극한   │ 1~1000 │ 10번 기회\n");
    printf("\n  선택: ");
    scanf("%d", &choice);
    clear_input_buffer();

    switch (choice) {
        case 1:
            *max_num = 50;
            *max_attempts = 10;
            break;
        case 2:
            *max_num = 100;
            *max_attempts = 7;
            break;
        case 3:
            *max_num = 200;
            *max_attempts = 8;
            break;
        case 4:
            *max_num = 1000;
            *max_attempts = 10;
            break;
        default:
            *max_num = 100;
            *max_attempts = 7;
    }

    return choice;
}

int play_game(int max_num, int max_attempts, GameStats *stats) {
    int answer = rand() % max_num + 1;
    int guess;
    int attempts = 0;
    int low = 1, high = max_num;  // 힌트용 범위

    printf("\n  ──────────────────────────────────\n");
    printf("  1부터 %d 사이의 숫자를 맞춰보세요!\n", max_num);
    printf("  기회: %d번\n", max_attempts);
    printf("  ──────────────────────────────────\n\n");

    while (attempts < max_attempts) {
        int remaining = max_attempts - attempts;
        printf("  [남은 기회: %d] 현재 범위: %d~%d\n", remaining, low, high);
        printf("  추측: ");

        if (scanf("%d", &guess) != 1) {
            printf("  → 숫자를 입력해주세요.\n\n");
            clear_input_buffer();
            continue;
        }

        if (guess < 1 || guess > max_num) {
            printf("  → 1~%d 사이의 숫자를 입력해주세요.\n\n", max_num);
            continue;
        }

        attempts++;
        stats->total_attempts++;

        if (guess < answer) {
            printf("  → UP! ↑ (더 큰 숫자입니다)\n\n");
            if (guess > low) low = guess + 1;
        } else if (guess > answer) {
            printf("  → DOWN! ↓ (더 작은 숫자입니다)\n\n");
            if (guess < high) high = guess - 1;
        } else {
            // 정답!
            int score = (max_attempts - attempts + 1) * 100 + (max_num / 10);

            printf("\n  ★★★ 정답입니다! ★★★\n\n");
            printf("  시도 횟수: %d번\n", attempts);
            printf("  점수: %d점\n", score);

            stats->games_played++;
            stats->games_won++;

            if (score > stats->best_score) {
                stats->best_score = score;
                printf("\n  🏆 새로운 최고 기록입니다!\n");
                printf("  이름을 입력하세요: ");
                scanf("%49s", stats->best_player);
                clear_input_buffer();
            }

            return 1;
        }
    }

    // 패배
    printf("\n  ✗ 기회를 모두 사용했습니다.\n");
    printf("  정답은 %d였습니다.\n", answer);

    stats->games_played++;
    return 0;
}

void show_stats(GameStats *stats) {
    printf("\n  ══════════ 게임 통계 ══════════\n\n");

    if (stats->games_played == 0) {
        printf("  아직 플레이한 게임이 없습니다.\n");
        return;
    }

    printf("  총 게임 수: %d\n", stats->games_played);
    printf("  승리: %d\n", stats->games_won);
    printf("  패배: %d\n", stats->games_played - stats->games_won);

    float win_rate = (float)stats->games_won / stats->games_played * 100;
    printf("  승률: %.1f%%\n", win_rate);

    float avg_attempts = (float)stats->total_attempts / stats->games_played;
    printf("  평균 시도 횟수: %.1f\n", avg_attempts);

    if (stats->best_score > 0) {
        printf("\n  🏆 최고 기록\n");
        printf("     점수: %d점\n", stats->best_score);
        printf("     플레이어: %s\n", stats->best_player);
    }

    printf("\n  ═════════════════════════════════\n");
}

void clear_input_buffer(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}
```

---

## 컴파일 및 실행

```bash
gcc -Wall -Wextra -std=c11 guess_game.c -o guess_game
./guess_game
```

---

## 배운 내용 정리

| 개념 | 설명 |
|------|------|
| `rand()` | 의사 난수 생성 |
| `srand(time(NULL))` | 시드 초기화 |
| `while (1)` | 무한 루프 |
| `break` | 루프 탈출 |
| `continue` | 다음 반복으로 |
| 구조체 | 관련 데이터 묶기 |

---

## 연습 문제

1. **이진 탐색 AI**: 컴퓨터가 플레이어의 숫자를 맞추는 모드 추가
   - 힌트: 항상 범위의 중간값 선택

2. **멀티플레이어**: 두 플레이어가 번갈아 추측하는 모드

3. **파일 저장**: 최고 기록을 파일로 저장하고 불러오기

---

## 다음 단계

[05_Project_Address_Book.md](./05_Project_Address_Book.md) → 구조체와 파일 I/O를 배워봅시다!
