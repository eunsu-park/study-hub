# 프로젝트 2: 숫자 맞추기 게임

**이전**: [프로젝트 1: 사칙연산 계산기](./03_Project_Calculator.md) | **다음**: [프로젝트 3: 주소록 프로그램](./05_Project_Address_Book.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `rand`, `srand`, `time`을 사용하여 설정 가능한 범위 내에서 의사 난수(pseudo-random number)를 생성할 수 있습니다
2. 승리, 패배, 종료 조건이 충족될 때까지 반복되는 게임 루프(game loop)를 `while`과 `do-while`로 구현할 수 있습니다
3. 조건문(conditional statement)을 적용하여 방향 힌트(directional hint)와 근접도 피드백(proximity feedback)을 제공할 수 있습니다
4. `switch-case`를 통해 숫자 범위와 시도 횟수 제한을 조정하는 난이도 시스템(difficulty system)을 설계할 수 있습니다
5. 범위를 벗어나거나 숫자가 아닌 입력을 프로그램 충돌 없이 거부하는 입력 검증(input validation)을 구현할 수 있습니다
6. 구조체(struct)를 사용하여 세션 통계를 추적하고, 승률(win rate)과 평균 시도 횟수 같은 파생 지표(derived metric)를 계산할 수 있습니다

---

게임은 루프와 조건 논리를 내면화하기에 탁월한 방법입니다. 내리는 모든 결정에 프로그램이 즉각 반응하는 것을 눈으로 확인할 수 있기 때문입니다. 이 프로젝트에서는 무작위 정답을 생성하고, 플레이어에게 힌트를 제공하며, 점수를 추적하고, 여러 난이도를 지원하는 숫자 맞추기 게임을 만들면서 앞선 레슨에서 배운 C 언어 기초를 다집니다.

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

    printf("=== Number Guessing Game ===\n");
    printf("Guess a number between 1 and 100!\n\n");

    // 게임 루프
    while (1) {
        printf("Guess: ");
        scanf("%d", &guess);
        attempts++;

        if (guess < answer) {
            printf("UP! (The number is higher)\n\n");
        } else if (guess > answer) {
            printf("DOWN! (The number is lower)\n\n");
        } else {
            printf("\nCorrect!\n");
            printf("You got it in %d attempts!\n", attempts);
            break;
        }
    }

    return 0;
}
```

### 실행 예시

```
=== Number Guessing Game ===
Guess a number between 1 and 100!

Guess: 50
UP! (The number is higher)

Guess: 75
DOWN! (The number is lower)

Guess: 62
UP! (The number is higher)

Guess: 68
Correct!
You got it in 4 attempts!
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
    printf("================================\n");
    printf("     Number Guessing Game v2    \n");
    printf("================================\n");
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
            printf("UP!\n");
            // 추가 힌트
            if (answer - guess > max_num / 4) {
                printf("(많이 차이납니다)\n");
            }
            printf("\n");
        } else if (guess > answer) {
            printf("DOWN!\n");
            if (guess - answer > max_num / 4) {
                printf("(많이 차이납니다)\n");
            }
            printf("\n");
        } else {
            printf("\nCorrect!\n");
            printf("You got it in %d attempts!\n", attempts);

            // 점수 계산
            int score = (max_attempts - attempts + 1) * 100;
            printf("Score: %d points\n", score);
            return 1;  // 승리
        }
    }

    printf("\nOut of chances.\n");
    printf("The answer was %d.\n", answer);
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

        printf("Choice: ");
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
                printf("\nPress Enter to continue...");
                getchar();
                break;
            }
            case 2:
                show_stats(&stats);
                printf("\nPress Enter to continue...");
                getchar();
                break;
            case 3:
                printf("\nExiting game. Goodbye!\n\n");
                return 0;
            default:
                printf("\nInvalid choice.\n");
        }
    }

    return 0;
}

void print_title(void) {
    printf("\n");
    printf("  =====================================\n");
    printf("  |                                   |\n");
    printf("  |      Number Guessing Game         |\n");
    printf("  |                                   |\n");
    printf("  =====================================\n");
    printf("\n");
}

void print_menu(void) {
    printf("  ---------------------------------\n");
    printf("  |  1. Start Game                |\n");
    printf("  |  2. View Statistics           |\n");
    printf("  |  3. Exit                      |\n");
    printf("  ---------------------------------\n");
    printf("\n");
}

int get_difficulty(int *max_num, int *max_attempts) {
    int choice;

    printf("\n  난이도를 선택하세요:\n\n");
    printf("    1. 쉬움   | 1~50   | 10번 기회\n");
    printf("    2. 보통   | 1~100  | 7번 기회\n");
    printf("    3. 어려움 | 1~200  | 8번 기회\n");
    printf("    4. 극한   | 1~1000 | 10번 기회\n");
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

    printf("\n  ----------------------------------\n");
    printf("  1부터 %d 사이의 숫자를 맞춰보세요!\n", max_num);
    printf("  기회: %d번\n", max_attempts);
    printf("  ----------------------------------\n\n");

    while (attempts < max_attempts) {
        int remaining = max_attempts - attempts;
        printf("  [남은 기회: %d] 현재 범위: %d~%d\n", remaining, low, high);
        printf("  추측: ");

        if (scanf("%d", &guess) != 1) {
            printf("  -> 숫자를 입력해주세요.\n\n");
            clear_input_buffer();
            continue;
        }

        if (guess < 1 || guess > max_num) {
            printf("  -> 1~%d 사이의 숫자를 입력해주세요.\n\n", max_num);
            continue;
        }

        attempts++;
        stats->total_attempts++;

        if (guess < answer) {
            printf("  -> UP! (더 큰 숫자입니다)\n\n");
            if (guess > low) low = guess + 1;
        } else if (guess > answer) {
            printf("  -> DOWN! (더 작은 숫자입니다)\n\n");
            if (guess < high) high = guess - 1;
        } else {
            // 정답!
            int score = (max_attempts - attempts + 1) * 100 + (max_num / 10);

            printf("\n  *** Correct! ***\n\n");
            printf("  시도 횟수: %d번\n", attempts);
            printf("  점수: %d점\n", score);

            stats->games_played++;
            stats->games_won++;

            if (score > stats->best_score) {
                stats->best_score = score;
                printf("\n  새로운 최고 기록!\n");
                printf("  이름을 입력하세요: ");
                scanf("%49s", stats->best_player);
                clear_input_buffer();
            }

            return 1;
        }
    }

    // 패배
    printf("\n  X 기회를 모두 사용했습니다.\n");
    printf("  정답은 %d였습니다.\n", answer);

    stats->games_played++;
    return 0;
}

void show_stats(GameStats *stats) {
    printf("\n  ========== 게임 통계(Game Statistics) ==========\n\n");

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
        printf("\n  최고 기록\n");
        printf("     점수: %d점\n", stats->best_score);
        printf("     플레이어: %s\n", stats->best_player);
    }

    printf("\n  ================================================\n");
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
| `rand()` | 의사 난수(pseudo-random number) 생성 |
| `srand(time(NULL))` | 시드(seed) 초기화 |
| `while (1)` | 무한 루프(infinite loop) |
| `break` | 루프 탈출 |
| `continue` | 다음 반복으로 |
| 구조체(Struct) | 관련 데이터 묶기 |

---

## 연습 문제

1. **이진 탐색 AI**: 컴퓨터가 플레이어의 숫자를 맞추는 모드 추가
   - 힌트: 항상 범위의 중간값 선택

2. **멀티플레이어**: 두 플레이어가 번갈아 추측하는 모드

3. **파일 저장**: 최고 기록을 파일로 저장하고 불러오기

---

## 다음 단계

[프로젝트 3: 주소록 프로그램](./05_Project_Address_Book.md) → 구조체와 파일 I/O를 배워봅시다!

---

**이전**: [프로젝트 1: 사칙연산 계산기](./03_Project_Calculator.md) | **다음**: [프로젝트 3: 주소록 프로그램](./05_Project_Address_Book.md)
