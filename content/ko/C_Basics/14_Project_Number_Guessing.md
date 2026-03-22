# 프로젝트: 숫자 맞추기 게임

**이전**: [프로젝트: 계산기](./13_Project_Calculator.md) | **다음**: [프로젝트: 주소록](./15_Project_Address_Book.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `rand`, `srand`, `time`을 사용하여 구성 가능한 범위의 의사 난수 생성 구현하기
2. 승리, 패배, 또는 종료 조건까지 반복하는 `while`과 `do-while` 게임 루프 구축하기
3. 방향 힌트와 근접도 피드백을 제공하는 조건문 적용하기
4. `switch-case`를 통해 숫자 범위와 시도 제한을 조절하는 난이도 시스템 설계하기
5. 범위 밖이거나 숫자가 아닌 입력을 크래시 없이 거부하는 입력 유효성 검사 구현하기
6. 구조체를 사용하여 세션 통계를 추적하고 승률, 평균 시도 횟수 같은 파생 지표 계산하기

---

게임은 루프와 조건 로직을 내면화하는 데 훌륭한 방법입니다. 피드백이 즉각적이기 때문입니다 -- 프로그램이 매 결정에 반응하는 것을 볼 수 있습니다. 이 프로젝트에서는 난수 목표를 시드하고, 플레이어에게 힌트를 주고, 점수를 추적하며, 여러 난이도를 지원하는 숫자 맞추기 게임을 만들면서, 이전 레슨의 C 기초를 강화합니다.

## 게임 규칙

```
1. 컴퓨터가 1-100 사이의 숫자를 선택
2. 플레이어가 숫자를 추측
3. "UP!" 또는 "DOWN!" 힌트 제공
4. 정답을 맞출 때까지 반복
5. 시도 횟수 표시
```

---

## 단계 1: 난수 생성 이해하기

### 핵심 문법: rand()와 srand()

```c
#include <stdio.h>
#include <stdlib.h>  // rand, srand
#include <time.h>    // time

int main(void) {
    // Set seed (call only once)
    // time(NULL): use current time (seconds) as seed
    srand(time(NULL));

    // Generate random number
    printf("%d\n", rand());  // Random number between 0 ~ RAND_MAX

    // Specify range: 1 ~ 100
    int num = rand() % 100 + 1;
    printf("Random number between 1~100: %d\n", num);

    // Range formula: rand() % (max - min + 1) + min
    // Example: 50~100 -> rand() % 51 + 50

    return 0;
}
```

### 시드가 필요한 이유

```c
// Running without srand generates the same sequence of random numbers every time!
// srand(time(NULL)) uses current time as seed -> different random numbers each run
```

---

## 단계 2: 기본 게임 구현

```c
// guess_v1.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    // Initialize random
    srand(time(NULL));

    // Generate answer between 1~100
    int answer = rand() % 100 + 1;
    int guess;
    int attempts = 0;

    printf("=== Number Guessing Game ===\n");
    printf("Guess a number between 1 and 100!\n\n");

    // Game loop
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

### 출력 예시

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

## 단계 3: 기능 추가

### 추가 기능

1. 시도 횟수 제한
2. 입력 유효성 검사
3. 재시작 기능
4. 난이도 선택

```c
// guess_v2.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function declarations
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

        // Difficulty settings
        switch (difficulty) {
            case 1:  // Easy
                max_num = 50;
                max_attempts = 10;
                break;
            case 2:  // Normal
                max_num = 100;
                max_attempts = 7;
                break;
            case 3:  // Hard
                max_num = 200;
                max_attempts = 8;
                break;
            default:
                max_num = 100;
                max_attempts = 7;
        }

        // Run game
        int result = play_game(max_num, max_attempts);

        if (result) {
            printf("\nCongratulations! You win!\n");
        } else {
            printf("\nToo bad. Try again next time!\n");
        }

        // Restart confirmation
        printf("\nPlay again? (y/n): ");
        scanf(" %c", &play_again);
        clear_input_buffer();
        printf("\n");

    } while (play_again == 'y' || play_again == 'Y');

    printf("Exiting game. Goodbye!\n");
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

    printf("Select difficulty:\n");
    printf("  1. Easy   (1~50,  10 chances)\n");
    printf("  2. Normal (1~100, 7 chances)\n");
    printf("  3. Hard   (1~200, 8 chances)\n");
    printf("\nChoice: ");
    scanf("%d", &choice);
    clear_input_buffer();

    if (choice < 1 || choice > 3) {
        printf("Invalid choice. Starting with Normal difficulty.\n");
        choice = 2;
    }

    return choice;
}

int play_game(int max_num, int max_attempts) {
    int answer = rand() % max_num + 1;
    int guess;
    int attempts = 0;

    printf("\nGuess a number between 1 and %d!\n", max_num);
    printf("Chances: %d\n\n", max_attempts);

    while (attempts < max_attempts) {
        printf("[%d/%d] Guess: ", attempts + 1, max_attempts);

        if (scanf("%d", &guess) != 1) {
            printf("Please enter a number.\n\n");
            clear_input_buffer();
            continue;
        }

        // Range validation
        if (guess < 1 || guess > max_num) {
            printf("Please enter a number between 1~%d.\n\n", max_num);
            continue;
        }

        attempts++;

        if (guess < answer) {
            printf("UP!\n");
            // Additional hint
            if (answer - guess > max_num / 4) {
                printf("(Big difference)\n");
            }
            printf("\n");
        } else if (guess > answer) {
            printf("DOWN!\n");
            if (guess - answer > max_num / 4) {
                printf("(Big difference)\n");
            }
            printf("\n");
        } else {
            printf("\nCorrect!\n");
            printf("You got it in %d attempts!\n", attempts);

            // Calculate score
            int score = (max_attempts - attempts + 1) * 100;
            printf("Score: %d points\n", score);
            return 1;  // Win
        }
    }

    printf("\nOut of chances.\n");
    printf("The answer was %d.\n", answer);
    return 0;  // Lose
}

void clear_input_buffer(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}
```

---

## 단계 4: 최종 버전 (고급 기능)

### 추가 기능

- 최고 점수 저장 (세션 내)
- 통계 표시
- 개선된 UI

```c
// guess_game.c (Final)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Constant definitions
#define MAX_NAME_LEN 50

// Global variables (game statistics)
typedef struct {
    int games_played;
    int games_won;
    int best_score;
    int total_attempts;
    char best_player[MAX_NAME_LEN];
} GameStats;

// Function declarations
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

    printf("\n  Select difficulty:\n\n");
    printf("    1. Easy     | 1~50   | 10 chances\n");
    printf("    2. Normal   | 1~100  | 7 chances\n");
    printf("    3. Hard     | 1~200  | 8 chances\n");
    printf("    4. Extreme  | 1~1000 | 10 chances\n");
    printf("\n  Choice: ");
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
    int low = 1, high = max_num;  // Range for hints

    printf("\n  ----------------------------------\n");
    printf("  Guess a number between 1 and %d!\n", max_num);
    printf("  Chances: %d\n", max_attempts);
    printf("  ----------------------------------\n\n");

    while (attempts < max_attempts) {
        int remaining = max_attempts - attempts;
        printf("  [Remaining: %d] Current range: %d~%d\n", remaining, low, high);
        printf("  Guess: ");

        if (scanf("%d", &guess) != 1) {
            printf("  -> Please enter a number.\n\n");
            clear_input_buffer();
            continue;
        }

        if (guess < 1 || guess > max_num) {
            printf("  -> Please enter a number between 1~%d.\n\n", max_num);
            continue;
        }

        attempts++;
        stats->total_attempts++;

        if (guess < answer) {
            printf("  -> UP! (The number is higher)\n\n");
            if (guess > low) low = guess + 1;
        } else if (guess > answer) {
            printf("  -> DOWN! (The number is lower)\n\n");
            if (guess < high) high = guess - 1;
        } else {
            // Correct!
            int score = (max_attempts - attempts + 1) * 100 + (max_num / 10);

            printf("\n  *** Correct! ***\n\n");
            printf("  Attempts: %d\n", attempts);
            printf("  Score: %d points\n", score);

            stats->games_played++;
            stats->games_won++;

            if (score > stats->best_score) {
                stats->best_score = score;
                printf("\n  New high score!\n");
                printf("  Enter your name: ");
                scanf("%49s", stats->best_player);
                clear_input_buffer();
            }

            return 1;
        }
    }

    // Lose
    printf("\n  X Out of chances.\n");
    printf("  The answer was %d.\n", answer);

    stats->games_played++;
    return 0;
}

void show_stats(GameStats *stats) {
    printf("\n  ========== Game Statistics ==========\n\n");

    if (stats->games_played == 0) {
        printf("  No games played yet.\n");
        return;
    }

    printf("  Total games: %d\n", stats->games_played);
    printf("  Wins: %d\n", stats->games_won);
    printf("  Losses: %d\n", stats->games_played - stats->games_won);

    float win_rate = (float)stats->games_won / stats->games_played * 100;
    printf("  Win rate: %.1f%%\n", win_rate);

    float avg_attempts = (float)stats->total_attempts / stats->games_played;
    printf("  Average attempts: %.1f\n", avg_attempts);

    if (stats->best_score > 0) {
        printf("\n  High Score\n");
        printf("     Score: %d points\n", stats->best_score);
        printf("     Player: %s\n", stats->best_player);
    }

    printf("\n  =====================================\n");
}

void clear_input_buffer(void) {
    int c;
    while ((c = getchar()) != '\n' && c != EOF);
}
```

---

## 컴파일과 실행

```bash
gcc -Wall -Wextra -std=c11 guess_game.c -o guess_game
./guess_game
```

---

## 요약

| 개념 | 설명 |
|---------|-------------|
| `rand()` | 의사 난수 생성 |
| `srand(time(NULL))` | 시드 초기화 |
| `while (1)` | 무한 루프 |
| `break` | 루프 탈출 |
| `continue` | 다음 반복으로 건너뛰기 |
| 구조체 (Struct) | 관련 데이터 그룹화 |

---

## 연습문제

1. **이진 탐색 AI**: 컴퓨터가 플레이어의 숫자를 맞추는 모드 추가
   - 힌트: 항상 범위의 중간값 선택

2. **멀티플레이어**: 두 플레이어가 번갈아 추측하는 모드

3. **파일 저장**: 최고 점수를 파일에 저장하고 불러오기

4. **근접도 힌트**: "UP/DOWN" 대신 추측이 얼마나 가까운지에 따라 "매우 뜨겁다!", "따뜻하다", "차갑다", "얼어붙는다!" 같은 힌트 추가

5. **범위 설정**: 게임 시작 전에 플레이어가 직접 최소/최대 숫자를 설정할 수 있도록 하기

---

## 다음 단계

[프로젝트: 주소록](./15_Project_Address_Book.md) -- 구조체와 파일 입출력을 배우며 완전한 CRUD 애플리케이션을 만들어 보세요.
