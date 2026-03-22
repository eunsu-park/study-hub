# 프로젝트: 스네이크 게임

**이전**: [크로스 플랫폼 개발](./16_Cross_Platform_Development.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. ANSI 이스케이프 코드를 적용하여 화면 지우기, 커서 이동, 커서 숨김/표시, 컬러 텍스트 렌더링을 수행할 수 있다
2. `termios`를 사용하여 라인 버퍼링 없이 키 입력을 캡처하도록 터미널을 raw 모드로 설정할 수 있다
3. `VMIN`/`VTIME` 설정과 화살표 키를 위한 이스케이프 시퀀스 파싱을 사용하여 논블로킹 키보드 입력을 구현할 수 있다
4. `usleep`을 통한 프레임 레이트 제어와 함께 입력-업데이트-렌더링 사이클을 따르는 게임 루프를 설계할 수 있다
5. 매 프레임마다 머리가 자라고 꼬리가 줄어드는 연결 리스트를 사용하여 뱀 데이터 구조를 구축할 수 있다
6. 벽과 뱀 자체의 몸에 대한 충돌 감지를 구현할 수 있다
7. 원시 ANSI 기반 렌더링과 `ncurses` 라이브러리 접근 방식을 비교하고 트레이드오프를 식별할 수 있다

---

게임을 만드는 것은 지금까지 배운 거의 모든 것 -- 구조체, 연결 리스트, 동적 메모리, 비트 플래그, 터미널 I/O -- 을 실시간으로 사용자 입력에 응답해야 하는 하나의 프로그램으로 결합하게 합니다. 스네이크 게임은 규칙이 충분히 간단하여 시스템 프로그래밍 과제에 집중할 수 있기 때문에 완벽한 도구입니다: raw 터미널 제어, 논블로킹 입력, 일정한 프레임 레이트로 실행되어야 하는 촘촘한 업데이트-렌더 루프.

## 사전 지식
- 구조체와 포인터
- 동적 메모리 관리
- 연결 리스트 (뱀 몸체 표현용)

---

## 단계 1: ANSI 이스케이프 코드 이해

터미널에서 그래픽을 표시하기 위해 ANSI 이스케이프 코드를 사용합니다.

### 기본 ANSI 코드

```c
// ansi_demo.c
#include <stdio.h>
#include <unistd.h>

// ANSI Escape Codes
#define CLEAR_SCREEN "\033[2J"
#define CURSOR_HOME "\033[H"
#define HIDE_CURSOR "\033[?25l"
#define SHOW_CURSOR "\033[?25h"

// Cursor movement: \033[row;colH
#define MOVE_CURSOR(row, col) printf("\033[%d;%dH", row, col)

// Colors
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_CYAN "\033[36m"

int main(void) {
    printf(CLEAR_SCREEN CURSOR_HOME HIDE_CURSOR);

    MOVE_CURSOR(5, 10);
    printf(COLOR_RED "Red Text" COLOR_RESET);

    MOVE_CURSOR(7, 10);
    printf(COLOR_GREEN "Green Text" COLOR_RESET);

    MOVE_CURSOR(9, 10);
    printf(COLOR_BLUE "Blue Text" COLOR_RESET);

    sleep(3);

    printf(SHOW_CURSOR);
    MOVE_CURSOR(12, 1);

    return 0;
}
```

---

## 단계 2: 비동기 키보드 입력

게임에서는 키 입력을 기다리지 않고 실행이 계속되어야 합니다.

### termios를 사용한 입력 처리

```c
// input_demo.c
#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>

static struct termios original_termios;

void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO, &original_termios);

    struct termios raw = original_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;

    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

void disable_raw_mode(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &original_termios);
}

typedef enum {
    KEY_NONE = 0,
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT,
    KEY_QUIT, KEY_OTHER
} KeyCode;

KeyCode read_key(void) {
    int ch = getchar();
    if (ch == EOF) return KEY_NONE;
    if (ch == 'q' || ch == 'Q') return KEY_QUIT;

    if (ch == '\033') {
        int ch2 = getchar();
        if (ch2 == '[') {
            switch (getchar()) {
                case 'A': return KEY_UP;
                case 'B': return KEY_DOWN;
                case 'C': return KEY_RIGHT;
                case 'D': return KEY_LEFT;
            }
        }
    }

    switch (ch) {
        case 'w': case 'W': return KEY_UP;
        case 's': case 'S': return KEY_DOWN;
        case 'a': case 'A': return KEY_LEFT;
        case 'd': case 'D': return KEY_RIGHT;
    }

    return KEY_OTHER;
}
```

---

## 단계 3: 게임 데이터 구조

```c
// snake_types.h
#ifndef SNAKE_TYPES_H
#define SNAKE_TYPES_H

#include <stdbool.h>

#define SCREEN_WIDTH 40
#define SCREEN_HEIGHT 20
#define GAME_SPEED 150000

typedef enum { DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT } Direction;

typedef struct { int x, y; } Point;

typedef struct SnakeNode {
    Point pos;
    struct SnakeNode* next;
} SnakeNode;

typedef struct {
    SnakeNode* head;
    SnakeNode* tail;
    Direction dir;
    int length;
} Snake;

typedef struct {
    Snake snake;
    Point food;
    int score;
    bool game_over;
    bool paused;
} GameState;

#endif
```

---

## 단계 4: 완전한 스네이크 게임

```c
// snake_game.c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>

// ============ Config ============
#define WIDTH 40
#define HEIGHT 20
#define INITIAL_SPEED 150000

// ============ ANSI Codes ============
#define CLEAR "\033[2J"
#define HOME "\033[H"
#define HIDE_CURSOR "\033[?25l"
#define SHOW_CURSOR "\033[?25h"
#define MOVE(r,c) printf("\033[%d;%dH", r, c)

#define RESET "\033[0m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define RED "\033[31m"
#define CYAN "\033[36m"
#define BOLD "\033[1m"

// ============ Direction ============
typedef enum { UP, DOWN, LEFT, RIGHT } Direction;

typedef struct { int x, y; } Point;

typedef struct Node {
    Point pos;
    struct Node* next;
} Node;

// ============ Game State ============
typedef struct {
    Node* head;
    Node* tail;
    Direction dir;
    Point food;
    int score;
    int length;
    bool game_over;
    int speed;
} Game;

// ============ Terminal Setup ============
static struct termios orig_termios;

void disable_raw_mode(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
    printf(SHOW_CURSOR);
}

void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO, &orig_termios);
    atexit(disable_raw_mode);

    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;

    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    printf(HIDE_CURSOR);
}

// ============ Input ============
Direction read_direction(Direction current) {
    int ch = getchar();
    if (ch == EOF) return current;

    if (ch == '\033') {
        getchar();
        switch (getchar()) {
            case 'A': return (current != DOWN) ? UP : current;
            case 'B': return (current != UP) ? DOWN : current;
            case 'C': return (current != LEFT) ? RIGHT : current;
            case 'D': return (current != RIGHT) ? LEFT : current;
        }
    }

    switch (ch) {
        case 'w': case 'W': return (current != DOWN) ? UP : current;
        case 's': case 'S': return (current != UP) ? DOWN : current;
        case 'a': case 'A': return (current != RIGHT) ? LEFT : current;
        case 'd': case 'D': return (current != LEFT) ? RIGHT : current;
        case 'q': case 'Q': return -1;
    }

    return current;
}

// ============ Game Functions ============
bool snake_at(Node* head, int x, int y) {
    for (Node* n = head; n; n = n->next) {
        if (n->pos.x == x && n->pos.y == y) return true;
    }
    return false;
}

void spawn_food(Game* g) {
    do {
        g->food.x = 1 + rand() % (WIDTH - 2);
        g->food.y = 1 + rand() % (HEIGHT - 2);
    } while (snake_at(g->head, g->food.x, g->food.y));
}

Game* game_init(void) {
    Game* g = malloc(sizeof(Game));

    g->head = NULL;
    for (int i = 0; i < 3; i++) {
        Node* n = malloc(sizeof(Node));
        n->pos.x = WIDTH / 2 - i;
        n->pos.y = HEIGHT / 2;
        n->next = g->head;
        g->head = n;
        if (i == 0) g->tail = n;
    }

    Node* curr = g->head;
    while (curr->next) curr = curr->next;
    g->tail = curr;

    g->dir = RIGHT;
    g->score = 0;
    g->length = 3;
    g->game_over = false;
    g->speed = INITIAL_SPEED;

    spawn_food(g);
    return g;
}

void game_free(Game* g) {
    Node* n = g->head;
    while (n) {
        Node* next = n->next;
        free(n);
        n = next;
    }
    free(g);
}

bool game_update(Game* g) {
    Point next = g->head->pos;
    switch (g->dir) {
        case UP:    next.y--; break;
        case DOWN:  next.y++; break;
        case LEFT:  next.x--; break;
        case RIGHT: next.x++; break;
    }

    // Wall collision
    if (next.x <= 0 || next.x >= WIDTH - 1 ||
        next.y <= 0 || next.y >= HEIGHT - 1) {
        g->game_over = true;
        return false;
    }

    // Self collision
    if (snake_at(g->head, next.x, next.y)) {
        g->game_over = true;
        return false;
    }

    // Add new head
    Node* new_head = malloc(sizeof(Node));
    new_head->pos = next;
    new_head->next = g->head;
    g->head = new_head;

    // Check food
    if (next.x == g->food.x && next.y == g->food.y) {
        g->score += 10;
        g->length++;
        spawn_food(g);

        if (g->speed > 50000) {
            g->speed -= 5000;
        }
        return true;
    }

    // Remove tail
    Node* curr = g->head;
    while (curr->next && curr->next->next) {
        curr = curr->next;
    }
    free(curr->next);
    curr->next = NULL;
    g->tail = curr;

    return false;
}

// ============ Drawing ============
void draw_border(void) {
    MOVE(1, 1);
    printf(CYAN "+");
    for (int i = 1; i < WIDTH - 1; i++) printf("-");
    printf("+" RESET);

    for (int i = 2; i < HEIGHT; i++) {
        MOVE(i, 1);
        printf(CYAN "|" RESET);
        MOVE(i, WIDTH);
        printf(CYAN "|" RESET);
    }

    MOVE(HEIGHT, 1);
    printf(CYAN "+");
    for (int i = 1; i < WIDTH - 1; i++) printf("-");
    printf("+" RESET);
}

void draw_game(Game* g) {
    printf(CLEAR HOME);
    draw_border();

    MOVE(g->food.y + 1, g->food.x + 1);
    printf(RED "o" RESET);

    bool is_head = true;
    for (Node* n = g->head; n; n = n->next) {
        MOVE(n->pos.y + 1, n->pos.x + 1);
        if (is_head) {
            printf(BOLD GREEN "@" RESET);
            is_head = false;
        } else {
            printf(GREEN "#" RESET);
        }
    }

    MOVE(HEIGHT + 1, 1);
    printf(YELLOW "점수: %d  길이: %d" RESET, g->score, g->length);

    MOVE(HEIGHT + 2, 1);
    printf("조작: 화살표 키 또는 WASD, Q: 종료");

    fflush(stdout);
}

void draw_game_over(Game* g) {
    MOVE(HEIGHT / 2, WIDTH / 2 - 5);
    printf(BOLD RED "게임 오버!" RESET);

    MOVE(HEIGHT / 2 + 1, WIDTH / 2 - 6);
    printf("최종 점수: %d", g->score);

    MOVE(HEIGHT / 2 + 2, WIDTH / 2 - 8);
    printf("R: 재시작, Q: 종료");

    fflush(stdout);
}

// ============ Main ============
int main(void) {
    srand(time(NULL));
    enable_raw_mode();

    Game* game = game_init();
    draw_game(game);

    while (1) {
        Direction new_dir = read_direction(game->dir);
        if (new_dir == (Direction)-1) break;
        game->dir = new_dir;

        if (!game->game_over) {
            game_update(game);
            draw_game(game);

            if (game->game_over) {
                draw_game_over(game);
            }
        } else {
            int ch = getchar();
            if (ch == 'r' || ch == 'R') {
                game_free(game);
                game = game_init();
                draw_game(game);
            } else if (ch == 'q' || ch == 'Q') {
                break;
            }
        }

        usleep(game->speed);
    }

    game_free(game);

    MOVE(HEIGHT + 4, 1);
    printf("게임을 종료합니다.\n");

    return 0;
}
```

### 컴파일 및 실행

```bash
gcc -o snake snake_game.c -Wall -Wextra
./snake
```

---

## 단계 5: 기능 확장

### 벽 통과 모드

```c
// Replace wall collision with wrapping
if (next.x <= 0) next.x = WIDTH - 2;
else if (next.x >= WIDTH - 1) next.x = 1;

if (next.y <= 0) next.y = HEIGHT - 2;
else if (next.y >= HEIGHT - 1) next.y = 1;
```

### 장애물

```c
#define MAX_OBSTACLES 10

typedef struct {
    Point obstacles[MAX_OBSTACLES];
    int count;
} Obstacles;

void spawn_obstacles(Game* g, Obstacles* obs, int count) {
    obs->count = 0;
    for (int i = 0; i < count && obs->count < MAX_OBSTACLES; i++) {
        Point p;
        do {
            p.x = 2 + rand() % (WIDTH - 4);
            p.y = 2 + rand() % (HEIGHT - 4);
        } while (snake_at(g->head, p.x, p.y) ||
                 (p.x == g->food.x && p.y == g->food.y));
        obs->obstacles[obs->count++] = p;
    }
}
```

### 레벨 시스템

```c
typedef struct {
    int level;
    int food_to_next;
    int food_eaten;
} LevelSystem;

void level_init(LevelSystem* ls) {
    ls->level = 1;
    ls->food_to_next = 5;
    ls->food_eaten = 0;
}

bool level_eat_food(LevelSystem* ls) {
    ls->food_eaten++;
    if (ls->food_eaten >= ls->food_to_next) {
        ls->level++;
        ls->food_eaten = 0;
        ls->food_to_next += 2;
        return true;  // Level up!
    }
    return false;
}
```

### 최고 점수

```c
#define SCORE_FILE "snake_highscore.dat"

int load_highscore(void) {
    FILE* f = fopen(SCORE_FILE, "r");
    if (!f) return 0;
    int score;
    if (fscanf(f, "%d", &score) != 1) score = 0;
    fclose(f);
    return score;
}

void save_highscore(int score) {
    int current = load_highscore();
    if (score > current) {
        FILE* f = fopen(SCORE_FILE, "w");
        if (f) {
            fprintf(f, "%d", score);
            fclose(f);
        }
    }
}
```

---

## 단계 6: ncurses 버전 (선택 사항)

ncurses 라이브러리를 사용하면 내장된 색상 지원, 키 처리, 윈도우 관리로 더 깔끔한 코드를 작성할 수 있습니다.

```bash
# ncurses 설치
# macOS: brew install ncurses
# Ubuntu: sudo apt install libncurses5-dev

# 컴파일
gcc -o snake_ncurses snake_ncurses.c -lncurses
```

주요 ncurses 함수:
- `initscr()` / `endwin()` -- 초기화/정리
- `cbreak()` / `noecho()` -- Raw 입력
- `nodelay(stdscr, TRUE)` -- 논블로킹 입력
- `keypad(stdscr, TRUE)` -- 화살표 키 지원
- `mvaddch(y, x, ch)` -- 문자 그리기
- `mvprintw(y, x, fmt, ...)` -- 서식화된 텍스트 출력
- `attron(COLOR_PAIR(n))` -- 색상 설정

---

## 연습 문제

### 연습 문제 1: 일시 정지 기능
P 키를 누르면 일시 정지 기능을 구현하세요. 화면 중앙에 "일시 정지"를 표시하세요.

### 연습 문제 2: 특수 아이템
가끔 나타나는 특수 아이템을 추가하세요:
- 황금 사과: 30점
- 속도 감소: 일시적으로 속도를 줄임
- 투명화: 일시적으로 자기 몸을 통과할 수 있음

### 연습 문제 3: 2인 모드
각 플레이어가 WASD와 화살표 키를 사용하는 2인 모드를 구현하세요.

### 연습 문제 4: AI 뱀
자동으로 먹이를 찾는 AI 뱀을 추가하세요.
- 힌트: BFS 또는 간단한 휴리스틱을 사용하세요

### 연습 문제 5: 영구 리더보드
플레이어 이름과 함께 상위 10개 점수를 파일에 저장하세요. 게임 오버 시 리더보드를 표시하세요.

---

## 핵심 개념 요약

| 개념 | 설명 |
|------|------|
| ANSI 이스케이프 코드 | 터미널 화면 제어 (커서, 색상) |
| termios | 터미널 I/O 설정 |
| Raw 모드 | 버퍼링 없는 즉각적인 입력 |
| 게임 루프 | 입력 -> 업데이트 -> 렌더링 사이클 |
| 프레임 레이트 | usleep을 통한 속도 제어 |
| ncurses | 터미널 UI 라이브러리 |

---

## 다음 단계

C 고급 과정을 완료하신 것을 축하합니다! 이제 C를 사용한 시스템 프로그래밍에 깊은 전문성을 갖추게 되었습니다. [운영체제](../OS_Theory/00_Overview.md), [네트워킹](../Networking/00_Overview.md), [분산 시스템](../Distributed_Systems/00_Overview.md)과 같은 관련 토픽을 탐구해 보세요.
