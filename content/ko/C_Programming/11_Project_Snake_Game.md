# 프로젝트 10: 터미널 뱀 게임

**이전**: [프로젝트 8: 해시 테이블](./10_Project_Hash_Table.md) | **다음**: [프로젝트 11: 미니 쉘](./12_Project_Mini_Shell.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. ANSI 이스케이프 코드(ANSI escape codes)를 사용하여 화면 지우기, 커서 이동, 커서 숨기기/보이기, 색상 텍스트 렌더링을 수행
2. `termios`를 이용해 터미널을 로우 모드(raw mode)로 구성하여 라인 버퍼링 없이 키 입력 캡처
3. `VMIN`/`VTIME` 설정과 이스케이프 시퀀스(escape sequence) 파싱을 통한 방향키 비차단(non-blocking) 키보드 입력 구현
4. `usleep`으로 프레임 레이트를 제어하는 입력-업데이트-렌더 사이클의 게임 루프 설계
5. 매 프레임마다 머리가 성장하고 꼬리가 줄어드는 연결 리스트를 이용한 뱀 자료구조 구현
6. 벽 및 뱀 자신의 몸과의 충돌 감지 구현
7. 로우 ANSI 기반 렌더링과 `ncurses` 라이브러리 방식을 비교하고 장단점 파악

---

게임을 만들다 보면 구조체, 연결 리스트, 동적 메모리, 비트 플래그, 터미널 I/O 등 지금까지 배운 거의 모든 것을 하나의 프로그램으로 통합하게 됩니다. 뱀 게임은 이 목적에 딱 맞는 소재입니다. 규칙이 충분히 단순하여 시스템 프로그래밍의 핵심 도전 과제인 로우 터미널 제어, 비차단 입력, 일정한 프레임 레이트로 실행되어야 하는 타이트한 업데이트-렌더 루프에 집중할 수 있기 때문입니다.

## 사전 지식
- 구조체와 포인터
- 동적 메모리 관리
- 연결 리스트 (뱀 몸통 표현)

---

## 1단계: ANSI Escape Codes 이해

터미널에서 그래픽을 표현하기 위해 ANSI 이스케이프 코드(ANSI escape codes)를 사용합니다.

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

// 커서 이동: \033[row;colH
#define MOVE_CURSOR(row, col) printf("\033[%d;%dH", row, col)

// 색상
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN "\033[36m"

int main(void) {
    // 화면 지우기
    printf(CLEAR_SCREEN);
    printf(CURSOR_HOME);

    // 커서 숨기기
    printf(HIDE_CURSOR);

    // 여러 위치에 출력
    MOVE_CURSOR(5, 10);
    printf(COLOR_RED "Red Text" COLOR_RESET);

    MOVE_CURSOR(7, 10);
    printf(COLOR_GREEN "Green Text" COLOR_RESET);

    MOVE_CURSOR(9, 10);
    printf(COLOR_BLUE "Blue Text" COLOR_RESET);

    // 박스 그리기
    MOVE_CURSOR(12, 5);
    printf("┌────────────────────┐");
    for (int i = 13; i < 18; i++) {
        MOVE_CURSOR(i, 5);
        printf("│                    │");
    }
    MOVE_CURSOR(18, 5);
    printf("└────────────────────┘");

    MOVE_CURSOR(15, 10);
    printf(COLOR_YELLOW "Text in Box" COLOR_RESET);

    sleep(3);

    // 커서 보이기
    printf(SHOW_CURSOR);
    MOVE_CURSOR(20, 1);

    return 0;
}
```

---

## 2단계: 비동기 키보드 입력

게임에서는 키 입력을 기다리지 않고 계속 실행되어야 합니다.

### termios를 이용한 입력 처리

```c
// input_demo.c
#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

// 원래 터미널 설정 저장
static struct termios original_termios;

// 터미널을 raw 모드로 설정
void enable_raw_mode(void) {
    tcgetattr(STDIN_FILENO, &original_termios);

    struct termios raw = original_termios;

    // 입력 플래그: 에코 끄기, 라인 버퍼링 끄기
    raw.c_lflag &= ~(ECHO | ICANON);

    // 최소 입력 문자: 0 (non-blocking 가능)
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;  // 타임아웃 없음

    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
}

// 터미널 설정 복원
void disable_raw_mode(void) {
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &original_termios);
}

// 키 입력 확인 (non-blocking)
int kbhit(void) {
    int ch = getchar();
    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}

// 키 읽기
int getch(void) {
    return getchar();
}

// 방향키 읽기 (escape sequence 처리)
typedef enum {
    KEY_NONE = 0,
    KEY_UP,
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_QUIT,
    KEY_OTHER
} KeyCode;

KeyCode read_key(void) {
    int ch = getchar();

    if (ch == EOF) return KEY_NONE;
    if (ch == 'q' || ch == 'Q') return KEY_QUIT;

    // Escape sequence (방향키)
    if (ch == '\033') {
        int ch2 = getchar();
        if (ch2 == '[') {
            int ch3 = getchar();
            switch (ch3) {
                case 'A': return KEY_UP;
                case 'B': return KEY_DOWN;
                case 'C': return KEY_RIGHT;
                case 'D': return KEY_LEFT;
            }
        }
    }

    // WASD 키 지원
    switch (ch) {
        case 'w': case 'W': return KEY_UP;
        case 's': case 'S': return KEY_DOWN;
        case 'a': case 'A': return KEY_LEFT;
        case 'd': case 'D': return KEY_RIGHT;
    }

    return KEY_OTHER;
}

int main(void) {
    enable_raw_mode();
    atexit(disable_raw_mode);  // 프로그램 종료시 자동 복원

    printf("\033[2J\033[H");  // 화면 지우기
    printf("Move with arrow keys or WASD, Q to quit\n\n");

    int x = 10, y = 5;

    while (1) {
        KeyCode key = read_key();

        if (key == KEY_QUIT) break;

        // 이전 위치 지우기
        printf("\033[%d;%dH ", y, x);

        switch (key) {
            case KEY_UP:    if (y > 3) y--; break;
            case KEY_DOWN:  if (y < 20) y++; break;
            case KEY_LEFT:  if (x > 1) x--; break;
            case KEY_RIGHT: if (x < 40) x++; break;
            default: break;
        }

        // 새 위치에 출력
        printf("\033[%d;%dH@", y, x);
        fflush(stdout);

        usleep(50000);  // 50ms 대기
    }

    printf("\033[22;1HExiting.\n");
    return 0;
}
```

---

## 3단계: 기본 게임 구조

### 게임 데이터 구조 정의

```c
// snake_types.h
#ifndef SNAKE_TYPES_H
#define SNAKE_TYPES_H

#include <stdbool.h>

// 화면 크기
#define SCREEN_WIDTH 40
#define SCREEN_HEIGHT 20

// 게임 속도 (마이크로초)
#define GAME_SPEED 150000

// 방향
typedef enum {
    DIR_UP,
    DIR_DOWN,
    DIR_LEFT,
    DIR_RIGHT
} Direction;

// 좌표
typedef struct {
    int x;
    int y;
} Point;

// 뱀 몸통 노드
typedef struct SnakeNode {
    Point pos;
    struct SnakeNode* next;
} SnakeNode;

// 뱀
typedef struct {
    SnakeNode* head;
    SnakeNode* tail;
    Direction dir;
    int length;
} Snake;

// 게임 상태
typedef struct {
    Snake snake;
    Point food;
    int score;
    bool game_over;
    bool paused;
} GameState;

#endif
```

### 뱀 관리 함수

```c
// snake.c
#include <stdio.h>
#include <stdlib.h>
#include "snake_types.h"

// 뱀 생성
Snake* snake_create(int start_x, int start_y) {
    Snake* snake = malloc(sizeof(Snake));
    if (!snake) return NULL;

    // 초기 몸통 3칸
    snake->head = NULL;
    snake->tail = NULL;
    snake->length = 0;
    snake->dir = DIR_RIGHT;

    // 머리부터 꼬리 순서로 추가
    for (int i = 0; i < 3; i++) {
        SnakeNode* node = malloc(sizeof(SnakeNode));
        node->pos.x = start_x - i;
        node->pos.y = start_y;
        node->next = NULL;

        if (snake->head == NULL) {
            snake->head = node;
            snake->tail = node;
        } else {
            snake->tail->next = node;
            snake->tail = node;
        }
        snake->length++;
    }

    return snake;
}

// 뱀 해제
void snake_destroy(Snake* snake) {
    SnakeNode* current = snake->head;
    while (current) {
        SnakeNode* next = current->next;
        free(current);
        current = next;
    }
    free(snake);
}

// 방향 변경 (반대 방향 금지)
void snake_change_direction(Snake* snake, Direction new_dir) {
    // 현재 진행방향의 반대로는 못 감
    if ((snake->dir == DIR_UP && new_dir == DIR_DOWN) ||
        (snake->dir == DIR_DOWN && new_dir == DIR_UP) ||
        (snake->dir == DIR_LEFT && new_dir == DIR_RIGHT) ||
        (snake->dir == DIR_RIGHT && new_dir == DIR_LEFT)) {
        return;
    }
    snake->dir = new_dir;
}

// 다음 머리 위치 계산
Point snake_next_head(Snake* snake) {
    Point next = snake->head->pos;

    switch (snake->dir) {
        case DIR_UP:    next.y--; break;
        case DIR_DOWN:  next.y++; break;
        case DIR_LEFT:  next.x--; break;
        case DIR_RIGHT: next.x++; break;
    }

    return next;
}

// 뱀 이동 (음식 먹으면 true)
bool snake_move(Snake* snake, Point food) {
    Point next = snake_next_head(snake);

    // 새 머리 노드 생성
    SnakeNode* new_head = malloc(sizeof(SnakeNode));
    new_head->pos = next;
    new_head->next = snake->head;
    snake->head = new_head;
    snake->length++;

    // 음식을 먹었는지 확인
    if (next.x == food.x && next.y == food.y) {
        return true;  // 꼬리 유지 (길이 증가)
    }

    // 음식 안 먹었으면 꼬리 제거
    SnakeNode* current = snake->head;
    while (current->next != snake->tail) {
        current = current->next;
    }
    free(snake->tail);
    snake->tail = current;
    snake->tail->next = NULL;
    snake->length--;

    return false;
}

// 충돌 검사: 벽
bool snake_hit_wall(Snake* snake, int width, int height) {
    int x = snake->head->pos.x;
    int y = snake->head->pos.y;
    return (x < 1 || x >= width - 1 || y < 1 || y >= height - 1);
}

// 충돌 검사: 자기 몸
bool snake_hit_self(Snake* snake) {
    SnakeNode* head = snake->head;
    SnakeNode* current = head->next;

    while (current) {
        if (head->pos.x == current->pos.x &&
            head->pos.y == current->pos.y) {
            return true;
        }
        current = current->next;
    }
    return false;
}

// 특정 위치에 뱀이 있는지 확인
bool snake_occupies(Snake* snake, int x, int y) {
    SnakeNode* current = snake->head;
    while (current) {
        if (current->pos.x == x && current->pos.y == y) {
            return true;
        }
        current = current->next;
    }
    return false;
}
```

---

## 4단계: 완성된 뱀 게임

### 메인 게임 코드

```c
// snake_game.c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>

// ============ 설정 ============
#define WIDTH 40
#define HEIGHT 20
#define INITIAL_SPEED 150000  // 마이크로초

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

// ============ 방향 ============
typedef enum { UP, DOWN, LEFT, RIGHT } Direction;

// ============ 좌표 ============
typedef struct {
    int x, y;
} Point;

// ============ 뱀 노드 ============
typedef struct Node {
    Point pos;
    struct Node* next;
} Node;

// ============ 게임 상태 ============
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

// ============ 터미널 설정 ============
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

// ============ 입력 처리 ============
Direction read_direction(Direction current) {
    int ch = getchar();
    if (ch == EOF) return current;

    // ESC sequence (방향키)
    if (ch == '\033') {
        getchar();  // '['
        switch (getchar()) {
            case 'A': return (current != DOWN) ? UP : current;
            case 'B': return (current != UP) ? DOWN : current;
            case 'C': return (current != LEFT) ? RIGHT : current;
            case 'D': return (current != RIGHT) ? LEFT : current;
        }
    }

    // WASD
    switch (ch) {
        case 'w': case 'W': return (current != DOWN) ? UP : current;
        case 's': case 'S': return (current != UP) ? DOWN : current;
        case 'a': case 'A': return (current != RIGHT) ? LEFT : current;
        case 'd': case 'D': return (current != LEFT) ? RIGHT : current;
        case 'q': case 'Q': return -1;  // 종료 신호
    }

    return current;
}

// ============ 뱀 함수 ============
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

    // 뱀 초기화 (길이 3)
    g->head = NULL;
    for (int i = 0; i < 3; i++) {
        Node* n = malloc(sizeof(Node));
        n->pos.x = WIDTH / 2 - i;
        n->pos.y = HEIGHT / 2;
        n->next = g->head;
        g->head = n;
        if (i == 0) g->tail = n;
    }

    // 꼬리 찾기
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
    // 다음 머리 위치 계산
    Point next = g->head->pos;
    switch (g->dir) {
        case UP:    next.y--; break;
        case DOWN:  next.y++; break;
        case LEFT:  next.x--; break;
        case RIGHT: next.x++; break;
    }

    // 벽 충돌
    if (next.x <= 0 || next.x >= WIDTH - 1 ||
        next.y <= 0 || next.y >= HEIGHT - 1) {
        g->game_over = true;
        return false;
    }

    // 자기 몸 충돌
    if (snake_at(g->head, next.x, next.y)) {
        g->game_over = true;
        return false;
    }

    // 새 머리 추가
    Node* new_head = malloc(sizeof(Node));
    new_head->pos = next;
    new_head->next = g->head;
    g->head = new_head;

    // 음식 확인
    if (next.x == g->food.x && next.y == g->food.y) {
        g->score += 10;
        g->length++;
        spawn_food(g);

        // 속도 증가 (최소 50ms)
        if (g->speed > 50000) {
            g->speed -= 5000;
        }
        return true;
    }

    // 꼬리 제거
    Node* curr = g->head;
    while (curr->next && curr->next->next) {
        curr = curr->next;
    }
    free(curr->next);
    curr->next = NULL;
    g->tail = curr;

    return false;
}

// ============ 화면 그리기 ============
void draw_border(void) {
    // 상단
    MOVE(1, 1);
    printf(CYAN "╔");
    for (int i = 1; i < WIDTH - 1; i++) printf("═");
    printf("╗" RESET);

    // 측면
    for (int i = 2; i < HEIGHT; i++) {
        MOVE(i, 1);
        printf(CYAN "║" RESET);
        MOVE(i, WIDTH);
        printf(CYAN "║" RESET);
    }

    // 하단
    MOVE(HEIGHT, 1);
    printf(CYAN "╚");
    for (int i = 1; i < WIDTH - 1; i++) printf("═");
    printf("╝" RESET);
}

void draw_game(Game* g) {
    printf(CLEAR HOME);

    draw_border();

    // 음식
    MOVE(g->food.y + 1, g->food.x + 1);
    printf(RED "●" RESET);

    // 뱀
    bool is_head = true;
    for (Node* n = g->head; n; n = n->next) {
        MOVE(n->pos.y + 1, n->pos.x + 1);
        if (is_head) {
            printf(BOLD GREEN "◆" RESET);
            is_head = false;
        } else {
            printf(GREEN "■" RESET);
        }
    }

    // 점수
    MOVE(HEIGHT + 1, 1);
    printf(YELLOW "Score: %d  Length: %d" RESET, g->score, g->length);

    MOVE(HEIGHT + 2, 1);
    printf("Controls: ↑↓←→ or WASD, Q: Quit");

    fflush(stdout);
}

void draw_game_over(Game* g) {
    MOVE(HEIGHT / 2, WIDTH / 2 - 5);
    printf(BOLD RED "GAME OVER!" RESET);

    MOVE(HEIGHT / 2 + 1, WIDTH / 2 - 6);
    printf("Final Score: %d", g->score);

    MOVE(HEIGHT / 2 + 2, WIDTH / 2 - 8);
    printf("R: Restart, Q: Quit");

    fflush(stdout);
}

// ============ 메인 ============
int main(void) {
    srand(time(NULL));
    enable_raw_mode();

    Game* game = game_init();
    draw_game(game);

    while (1) {
        // 입력 처리
        Direction new_dir = read_direction(game->dir);
        if (new_dir == (Direction)-1) break;  // Q 종료
        game->dir = new_dir;

        if (!game->game_over) {
            // 게임 업데이트
            game_update(game);
            draw_game(game);

            if (game->game_over) {
                draw_game_over(game);
            }
        } else {
            // 게임 오버 상태에서 R 누르면 재시작
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
    printf("Exiting game.\n");

    return 0;
}
```

### 컴파일 및 실행

```bash
gcc -o snake snake_game.c
./snake
```

---

## 5단계: 기능 확장

### 벽 통과 모드 추가

```c
// 설정에 추가
#define WALL_WRAP true  // true면 반대편으로 나옴

// game_update 함수의 벽 충돌 부분 수정
bool game_update(Game* g) {
    Point next = g->head->pos;
    switch (g->dir) {
        case UP:    next.y--; break;
        case DOWN:  next.y++; break;
        case LEFT:  next.x--; break;
        case RIGHT: next.x++; break;
    }

#if WALL_WRAP
    // 벽 통과: 반대편으로 나옴
    if (next.x <= 0) next.x = WIDTH - 2;
    else if (next.x >= WIDTH - 1) next.x = 1;

    if (next.y <= 0) next.y = HEIGHT - 2;
    else if (next.y >= HEIGHT - 1) next.y = 1;
#else
    // 벽 충돌: 게임 오버
    if (next.x <= 0 || next.x >= WIDTH - 1 ||
        next.y <= 0 || next.y >= HEIGHT - 1) {
        g->game_over = true;
        return false;
    }
#endif

    // ... 나머지 코드
}
```

### 장애물 추가

```c
// 장애물 구조체
#define MAX_OBSTACLES 10

typedef struct {
    Point obstacles[MAX_OBSTACLES];
    int count;
} Obstacles;

// 장애물 생성
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

// 장애물 충돌 검사
bool hit_obstacle(Obstacles* obs, int x, int y) {
    for (int i = 0; i < obs->count; i++) {
        if (obs->obstacles[i].x == x && obs->obstacles[i].y == y) {
            return true;
        }
    }
    return false;
}

// 장애물 그리기
void draw_obstacles(Obstacles* obs) {
    for (int i = 0; i < obs->count; i++) {
        MOVE(obs->obstacles[i].y + 1, obs->obstacles[i].x + 1);
        printf("\033[35m█\033[0m");  // 마젠타 색상
    }
}
```

### 레벨 시스템

```c
typedef struct {
    int level;
    int food_to_next;  // 다음 레벨까지 필요한 음식
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
        ls->food_to_next += 2;  // 점점 더 많은 음식 필요
        return true;  // 레벨 업!
    }
    return false;
}

// 레벨에 따른 속도 계산
int get_speed_for_level(int level) {
    int base_speed = 150000;
    int speed = base_speed - (level - 1) * 15000;
    return (speed < 50000) ? 50000 : speed;
}
```

### 점수 저장 (최고 기록)

```c
#include <stdio.h>

#define SCORE_FILE "snake_highscore.dat"

int load_highscore(void) {
    FILE* f = fopen(SCORE_FILE, "r");
    if (!f) return 0;

    int score;
    if (fscanf(f, "%d", &score) != 1) {
        score = 0;
    }
    fclose(f);
    return score;
}

void save_highscore(int score) {
    int current_high = load_highscore();
    if (score > current_high) {
        FILE* f = fopen(SCORE_FILE, "w");
        if (f) {
            fprintf(f, "%d", score);
            fclose(f);
        }
    }
}

// 게임 종료시 호출
void game_end(Game* g) {
    int highscore = load_highscore();

    if (g->score > highscore) {
        MOVE(HEIGHT / 2 + 3, WIDTH / 2 - 8);
        printf("\033[33m★ New Record! ★\033[0m");
        save_highscore(g->score);
    } else {
        MOVE(HEIGHT / 2 + 3, WIDTH / 2 - 8);
        printf("High Score: %d", highscore);
    }
}
```

---

## 6단계: ncurses 버전 (선택사항)

ncurses 라이브러리를 사용하면 더 깔끔한 코드가 가능합니다.

### ncurses 설치

```bash
# macOS
brew install ncurses

# Ubuntu/Debian
sudo apt install libncurses5-dev

# Fedora
sudo dnf install ncurses-devel
```

### ncurses 버전 기본 구조

```c
// snake_ncurses.c
#include <ncurses.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define WIDTH 40
#define HEIGHT 20

// 방향
enum { UP, DOWN, LEFT, RIGHT };

// 노드
typedef struct Node {
    int x, y;
    struct Node* next;
} Node;

// 전역 상태
Node* head = NULL;
int dir = RIGHT;
int food_x, food_y;
int score = 0;
bool game_over = false;

void spawn_food(void) {
    do {
        food_x = 1 + rand() % (WIDTH - 2);
        food_y = 1 + rand() % (HEIGHT - 2);
    } while (/* 뱀 위치 체크 */ 0);
}

void init_game(void) {
    // ncurses 초기화
    initscr();
    cbreak();
    noecho();
    nodelay(stdscr, TRUE);  // non-blocking input
    keypad(stdscr, TRUE);   // 방향키 활성화
    curs_set(0);            // 커서 숨기기

    // 색상 초기화
    if (has_colors()) {
        start_color();
        init_pair(1, COLOR_GREEN, COLOR_BLACK);  // 뱀
        init_pair(2, COLOR_RED, COLOR_BLACK);    // 음식
        init_pair(3, COLOR_CYAN, COLOR_BLACK);   // 벽
    }

    // 뱀 초기화
    for (int i = 0; i < 3; i++) {
        Node* n = malloc(sizeof(Node));
        n->x = WIDTH / 2 - i;
        n->y = HEIGHT / 2;
        n->next = head;
        head = n;
    }

    srand(time(NULL));
    spawn_food();
}

void draw(void) {
    clear();

    // 벽
    attron(COLOR_PAIR(3));
    for (int i = 0; i < WIDTH; i++) {
        mvaddch(0, i, ACS_HLINE);
        mvaddch(HEIGHT - 1, i, ACS_HLINE);
    }
    for (int i = 0; i < HEIGHT; i++) {
        mvaddch(i, 0, ACS_VLINE);
        mvaddch(i, WIDTH - 1, ACS_VLINE);
    }
    mvaddch(0, 0, ACS_ULCORNER);
    mvaddch(0, WIDTH - 1, ACS_URCORNER);
    mvaddch(HEIGHT - 1, 0, ACS_LLCORNER);
    mvaddch(HEIGHT - 1, WIDTH - 1, ACS_LRCORNER);
    attroff(COLOR_PAIR(3));

    // 음식
    attron(COLOR_PAIR(2));
    mvaddch(food_y, food_x, 'O');
    attroff(COLOR_PAIR(2));

    // 뱀
    attron(COLOR_PAIR(1));
    for (Node* n = head; n; n = n->next) {
        mvaddch(n->y, n->x, n == head ? '@' : '#');
    }
    attroff(COLOR_PAIR(1));

    // 점수
    mvprintw(HEIGHT + 1, 0, "Score: %d", score);

    refresh();
}

void input(void) {
    int ch = getch();
    switch (ch) {
        case KEY_UP:    if (dir != DOWN) dir = UP; break;
        case KEY_DOWN:  if (dir != UP) dir = DOWN; break;
        case KEY_LEFT:  if (dir != RIGHT) dir = LEFT; break;
        case KEY_RIGHT: if (dir != LEFT) dir = RIGHT; break;
        case 'q': game_over = true; break;
    }
}

void update(void) {
    // 다음 위치
    int nx = head->x, ny = head->y;
    switch (dir) {
        case UP:    ny--; break;
        case DOWN:  ny++; break;
        case LEFT:  nx--; break;
        case RIGHT: nx++; break;
    }

    // 벽 충돌
    if (nx <= 0 || nx >= WIDTH - 1 || ny <= 0 || ny >= HEIGHT - 1) {
        game_over = true;
        return;
    }

    // 새 머리
    Node* new_head = malloc(sizeof(Node));
    new_head->x = nx;
    new_head->y = ny;
    new_head->next = head;
    head = new_head;

    // 음식 확인
    if (nx == food_x && ny == food_y) {
        score += 10;
        spawn_food();
    } else {
        // 꼬리 제거
        Node* curr = head;
        while (curr->next && curr->next->next) curr = curr->next;
        free(curr->next);
        curr->next = NULL;
    }
}

void cleanup(void) {
    while (head) {
        Node* next = head->next;
        free(head);
        head = next;
    }
    endwin();
}

int main(void) {
    init_game();

    while (!game_over) {
        input();
        update();
        draw();
        usleep(100000);
    }

    // 게임 오버 메시지
    mvprintw(HEIGHT / 2, WIDTH / 2 - 5, "GAME OVER!");
    mvprintw(HEIGHT / 2 + 1, WIDTH / 2 - 6, "Score: %d", score);
    refresh();
    nodelay(stdscr, FALSE);
    getch();

    cleanup();
    return 0;
}
```

### 컴파일

```bash
# macOS
gcc -o snake_ncurses snake_ncurses.c -lncurses

# Linux
gcc -o snake_ncurses snake_ncurses.c -lncurses
```

---

## 연습 문제

### 연습 1: 일시정지 기능
P 키를 누르면 게임이 일시정지되도록 구현하세요.

### 연습 2: 스페셜 아이템
가끔 나타나는 특별 아이템을 추가하세요:
- 골든 사과: 30점
- 스피드 다운: 일시적으로 속도 감소
- 투명화: 잠시 자기 몸 통과 가능

### 연습 3: 2인 플레이
WASD와 방향키로 각각 조작하는 2인 모드를 구현하세요.

### 연습 4: AI 뱀
자동으로 음식을 찾아가는 AI 뱀을 추가하세요.
- 힌트: BFS 또는 간단한 휴리스틱(heuristic) 사용

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| ANSI Escape Codes | 터미널 화면 제어 (커서, 색상) |
| termios | 터미널 입출력 설정 |
| Raw 모드(Raw mode) | 버퍼링 없는 즉시 입력 |
| 게임 루프(Game loop) | 입력 → 업데이트 → 렌더링 반복 |
| 프레임 레이트(Frame rate) | usleep으로 속도 조절 |
| ncurses | 터미널 UI 라이브러리 |

---

## 다음 단계

뱀 게임을 완성했다면 다음 프로젝트로 넘어가세요:
- [프로젝트 11: 미니 쉘](12_Project_Mini_Shell.md) - 간단한 명령어 쉘 구현

---

**이전**: [프로젝트 8: 해시 테이블](./10_Project_Hash_Table.md) | **다음**: [프로젝트 11: 미니 쉘](./12_Project_Mini_Shell.md)
