# C 네트워크 프로그래밍

**이전**: [멀티스레딩](./11_Multithreading.md) | **다음**: [프로세스 간 통신과 시그널](./13_IPC_and_Signals.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. TCP 및 UDP 소켓을 생성하고 적절한 바이트 순서 변환으로 주소 구조체를 설정할 수 있다
2. 바인드, 리슨, 연결 수락, 데이터 교환을 수행하는 TCP 에코 서버를 구현할 수 있다
3. 원격 서버에 연결하고 send/recv 루프를 처리하는 TCP 클라이언트를 작성할 수 있다
4. 견고한 `send_all` 및 `recv_exact` 헬퍼 함수로 부분 읽기/쓰기를 처리할 수 있다
5. `sendto()`와 `recvfrom()`을 사용하여 UDP 송신자와 수신자를 구축할 수 있다
6. `select()` 또는 `poll()`을 사용하여 단일 스레드에서 여러 클라이언트 연결을 다중화할 수 있다
7. 길이 접두사 메시지 프레이밍 및 논블로킹 소켓과 같은 실용 패턴을 적용할 수 있다

---

네트워크 프로그래밍은 독립형 프로그램을 연결된 서비스로 변환합니다. 채팅 서버, REST API, 분산 시스템을 구축하든, Berkeley 소켓 API는 모든 상위 수준 네트워킹 라이브러리의 기반입니다. C에서 소켓을 배우면 데이터가 실제로 네트워크를 통해 어떻게 이동하는지에 대한 명확한 시각을 얻게 되며, 이 지식은 나중에 어떤 언어나 프레임워크를 사용하든 도움이 됩니다.

**난이도**: 고급

---

## 목차

1. [소켓 기초](#1-소켓-기초)
2. [TCP 통신](#2-tcp-통신)
3. [UDP 통신](#3-udp-통신)
4. [I/O 다중화](#4-io-다중화)
5. [실용 패턴](#5-실용-패턴)
6. [연습 문제](#6-연습-문제)
7. [참고 자료](#7-참고-자료)

---

## 1. 소켓 기초

### 1.1 소켓이란?

소켓은 네트워크 통신의 끝점(endpoint)입니다. IP 주소와 포트 번호를 결합하여 특정 머신의 특정 프로세스를 식별합니다.

> **소켓 통신**
>
> | 클라이언트 머신 | | 서버 머신 |
> |---|---|---|
> | 응용 프로세스 | 네트워크 | 응용 프로세스 |
> | Socket fd=3 | <------> | Socket fd=4 |
> | 192.168.1.10:54321 | | 192.168.1.20:8080 |

### 1.2 소켓 API 개요

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// Key functions:
// socket()   - Create a socket
// bind()     - Bind socket to address
// listen()   - Mark socket as passive (server)
// accept()   - Accept incoming connection
// connect()  - Initiate connection (client)
// send/recv  - Data transfer (TCP)
// sendto/recvfrom - Data transfer (UDP)
// close()    - Close socket
```

### 1.3 주소 구조체

```c
// IPv4 address structure
struct sockaddr_in {
    sa_family_t    sin_family;   // AF_INET
    in_port_t      sin_port;     // Port (network byte order)
    struct in_addr sin_addr;     // IPv4 address
};

// Generic address structure (used in API)
struct sockaddr {
    sa_family_t sa_family;
    char        sa_data[14];
};
```

### 1.4 바이트 순서 변환

네트워크 프로토콜은 빅 엔디언(네트워크 바이트 순서)을 사용하지만, 대부분의 최신 CPU는 리틀 엔디언을 사용합니다.

```c
#include <arpa/inet.h>

uint16_t port = 8080;

// Host to Network
uint16_t net_port = htons(port);     // host to network short
uint32_t net_addr = htonl(INADDR_ANY); // host to network long

// Network to Host
uint16_t host_port = ntohs(net_port);  // network to host short
uint32_t host_addr = ntohl(net_addr);  // network to host long

// Address conversion
const char *ip_str = "192.168.1.10";
struct in_addr addr;
inet_pton(AF_INET, ip_str, &addr);  // String -> binary

char buf[INET_ADDRSTRLEN];
inet_ntop(AF_INET, &addr, buf, sizeof(buf));  // Binary -> string
printf("Address: %s\n", buf);  // "192.168.1.10"
```

---

## 2. TCP 통신

### 2.1 TCP 클라이언트-서버 흐름

> **TCP 클라이언트-서버 흐름**
>
> | 서버 | | 클라이언트 |
> |---|---|---|
> | `socket()` | | `socket()` |
> | `bind()` | | |
> | `listen()` | | |
> | `accept()` | <-- 3-way 핸드셰이크 --> | `connect()` |
> | `recv()` | <-- 데이터 --- | `send()` |
> | `send()` | --- 데이터 --> | `recv()` |
> | `close()` | <-- 4-way 종료 --> | `close()` |

### 2.2 TCP 에코 서버

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 8080
#define BUF_SIZE 1024

int main(void) {
    int server_fd, client_fd;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);
    char buffer[BUF_SIZE];

    // 1. Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    // Allow address reuse (avoid "Address already in use")
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // 2. Bind to address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;  // All interfaces
    server_addr.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&server_addr,
             sizeof(server_addr)) < 0) {
        perror("bind");
        close(server_fd);
        exit(EXIT_FAILURE);
    }

    // 3. Listen for connections
    if (listen(server_fd, 5) < 0) {
        perror("listen");
        close(server_fd);
        exit(EXIT_FAILURE);
    }
    printf("Server listening on port %d...\n", PORT);

    // 4. Accept and handle clients
    while (1) {
        client_fd = accept(server_fd, (struct sockaddr *)&client_addr,
                           &client_len);
        if (client_fd < 0) {
            perror("accept");
            continue;
        }

        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip,
                  sizeof(client_ip));
        printf("Client connected: %s:%d\n", client_ip,
               ntohs(client_addr.sin_port));

        // Echo loop
        ssize_t bytes;
        while ((bytes = recv(client_fd, buffer, BUF_SIZE - 1, 0)) > 0) {
            buffer[bytes] = '\0';
            printf("Received: %s", buffer);
            send(client_fd, buffer, bytes, 0);
        }

        printf("Client disconnected\n");
        close(client_fd);
    }

    close(server_fd);
    return 0;
}
```

### 2.3 TCP 에코 클라이언트

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 8080
#define BUF_SIZE 1024

int main(int argc, char *argv[]) {
    const char *server_ip = (argc > 1) ? argv[1] : "127.0.0.1";

    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        fprintf(stderr, "Invalid address: %s\n", server_ip);
        close(sock_fd);
        exit(EXIT_FAILURE);
    }

    if (connect(sock_fd, (struct sockaddr *)&server_addr,
                sizeof(server_addr)) < 0) {
        perror("connect");
        close(sock_fd);
        exit(EXIT_FAILURE);
    }
    printf("Connected to %s:%d\n", server_ip, PORT);

    char buffer[BUF_SIZE];
    while (fgets(buffer, BUF_SIZE, stdin) != NULL) {
        send(sock_fd, buffer, strlen(buffer), 0);

        ssize_t bytes = recv(sock_fd, buffer, BUF_SIZE - 1, 0);
        if (bytes <= 0) break;
        buffer[bytes] = '\0';
        printf("Echo: %s", buffer);
    }

    close(sock_fd);
    return 0;
}
```

### 2.4 부분 읽기/쓰기 처리

TCP는 스트림 프로토콜입니다. `send()`와 `recv()`는 요청한 것보다 적은 바이트를 전송할 수 있습니다.

```c
// Robust send: ensure all bytes are sent
ssize_t send_all(int fd, const void *buf, size_t len) {
    const char *p = buf;
    size_t remaining = len;

    while (remaining > 0) {
        ssize_t sent = send(fd, p, remaining, 0);
        if (sent < 0) return -1;
        if (sent == 0) return len - remaining;
        p += sent;
        remaining -= sent;
    }
    return len;
}

// Robust recv: read exactly n bytes
ssize_t recv_exact(int fd, void *buf, size_t len) {
    char *p = buf;
    size_t remaining = len;

    while (remaining > 0) {
        ssize_t received = recv(fd, p, remaining, 0);
        if (received < 0) return -1;
        if (received == 0) return len - remaining;  // Connection closed
        p += received;
        remaining -= received;
    }
    return len;
}
```

---

## 3. UDP 통신

### 3.1 UDP vs TCP

| 특성 | TCP | UDP |
|------|-----|-----|
| 연결 | 연결 지향 | 비연결 |
| 신뢰성 | 보장 | 최선 노력 |
| 순서 | 보존 | 보장되지 않음 |
| 흐름 제어 | 있음 | 없음 |
| 오버헤드 | 높음 | 낮음 |
| 사용 사례 | HTTP, SSH, 파일 전송 | DNS, VoIP, 게임, 스트리밍 |

### 3.2 UDP 송신자와 수신자

```c
// --- UDP 수신자 (서버) ---
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 9090
#define BUF_SIZE 1024

int main(void) {
    int sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(PORT);

    if (bind(sock_fd, (struct sockaddr *)&server_addr,
             sizeof(server_addr)) < 0) {
        perror("bind");
        close(sock_fd);
        exit(EXIT_FAILURE);
    }
    printf("UDP receiver listening on port %d...\n", PORT);

    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    char buffer[BUF_SIZE];

    while (1) {
        ssize_t bytes = recvfrom(sock_fd, buffer, BUF_SIZE - 1, 0,
                                 (struct sockaddr *)&client_addr,
                                 &client_len);
        if (bytes < 0) {
            perror("recvfrom");
            continue;
        }
        buffer[bytes] = '\0';

        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip,
                  sizeof(client_ip));
        printf("[%s:%d] %s", client_ip,
               ntohs(client_addr.sin_port), buffer);

        // Echo back
        sendto(sock_fd, buffer, bytes, 0,
               (struct sockaddr *)&client_addr, client_len);
    }

    close(sock_fd);
    return 0;
}
```

```c
// --- UDP 송신자 (클라이언트) ---
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 9090
#define BUF_SIZE 1024

int main(int argc, char *argv[]) {
    const char *server_ip = (argc > 1) ? argv[1] : "127.0.0.1";

    int sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fd < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    inet_pton(AF_INET, server_ip, &server_addr.sin_addr);

    char buffer[BUF_SIZE];
    while (fgets(buffer, BUF_SIZE, stdin) != NULL) {
        sendto(sock_fd, buffer, strlen(buffer), 0,
               (struct sockaddr *)&server_addr, sizeof(server_addr));

        struct sockaddr_in from_addr;
        socklen_t from_len = sizeof(from_addr);
        ssize_t bytes = recvfrom(sock_fd, buffer, BUF_SIZE - 1, 0,
                                 (struct sockaddr *)&from_addr,
                                 &from_len);
        if (bytes > 0) {
            buffer[bytes] = '\0';
            printf("Echo: %s", buffer);
        }
    }

    close(sock_fd);
    return 0;
}
```

---

## 4. I/O 다중화

### 4.1 다중화가 필요한 이유

루프에서 `accept()`를 사용하는 단순 서버는 한 번에 하나의 클라이언트만 처리할 수 있습니다. I/O 다중화를 사용하면 단일 스레드에서 여러 파일 디스크립터를 모니터링할 수 있습니다.

### 4.2 select()

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 8080
#define MAX_CLIENTS 10
#define BUF_SIZE 1024

int main(void) {
    int server_fd, client_fds[MAX_CLIENTS];
    fd_set read_fds, active_fds;
    int max_fd;

    // Initialize client array
    for (int i = 0; i < MAX_CLIENTS; i++)
        client_fds[i] = -1;

    // Create and setup server socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = INADDR_ANY,
        .sin_port = htons(PORT)
    };
    bind(server_fd, (struct sockaddr *)&addr, sizeof(addr));
    listen(server_fd, 5);
    printf("Select server on port %d\n", PORT);

    FD_ZERO(&active_fds);
    FD_SET(server_fd, &active_fds);
    max_fd = server_fd;

    char buffer[BUF_SIZE];

    while (1) {
        read_fds = active_fds;  // select modifies the set

        int ready = select(max_fd + 1, &read_fds, NULL, NULL, NULL);
        if (ready < 0) {
            perror("select");
            break;
        }

        // Check server socket for new connections
        if (FD_ISSET(server_fd, &read_fds)) {
            struct sockaddr_in client_addr;
            socklen_t len = sizeof(client_addr);
            int new_fd = accept(server_fd,
                               (struct sockaddr *)&client_addr, &len);
            if (new_fd >= 0) {
                // Add to client list
                for (int i = 0; i < MAX_CLIENTS; i++) {
                    if (client_fds[i] == -1) {
                        client_fds[i] = new_fd;
                        FD_SET(new_fd, &active_fds);
                        if (new_fd > max_fd) max_fd = new_fd;
                        printf("New client connected (fd=%d)\n", new_fd);
                        break;
                    }
                }
            }
        }

        // Check client sockets for data
        for (int i = 0; i < MAX_CLIENTS; i++) {
            int fd = client_fds[i];
            if (fd == -1) continue;

            if (FD_ISSET(fd, &read_fds)) {
                ssize_t bytes = recv(fd, buffer, BUF_SIZE - 1, 0);
                if (bytes <= 0) {
                    // Client disconnected
                    printf("Client disconnected (fd=%d)\n", fd);
                    close(fd);
                    FD_CLR(fd, &active_fds);
                    client_fds[i] = -1;
                } else {
                    buffer[bytes] = '\0';
                    // Echo to all clients
                    for (int j = 0; j < MAX_CLIENTS; j++) {
                        if (client_fds[j] != -1) {
                            send(client_fds[j], buffer, bytes, 0);
                        }
                    }
                }
            }
        }
    }

    close(server_fd);
    return 0;
}
```

### 4.3 poll()

`poll()`은 `select()`의 `FD_SETSIZE` 제한을 제거하고 더 깔끔한 인터페이스를 제공합니다.

```c
#include <poll.h>

#define MAX_FDS 100

struct pollfd fds[MAX_FDS];
int nfds = 1;

// Setup server socket
fds[0].fd = server_fd;
fds[0].events = POLLIN;

while (1) {
    int ready = poll(fds, nfds, -1);  // -1 = block indefinitely
    if (ready < 0) {
        perror("poll");
        break;
    }

    // New connection?
    if (fds[0].revents & POLLIN) {
        int new_fd = accept(server_fd, NULL, NULL);
        if (new_fd >= 0 && nfds < MAX_FDS) {
            fds[nfds].fd = new_fd;
            fds[nfds].events = POLLIN;
            nfds++;
        }
    }

    // Check existing clients
    for (int i = 1; i < nfds; i++) {
        if (fds[i].revents & POLLIN) {
            char buf[1024];
            ssize_t n = recv(fds[i].fd, buf, sizeof(buf), 0);
            if (n <= 0) {
                close(fds[i].fd);
                fds[i] = fds[nfds - 1];  // Remove by swapping
                nfds--;
                i--;
            } else {
                send(fds[i].fd, buf, n, 0);  // Echo
            }
        }
    }
}
```

### 4.4 비교: select vs poll vs epoll

| 특성 | select | poll | epoll |
|------|--------|------|-------|
| 최대 FD 수 | FD_SETSIZE (보통 1024) | 무제한 | 무제한 |
| 복잡도 | O(n) | O(n) | O(1) 분할 상환 |
| 이식성 | POSIX | POSIX | Linux 전용 |
| 오버헤드 | 매 호출마다 fd_set 복사 | 매 호출마다 배열 복사 | 커널 관리 |
| 적합한 용도 | 적은 수의 fd | 중간 수의 fd | 수천 개의 fd |

---

## 5. 실용 패턴

### 5.1 길이 접두사를 사용한 메시지 프레이밍

TCP는 바이트 스트림입니다. 개별 메시지를 보내려면 길이 접두사를 사용합니다.

```c
#include <stdint.h>

// Send a length-prefixed message
int send_message(int fd, const char *msg, uint32_t len) {
    uint32_t net_len = htonl(len);
    if (send_all(fd, &net_len, sizeof(net_len)) < 0) return -1;
    if (send_all(fd, msg, len) < 0) return -1;
    return 0;
}

// Receive a length-prefixed message
int recv_message(int fd, char *buf, uint32_t buf_size, uint32_t *out_len) {
    uint32_t net_len;
    if (recv_exact(fd, &net_len, sizeof(net_len)) <= 0) return -1;

    uint32_t len = ntohl(net_len);
    if (len > buf_size - 1) return -1;  // Message too large

    if (recv_exact(fd, buf, len) <= 0) return -1;
    buf[len] = '\0';
    *out_len = len;
    return 0;
}
```

### 5.2 논블로킹 소켓

```c
#include <fcntl.h>
#include <errno.h>

// Set socket to non-blocking mode
int set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

// Non-blocking recv check
ssize_t bytes = recv(fd, buffer, sizeof(buffer), 0);
if (bytes < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // No data available right now - not an error
    } else {
        perror("recv");
    }
}
```

### 5.3 정상 종료

```c
// Graceful shutdown: signal that no more data will be sent
shutdown(client_fd, SHUT_WR);  // Close write direction

// Then drain remaining data from the other side
char drain[256];
while (recv(client_fd, drain, sizeof(drain), 0) > 0)
    ;

close(client_fd);
```

### 5.4 getaddrinfo()를 사용한 프로토콜 독립적 주소 정보 조회(getaddrinfo)

`inet_pton`은 숫자 IP 문자열만 처리합니다. `getaddrinfo()`는 호스트명을 해석하고 IPv4와 IPv6 모두에 투명하게 동작하여, 프로토콜별 분기 없이 미래 호환성 있는 코드를 만들 수 있습니다.

```c
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

int connect_to(const char *host, const char *port) {
    struct addrinfo hints = {
        .ai_family   = AF_UNSPEC,    // Accept IPv4 or IPv6
        .ai_socktype = SOCK_STREAM,
    };
    struct addrinfo *res;

    int err = getaddrinfo(host, port, &hints, &res);
    if (err != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(err));
        return -1;
    }

    int fd = -1;
    for (struct addrinfo *p = res; p != NULL; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }

    freeaddrinfo(res);
    return fd;  // -1 if all addresses failed
}
```

### 5.5 이벤트 폴링(epoll)을 사용한 확장 가능한 I/O (Linux)

`select`와 `poll`은 매 호출마다 모니터링하는 모든 파일 디스크립터를 검사합니다 -- 깨어날 때마다 O(n). `epoll`은 커널 측 관심 목록을 유지하고 준비된 이벤트만 반환하여, 고연결 서버에서 분할 상환 O(1) 성능을 제공합니다.

```c
#include <sys/epoll.h>

#define MAX_EVENTS 64

int epfd = epoll_create1(0);

// Register the server socket for read events
struct epoll_event ev = { .events = EPOLLIN, .data.fd = server_fd };
epoll_ctl(epfd, EPOLL_CTL_ADD, server_fd, &ev);

struct epoll_event events[MAX_EVENTS];

while (1) {
    int n = epoll_wait(epfd, events, MAX_EVENTS, -1);
    for (int i = 0; i < n; i++) {
        if (events[i].data.fd == server_fd) {
            // New connection
            int client_fd = accept(server_fd, NULL, NULL);
            struct epoll_event cev = { .events = EPOLLIN, .data.fd = client_fd };
            epoll_ctl(epfd, EPOLL_CTL_ADD, client_fd, &cev);
        } else {
            // Data from existing client
            int fd = events[i].data.fd;
            char buf[1024];
            ssize_t n = recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) {
                epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL);
                close(fd);
            } else {
                send(fd, buf, n, 0);  // Echo
            }
        }
    }
}
```

비블로킹 소켓과 함께 `EPOLLET` (엣지 트리거 모드)를 사용하면 더 높은 처리량을 얻을 수 있습니다; 레벨 트리거 (기본값)는 더 단순하고 대부분의 애플리케이션에서 안전합니다.

### 5.6 프로덕션 서버를 위한 필수 소켓 옵션

```c
int opt = 1;

// SO_REUSEADDR: TIME_WAIT 소켓이 존재하는 동안 바인딩 허용
// (서버 재시작 후 "Address already in use" 방지)
setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

// SO_REUSEPORT: 여러 소켓이 같은 포트에 바인딩 허용
// 공유 잠금 없이 멀티 프로세스/스레드 accept() 활성화
setsockopt(server_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
```

`SO_REUSEADDR`는 어떤 서버에서도 사실상 필수입니다. `SO_REUSEPORT` (Linux 3.9+)는 여러 워커 프로세스가 각각 독립적으로 `accept()`를 호출할 수 있게 하여, 병목 없이 코어 전반에 걸쳐 연결 부하를 분산합니다.

---

## 6. 연습 문제

### 연습 문제 1: 바이트 순서 변환 연습

바이트 순서 변환을 시연하는 독립 프로그램을 작성하세요:

1. `uint16_t` 포트 값 `8080`과 `"192.168.10.1"`에 대한 `uint32_t` IP 주소를 선언합니다.
2. `htons` / `htonl`로 네트워크 바이트 순서로 변환하고 변환 전후의 원시 16진수 값을 출력합니다 (`%x` 사용).
3. `ntohs` / `ntohl`로 다시 변환하고 원래 값을 복원하는지 확인합니다.
4. `inet_pton`을 사용하여 문자열 `"192.168.10.1"`을 이진 `struct in_addr`로 변환한 후, `inet_ntop`으로 다시 문자열로 변환하여 일치하는지 확인합니다.

리틀 엔디언 머신에서 바이트 순서 변환을 건너뛰면 왜 눈에 보이지 않는 버그가 발생하는지 주석 블록에 설명하세요.

### 연습 문제 2: TCP 시간 서버

연결하는 모든 클라이언트에게 현재 날짜와 시간을 보내는 최소 TCP 서버를 구축하세요:

1. `SO_REUSEADDR`를 설정한 포트 `7777`에 리스닝 소켓을 생성합니다.
2. accept 루프에서 `time()`과 `ctime()`을 호출하여 현재 시간을 문자열로 가져옵니다.
3. 부분 쓰기를 처리하기 위해 `send_all`(섹션 2.4의 견고한 헬퍼)을 사용하여 시간 문자열을 클라이언트에 전송합니다.
4. 전송 후 즉시 클라이언트 소켓을 닫습니다.
5. `nc 127.0.0.1 7777` 또는 연결하여 한 줄을 읽고 출력하는 간단한 클라이언트로 테스트합니다.

### 연습 문제 3: 패킷 손실 시뮬레이션을 포함한 UDP 핑퐁

섹션 3.2의 UDP 송신자/수신자를 확장하세요:

1. 수신자에서 20% 패킷 손실을 시뮬레이션합니다: 각 `recvfrom`에서 난수를 생성하고 20%의 확률로 패킷을 드롭(에코하지 않음)하며 stderr에 "DROPPED"를 출력합니다.
2. 송신자에서 타임아웃 기반 재시도 루프를 구현합니다: `sendto` 후 `setsockopt(SO_RCVTIMEO)`로 소켓에 500ms 수신 타임아웃을 설정하고, `recvfrom`이 `EAGAIN`/`EWOULDBLOCK`을 반환하면 포기하기 전 최대 5회까지 재전송합니다.
3. 두 프로그램을 실행하고 출력에서 재시도 동작을 관찰합니다.

### 연습 문제 4: poll() 기반 에코 서버 업그레이드

섹션 4.2의 `select()` 기반 서버에서 시작하여 I/O 다중화 레이어를 `poll()`을 사용하여 다시 작성하세요:

1. `fd_set` / `FD_SET` / `FD_ISSET`를 `struct pollfd` 배열로 대체합니다.
2. 클라이언트가 연결 해제되면 배열의 마지막 항목과 교체하고 `nfds`를 감소시켜 제거합니다 (섹션 4.3에서 보여준 기법과 동일).
3. `poll()`에 5초 유휴 타임아웃을 추가합니다: 5초 동안 활동이 없으면 "Server idle..."을 출력하고 계속 대기합니다.
4. `telnet`이나 `nc`로 여러 클라이언트가 연결, 메시지 전송, 연결 해제할 수 있으며 서로에게 영향을 주지 않는지 확인합니다.

### 연습 문제 5: 길이 접두사 메시지 프로토콜

섹션 5.1의 길이 접두사 프레이밍을 사용하여 완전한 요청-응답 프로토콜을 구현하세요:

1. 간단한 프로토콜을 정의합니다: 클라이언트가 명령(`"UPPER"`, `"LOWER"`, 또는 `"REVERSE"`)과 공백, 그리고 문자열을 포함하는 길이 접두사 메시지를 보냅니다 (예: `"UPPER hello world"`).
2. 서버가 `recv_message`로 길이 접두사 메시지를 읽고 명령과 인수를 파싱하여 문자열을 적절히 변환한 후 길이 접두사 응답으로 회신합니다.
3. 클라이언트가 응답을 수신하고 출력한 다음, 미리 정의된 목록에서 다음 명령을 보냅니다.
4. 양쪽 모두 부분 읽기와 쓰기를 올바르게 처리하기 위해 `send_all`과 `recv_exact`를 전체적으로 사용합니다.

---

## 7. 참고 자료

- W. Richard Stevens, *Unix Network Programming, Volume 1* (3rd ed.)
- Beej's Guide to Network Programming: https://beej.us/guide/bgnet/
- `man 2 socket`, `man 2 bind`, `man 2 select`, `man 2 poll`

---
