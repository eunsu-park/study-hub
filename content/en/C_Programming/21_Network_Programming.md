# Network Programming in C

**Previous**: [Advanced C Pointers](./20_Advanced_Pointers.md) | **Next**: [Inter-Process Communication and Signals](./22_IPC_and_Signals.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Create TCP and UDP sockets and configure address structures with proper byte-order conversion
2. Implement a TCP echo server that binds, listens, accepts connections, and exchanges data
3. Write a TCP client that connects to a remote server and handles the send/recv loop
4. Handle partial reads and writes with robust `send_all` and `recv_exact` helper functions
5. Build a UDP sender and receiver using `sendto()` and `recvfrom()`
6. Multiplex multiple client connections in a single thread using `select()` or `poll()`
7. Apply practical patterns such as length-prefix message framing and non-blocking sockets

---

Network programming turns a standalone program into a connected service. Whether you are building a chat server, a REST API, or a distributed system, the Berkeley socket API is the foundation upon which all higher-level networking libraries are built. Learning sockets in C gives you an unobstructed view of how data actually moves across the wire -- knowledge that pays dividends no matter which language or framework you use later.

**Difficulty**: Advanced

---

## Table of Contents

1. [Socket Fundamentals](#1-socket-fundamentals)
2. [TCP Communication](#2-tcp-communication)
3. [UDP Communication](#3-udp-communication)
4. [I/O Multiplexing](#4-io-multiplexing)
5. [Practical Patterns](#5-practical-patterns)
6. [Practice Problems](#6-practice-problems)
7. [References](#7-references)

---

## 1. Socket Fundamentals

### 1.1 What is a Socket?

A socket is an endpoint for network communication. It combines an IP address and a port number to identify a specific process on a specific machine.

```
┌────────────────────────────────────────────────────────────┐
│                   Socket Communication                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  [Client Machine]              [Server Machine]            │
│  ┌──────────────┐              ┌──────────────┐            │
│  │  Application │              │  Application │            │
│  │    Process   │              │    Process   │            │
│  │  ┌────────┐  │              │  ┌────────┐  │            │
│  │  │ Socket │  │   Network    │  │ Socket │  │            │
│  │  │  fd=3  │◀─┼─────────────┼─▶│  fd=4  │  │            │
│  │  └────────┘  │              │  └────────┘  │            │
│  │ 192.168.1.10 │              │ 192.168.1.20 │            │
│  │   :54321     │              │   :8080      │            │
│  └──────────────┘              └──────────────┘            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 1.2 Socket API Overview

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

### 1.3 Address Structures

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

### 1.4 Byte Order Conversion

Network protocols use big-endian (network byte order), but most modern CPUs use little-endian.

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
inet_pton(AF_INET, ip_str, &addr);  // String → binary

char buf[INET_ADDRSTRLEN];
inet_ntop(AF_INET, &addr, buf, sizeof(buf));  // Binary → string
printf("Address: %s\n", buf);  // "192.168.1.10"
```

---

## 2. TCP Communication

### 2.1 TCP Client-Server Flow

```
┌──────────────────────────────────────────────────────────┐
│  Server                              Client              │
│  ──────                              ──────              │
│  socket()                            socket()            │
│     │                                   │                │
│  bind()                                 │                │
│     │                                   │                │
│  listen()                               │                │
│     │                                   │                │
│  accept() ◀── 3-way handshake ──── connect()            │
│     │                                   │                │
│  recv() ◀──────── data ────────── send()                │
│     │                                   │                │
│  send() ────────── data ──────────▶ recv()              │
│     │                                   │                │
│  close() ◀── 4-way teardown ───── close()              │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 2.2 TCP Echo Server

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

### 2.3 TCP Echo Client

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

### 2.4 Handling Partial Reads/Writes

TCP is a stream protocol. `send()` and `recv()` may transfer fewer bytes than requested.

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

## 3. UDP Communication

### 3.1 UDP vs TCP

```
┌─────────────────────────────────────────────────────────┐
│  Feature          │  TCP             │  UDP              │
├───────────────────┼──────────────────┼───────────────────┤
│  Connection       │  Connection-     │  Connectionless   │
│                   │  oriented        │                   │
│  Reliability      │  Guaranteed      │  Best-effort      │
│  Ordering         │  Preserved       │  Not guaranteed   │
│  Flow Control     │  Yes             │  No               │
│  Overhead         │  Higher          │  Lower            │
│  Use Cases        │  HTTP, SSH,      │  DNS, VoIP,       │
│                   │  file transfer   │  gaming, streaming│
└─────────────────────────────────────────────────────────┘
```

### 3.2 UDP Sender and Receiver

```c
// --- UDP Receiver (Server) ---
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
// --- UDP Sender (Client) ---
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

## 4. I/O Multiplexing

### 4.1 Why Multiplexing?

A simple server using `accept()` in a loop can only handle one client at a time. I/O multiplexing lets a single thread monitor multiple file descriptors.

```
┌────────────────────────────────────────────────────────────┐
│                   I/O Multiplexing                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────┐                                              │
│  │ Client 1 │──┐                                           │
│  └──────────┘  │     ┌──────────────┐    ┌──────────────┐ │
│  ┌──────────┐  ├────▶│ select/poll/ │───▶│   Server     │ │
│  │ Client 2 │──┤     │    epoll     │    │   Handler    │ │
│  └──────────┘  │     └──────────────┘    └──────────────┘ │
│  ┌──────────┐  │     "Which fd is ready?"                  │
│  │ Client 3 │──┘                                           │
│  └──────────┘                                              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

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

`poll()` removes the `FD_SETSIZE` limit of `select()` and provides a cleaner interface.

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

### 4.4 Comparison: select vs poll vs epoll

```
┌──────────────┬────────────────┬──────────────┬───────────────┐
│              │  select        │  poll        │  epoll        │
├──────────────┼────────────────┼──────────────┼───────────────┤
│ Max FDs      │ FD_SETSIZE     │ Unlimited    │ Unlimited     │
│              │ (usually 1024) │              │               │
│ Complexity   │ O(n)           │ O(n)         │ O(1) amortized│
│ Portability  │ POSIX          │ POSIX        │ Linux only    │
│ Overhead     │ Copy fd_set    │ Copy array   │ Kernel-managed│
│              │ each call      │ each call    │               │
│ Best for     │ Small # fds    │ Moderate fds │ Thousands fds │
└──────────────┴────────────────┴──────────────┴───────────────┘
```

---

## 5. Practical Patterns

### 5.1 Message Framing with Length Prefix

TCP is a byte stream. To send discrete messages, use a length prefix.

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

### 5.2 Non-blocking Socket

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

### 5.3 Graceful Shutdown

```c
// Graceful shutdown: signal that no more data will be sent
shutdown(client_fd, SHUT_WR);  // Close write direction

// Then drain remaining data from the other side
char drain[256];
while (recv(client_fd, drain, sizeof(drain), 0) > 0)
    ;

close(client_fd);
```

---

## 6. Practice Problems

### Problem 1: Multi-client Chat Server

Build a chat server where messages from one client are broadcast to all connected clients. Use `select()` or `poll()` for multiplexing.

**Requirements**:
- Support at least 10 simultaneous clients
- Display "[username] message" format
- Handle client disconnection gracefully

### Problem 2: File Transfer

Implement a simple file transfer protocol:
- Client sends a filename, server responds with the file contents
- Use length-prefix framing for messages
- Handle file-not-found errors

### Problem 3: HTTP Client

Write a minimal HTTP/1.1 client that:
- Connects to a web server on port 80
- Sends a GET request
- Parses and displays the response headers and body

---

## 7. References

- W. Richard Stevens, *Unix Network Programming, Volume 1* (3rd ed.)
- Beej's Guide to Network Programming: https://beej.us/guide/bgnet/
- `man 2 socket`, `man 2 bind`, `man 2 select`, `man 2 poll`

---

## Exercises

### Exercise 1: Byte-Order Conversion Practice

Write a standalone program that demonstrates byte-order conversion:

1. Declare a `uint16_t` port value of `8080` and a `uint32_t` IP address for `"192.168.10.1"`.
2. Convert each to network byte order with `htons` / `htonl` and print the raw hexadecimal value before and after conversion (use `%x`).
3. Convert back with `ntohs` / `ntohl` and verify you recover the original values.
4. Use `inet_pton` to convert the string `"192.168.10.1"` to a binary `struct in_addr`, then convert it back to a string with `inet_ntop` and confirm they match.

Explain in a comment block why skipping byte-order conversion causes silent bugs on little-endian machines.

### Exercise 2: TCP Time Server

Build a minimal TCP server that sends the current date and time to every client that connects:

1. Create a listening socket on port `7777` with `SO_REUSEADDR` set.
2. In the accept loop, call `time()` and `ctime()` to get the current time as a string.
3. Send the time string to the client using `send_all` (the robust helper from Section 2.4) to handle partial writes.
4. Close the client socket immediately after sending.
5. Test with `nc 127.0.0.1 7777` or a simple client that connects, reads one line, and prints it.

### Exercise 3: UDP Ping-Pong with Packet Loss Simulation

Extend the UDP sender/receiver from Section 3.2:

1. In the receiver, add a simulated 20% packet loss: generate a random number on each `recvfrom` and drop (do not echo) the packet 20% of the time, printing "DROPPED" to stderr.
2. In the sender, implement a timeout-based retry loop: after `sendto`, use `setsockopt(SO_RCVTIMEO)` to set a 500 ms receive timeout on the socket; if `recvfrom` returns `EAGAIN`/`EWOULDBLOCK`, retransmit up to 5 times before giving up.
3. Run both programs and observe the retry behavior in the output.

### Exercise 4: poll()-Based Echo Server Upgrade

Start from the `select()`-based server in Section 4.2 and rewrite the I/O multiplexing layer using `poll()`:

1. Replace `fd_set` / `FD_SET` / `FD_ISSET` with a `struct pollfd` array.
2. When a client disconnects, remove it by swapping with the last entry in the array and decrementing `nfds` (the same technique shown in Section 4.3).
3. Add a 5-second idle timeout to `poll()`: if no activity occurs for 5 seconds, print "Server idle..." and continue waiting.
4. Verify with `telnet` or `nc` that multiple clients can connect, send messages, and disconnect cleanly without affecting each other.

### Exercise 5: Length-Prefixed Message Protocol

Implement a complete request-response protocol using the length-prefix framing from Section 5.1:

1. Define a simple protocol: the client sends a length-prefixed message containing a command (`"UPPER"`, `"LOWER"`, or `"REVERSE"`), followed by a space, followed by a string (e.g., `"UPPER hello world"`).
2. The server reads the length-prefixed message with `recv_message`, parses the command and argument, transforms the string accordingly, and replies with a length-prefixed response.
3. The client receives and prints the response, then sends the next command from a predefined list.
4. Ensure both sides use `send_all` and `recv_exact` throughout to handle partial reads and writes correctly.

---
