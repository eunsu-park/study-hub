/*
 * Exercises for Lesson 21: Network Programming
 * Topic: C_Programming
 * Solutions to practice problems from the lesson.
 *
 * Compile: gcc -Wall -Wextra -std=c11 -o ex21 21_network_programming.c
 *
 * Note: Exercises 2-5 require POSIX networking (Linux/macOS).
 * On macOS, compile without -lrt. On Linux, you may need -lpthread.
 * The code uses #ifdef guards so it compiles on all platforms,
 * but networking exercises only run on POSIX systems.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
  typedef int socklen_t;
#else
  #include <unistd.h>
  #include <arpa/inet.h>
  #include <netinet/in.h>
  #include <sys/socket.h>
  #include <sys/select.h>
  #include <poll.h>
  #include <time.h>
  #include <errno.h>
  #include <fcntl.h>
#endif

/* === Exercise 1: Byte-Order Conversion Practice === */
/* Problem: Demonstrate htons/htonl/ntohs/ntohl and inet_pton/inet_ntop. */

void exercise_1(void) {
    printf("=== Exercise 1: Byte-Order Conversion Practice ===\n");

    /* Part 1: Port conversion with htons/ntohs */
    uint16_t port = 8080;
    uint16_t port_net = htons(port);
    uint16_t port_back = ntohs(port_net);

    printf("\nPort conversion (htons / ntohs):\n");
    printf("  Host byte order:    %u (0x%04X)\n", port, port);
    printf("  Network byte order: %u (0x%04X)\n", port_net, port_net);
    printf("  Back to host:       %u (0x%04X)\n", port_back, port_back);

    /* Part 2: IP address conversion with htonl/ntohl */
    /* 192.168.10.1 = 192*2^24 + 168*2^16 + 10*2^8 + 1 = 0xC0A80A01 */
    uint32_t ip = (192U << 24) | (168U << 16) | (10U << 8) | 1U;
    uint32_t ip_net = htonl(ip);
    uint32_t ip_back = ntohl(ip_net);

    printf("\nIP conversion (htonl / ntohl):\n");
    printf("  Host byte order:    0x%08X\n", ip);
    printf("  Network byte order: 0x%08X\n", ip_net);
    printf("  Back to host:       0x%08X\n", ip_back);
    printf("  Verified match:     %s\n", ip == ip_back ? "YES" : "NO");

    /* Part 3: inet_pton / inet_ntop string conversion */
    const char *ip_str = "192.168.10.1";
    struct in_addr addr;
    char ip_buf[INET_ADDRSTRLEN];

    if (inet_pton(AF_INET, ip_str, &addr) == 1) {
        inet_ntop(AF_INET, &addr, ip_buf, sizeof(ip_buf));
        printf("\ninet_pton / inet_ntop:\n");
        printf("  Original string:  \"%s\"\n", ip_str);
        printf("  Binary (network): 0x%08X\n", addr.s_addr);
        printf("  Back to string:   \"%s\"\n", ip_buf);
        printf("  Strings match:    %s\n",
               strcmp(ip_str, ip_buf) == 0 ? "YES" : "NO");
    }

    /*
     * Why skipping byte-order conversion causes silent bugs:
     *
     * On little-endian machines (x86/x64, most modern CPUs), the byte order
     * in memory is reversed compared to network byte order (big-endian).
     *
     * Example: port 8080 = 0x1F90
     *   Big-endian (network):    [0x1F][0x90]
     *   Little-endian (host):    [0x90][0x1F]
     *
     * If you skip htons() and send 0x901F to a big-endian server, it reads
     * port 36895 instead of 8080. The connection goes to the wrong port
     * or fails silently -- no compiler warning, no runtime error.
     *
     * Similarly for IP: 192.168.10.1 becomes 1.10.168.192 on the wire,
     * causing packets to be routed to the wrong destination.
     */
    printf("\n  Key lesson: Always use htons/htonl before sending,\n");
    printf("  ntohs/ntohl after receiving. Skipping causes silent bugs\n");
    printf("  on little-endian machines (x86/ARM in LE mode).\n");
}

/* === Exercise 2: TCP Time Server === */
/* Problem: Build a TCP server that sends current date/time to each client. */

/* Robust send helper: handles partial writes */
static ssize_t send_all(int sockfd, const void *buf, size_t len) {
    const char *ptr = (const char *)buf;
    size_t remaining = len;

    while (remaining > 0) {
        ssize_t sent = send(sockfd, ptr, remaining, 0);
        if (sent <= 0) {
            if (errno == EINTR) continue; /* Interrupted, retry */
            return -1; /* Real error */
        }
        ptr += sent;
        remaining -= (size_t)sent;
    }
    return (ssize_t)len;
}

void exercise_2(void) {
    printf("\n=== Exercise 2: TCP Time Server ===\n");

    /*
     * This creates a TCP server on port 7777 that:
     * 1. Accepts a connection
     * 2. Sends the current time string
     * 3. Closes the client socket
     * 4. Repeats (for demonstration, we handle just 3 clients then exit)
     *
     * Test with: nc 127.0.0.1 7777
     */

    printf("  (Server code demonstration -- not starting a live server)\n\n");

    /* Show the implementation as documentation */
    printf("  Implementation outline:\n");
    printf("  1. socket(AF_INET, SOCK_STREAM, 0)  - create TCP socket\n");
    printf("  2. setsockopt(SO_REUSEADDR)          - allow port reuse\n");
    printf("  3. bind() to port 7777               - assign address\n");
    printf("  4. listen(backlog=5)                  - mark as passive\n");
    printf("  5. Loop:\n");
    printf("     a. accept()                        - wait for client\n");
    printf("     b. time() + ctime()                - get time string\n");
    printf("     c. send_all()                      - robust send\n");
    printf("     d. close(client_fd)                - done with client\n");

    /* Demonstrate the time string generation */
    time_t now = time(NULL);
    char *time_str = ctime(&now);
    printf("\n  Current time string: %s", time_str);

    /*
     * Full server code (compile separately to run):
     *
     *   int server_fd = socket(AF_INET, SOCK_STREAM, 0);
     *   int opt = 1;
     *   setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
     *
     *   struct sockaddr_in addr = {
     *       .sin_family = AF_INET,
     *       .sin_addr.s_addr = INADDR_ANY,
     *       .sin_port = htons(7777)
     *   };
     *   bind(server_fd, (struct sockaddr *)&addr, sizeof(addr));
     *   listen(server_fd, 5);
     *
     *   while (1) {
     *       int client_fd = accept(server_fd, NULL, NULL);
     *       if (client_fd < 0) continue;
     *
     *       time_t now = time(NULL);
     *       char *ts = ctime(&now);
     *       send_all(client_fd, ts, strlen(ts));
     *       close(client_fd);
     *   }
     *   close(server_fd);
     */
}

/* === Exercise 3: UDP Ping-Pong with Packet Loss Simulation === */
/* Problem: UDP sender with timeout-based retry and receiver with 20% simulated loss. */

void exercise_3(void) {
    printf("\n=== Exercise 3: UDP Ping-Pong with Packet Loss ===\n");

    /*
     * Receiver logic (simulated):
     * - recvfrom() to receive a packet
     * - Generate random number; if < 0.2, drop the packet (don't echo)
     * - Otherwise, sendto() to echo it back
     *
     * Sender logic (simulated):
     * - sendto() to send a packet
     * - Set SO_RCVTIMEO to 500ms
     * - Wait for echo with recvfrom()
     * - If EAGAIN/EWOULDBLOCK (timeout), retry up to 5 times
     */

    printf("  Simulating UDP ping-pong with 20%% packet loss:\n\n");

    srand((unsigned)time(NULL));
    int total_sent = 10;
    int total_received = 0;
    int total_retries = 0;

    for (int seq = 1; seq <= total_sent; seq++) {
        int retries = 0;
        int acked = 0;

        while (retries < 5 && !acked) {
            if (retries > 0) {
                printf("    [Retry %d] Resending packet %d\n", retries, seq);
                total_retries++;
            } else {
                printf("  Sending packet %d... ", seq);
            }

            /* Simulate 20% packet loss at receiver */
            int lost = (rand() % 100) < 20;
            if (lost) {
                printf("DROPPED (simulated loss)\n");
                printf("    [Timeout 500ms] No response\n");
            } else {
                printf("ACK received\n");
                acked = 1;
                total_received++;
            }
            retries++;
        }

        if (!acked) {
            printf("    GAVE UP after 5 retries for packet %d\n", seq);
        }
    }

    printf("\n  Summary:\n");
    printf("  Packets sent:     %d\n", total_sent);
    printf("  Packets received: %d\n", total_received);
    printf("  Total retries:    %d\n", total_retries);

    /*
     * Key concepts:
     * - SO_RCVTIMEO sets a read timeout on the socket
     * - EAGAIN/EWOULDBLOCK indicate timeout (no data available)
     * - Application-level retry is necessary for UDP reliability
     * - TCP handles this automatically with built-in retransmission
     */
}

/* === Exercise 4: poll()-Based Echo Server Upgrade === */
/* Problem: Rewrite select()-based server using poll() with idle timeout. */

void exercise_4(void) {
    printf("\n=== Exercise 4: poll()-Based Echo Server ===\n");

    printf("  (Server architecture demonstration)\n\n");

    /*
     * Key differences between select() and poll():
     *
     * select():
     * - Uses fd_set bitmask (FD_SETSIZE limit, typically 1024)
     * - Must rebuild fd_set before each call
     * - O(n) scan on return to find ready fds
     *
     * poll():
     * - Uses struct pollfd array (no FD_SETSIZE limit)
     * - Array persists between calls (no rebuild needed)
     * - Events and revents are separate fields (cleaner API)
     * - Swap-with-last removal: O(1) client disconnect handling
     */

    printf("  poll() advantages over select():\n");
    printf("  1. No FD_SETSIZE limit (scales to thousands of fds)\n");
    printf("  2. No need to rebuild descriptor sets each iteration\n");
    printf("  3. Separate events/revents fields (less error-prone)\n");
    printf("  4. Swap-with-last technique for O(1) client removal\n\n");

    /* Demonstrate the swap-with-last removal technique */
    printf("  Swap-with-last removal technique:\n");
    int fds[] = {3, 5, 7, 9, 11};
    int nfds = 5;

    printf("  Before: [");
    for (int i = 0; i < nfds; i++) printf("%d%s", fds[i], i < nfds-1 ? ", " : "");
    printf("] (nfds=%d)\n", nfds);

    /* Remove fd=7 (index 2) by swapping with last */
    int remove_idx = 2;
    printf("  Remove fd=%d at index %d:\n", fds[remove_idx], remove_idx);
    fds[remove_idx] = fds[nfds - 1]; /* Swap with last */
    nfds--;                            /* Shrink array */

    printf("  After:  [");
    for (int i = 0; i < nfds; i++) printf("%d%s", fds[i], i < nfds-1 ? ", " : "");
    printf("] (nfds=%d)\n", nfds);

    /*
     * Full poll() echo server skeleton:
     *
     *   #define MAX_CLIENTS 100
     *   struct pollfd pfds[MAX_CLIENTS + 1];
     *   int nfds = 1;
     *
     *   // pfds[0] is always the listening socket
     *   pfds[0].fd = server_fd;
     *   pfds[0].events = POLLIN;
     *
     *   while (1) {
     *       int ready = poll(pfds, nfds, 5000); // 5s idle timeout
     *
     *       if (ready == 0) {
     *           fprintf(stderr, "Server idle...\n");
     *           continue;
     *       }
     *
     *       // Check listener
     *       if (pfds[0].revents & POLLIN) {
     *           int client_fd = accept(server_fd, NULL, NULL);
     *           pfds[nfds].fd = client_fd;
     *           pfds[nfds].events = POLLIN;
     *           nfds++;
     *       }
     *
     *       // Check clients
     *       for (int i = 1; i < nfds; i++) {
     *           if (pfds[i].revents & POLLIN) {
     *               char buf[1024];
     *               int n = recv(pfds[i].fd, buf, sizeof(buf), 0);
     *               if (n <= 0) {
     *                   // Client disconnected: swap-with-last
     *                   close(pfds[i].fd);
     *                   pfds[i] = pfds[nfds - 1];
     *                   nfds--;
     *                   i--; // Recheck this index
     *               } else {
     *                   send(pfds[i].fd, buf, n, 0); // Echo
     *               }
     *           }
     *       }
     *   }
     */
}

/* === Exercise 5: Length-Prefixed Message Protocol === */
/* Problem: Implement a request-response protocol with length-prefix framing. */

/* Encode a length-prefixed message into a buffer */
static size_t encode_message(const char *msg, uint8_t *buf, size_t buf_size) {
    /*
     * Length-prefix framing: [4 bytes length][payload]
     *
     * Why length-prefix instead of delimiter?
     * - Delimiter ('\n'): Cannot send messages containing the delimiter
     * - Length-prefix: Any byte value is valid in the payload
     * - Receiver knows exactly how many bytes to read (no scanning)
     * - More efficient: no need to scan for delimiters in large messages
     */
    uint32_t len = (uint32_t)strlen(msg);
    if (4 + len > buf_size) return 0;

    /* Store length in network byte order (big-endian) */
    uint32_t net_len = htonl(len);
    memcpy(buf, &net_len, 4);
    memcpy(buf + 4, msg, len);

    return 4 + len;
}

/* Decode a length-prefixed message from a buffer */
static int decode_message(const uint8_t *buf, size_t buf_len,
                          char *msg, size_t msg_size) {
    if (buf_len < 4) return -1; /* Not enough data for length header */

    uint32_t net_len;
    memcpy(&net_len, buf, 4);
    uint32_t len = ntohl(net_len);

    if (buf_len < 4 + len) return -1; /* Incomplete message */
    if (len >= msg_size) return -1;    /* Message too large for buffer */

    memcpy(msg, buf + 4, len);
    msg[len] = '\0';

    return (int)(4 + len); /* Total bytes consumed */
}

/* Process a command: UPPER, LOWER, or REVERSE */
static void process_command(const char *input, char *output, size_t out_size) {
    /* Parse command and argument (e.g., "UPPER hello world") */
    char cmd[16] = {0};
    const char *arg = NULL;

    /* Extract command (first word) */
    int i = 0;
    while (input[i] && input[i] != ' ' && i < 15) {
        cmd[i] = input[i];
        i++;
    }
    cmd[i] = '\0';

    /* Skip space to get argument */
    if (input[i] == ' ') {
        arg = input + i + 1;
    } else {
        snprintf(output, out_size, "ERROR: missing argument");
        return;
    }

    size_t arg_len = strlen(arg);
    if (arg_len >= out_size) {
        snprintf(output, out_size, "ERROR: argument too long");
        return;
    }

    if (strcmp(cmd, "UPPER") == 0) {
        for (size_t j = 0; j < arg_len; j++) {
            output[j] = (char)toupper((unsigned char)arg[j]);
        }
        output[arg_len] = '\0';
    } else if (strcmp(cmd, "LOWER") == 0) {
        for (size_t j = 0; j < arg_len; j++) {
            output[j] = (char)tolower((unsigned char)arg[j]);
        }
        output[arg_len] = '\0';
    } else if (strcmp(cmd, "REVERSE") == 0) {
        for (size_t j = 0; j < arg_len; j++) {
            output[j] = arg[arg_len - 1 - j];
        }
        output[arg_len] = '\0';
    } else {
        snprintf(output, out_size, "ERROR: unknown command '%s'", cmd);
    }
}

void exercise_5(void) {
    printf("\n=== Exercise 5: Length-Prefixed Message Protocol ===\n");

    /* Simulate client-server message exchange */
    const char *commands[] = {
        "UPPER hello world",
        "LOWER HELLO WORLD",
        "REVERSE abcdef",
        "UPPER Network Programming",
    };
    int num_commands = (int)(sizeof(commands) / sizeof(commands[0]));

    uint8_t wire_buf[256];

    for (int i = 0; i < num_commands; i++) {
        printf("\n  --- Request %d ---\n", i + 1);

        /* Client side: encode and "send" */
        size_t encoded_len = encode_message(commands[i], wire_buf, sizeof(wire_buf));
        uint32_t payload_len;
        memcpy(&payload_len, wire_buf, 4);
        payload_len = ntohl(payload_len);

        printf("  Client sends: \"%s\"\n", commands[i]);
        printf("  Wire format: [len=%u][payload=%zu bytes total]\n",
               payload_len, encoded_len);

        /* Server side: decode, process, encode response */
        char request[256];
        int consumed = decode_message(wire_buf, encoded_len, request, sizeof(request));
        if (consumed < 0) {
            printf("  ERROR: Failed to decode message\n");
            continue;
        }

        char response[256];
        process_command(request, response, sizeof(response));

        /* Server encodes response */
        size_t resp_len = encode_message(response, wire_buf, sizeof(wire_buf));
        (void)resp_len; /* Used in actual send_all() */

        /* Client decodes response */
        char decoded_response[256];
        decode_message(wire_buf, resp_len, decoded_response, sizeof(decoded_response));

        printf("  Server response: \"%s\"\n", decoded_response);
    }

    printf("\n  Protocol design notes:\n");
    printf("  - 4-byte network-order length prefix before every message\n");
    printf("  - send_all() handles partial writes (TCP can split any send)\n");
    printf("  - recv_exact() reads exactly N bytes (loop until complete)\n");
    printf("  - Both sides use same framing: symmetric and simple\n");
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    exercise_5();

    printf("\nAll exercises completed!\n");
    return 0;
}
