# 20. Network Programming

**Previous**: [Build and Deploy](./08_Build_and_Deploy.md) | **Next**: [Cloud Native Patterns](./10_Cloud_Native_Patterns.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Build TCP and UDP servers and clients with the `net` package
2. Implement custom protocols over TCP
3. Use WebSockets for real-time communication
4. Build gRPC services with Protocol Buffers
5. Handle connection management and timeouts

---

Go's `net` package provides low-level networking primitives. Combined with goroutines, it makes concurrent network programming natural — each connection gets its own goroutine, and the runtime handles the multiplexing.

## Table of Contents
1. [TCP Server and Client](#1-tcp-server-and-client)
2. [UDP](#2-udp)
3. [Custom Protocols](#3-custom-protocols)
4. [WebSocket](#4-websocket)
5. [gRPC](#5-grpc)
6. [Connection Management](#6-connection-management)
7. [Summary](#7-summary)

---

## 1. TCP Server and Client

### 1.1 TCP Echo Server

```go
package main

import (
    "bufio"
    "fmt"
    "io"
    "log"
    "net"
)

func main() {
    listener, err := net.Listen("tcp", ":9000")
    if err != nil {
        log.Fatal(err)
    }
    defer listener.Close()
    log.Println("TCP server listening on :9000")

    for {
        conn, err := listener.Accept()
        if err != nil {
            log.Println("accept error:", err)
            continue
        }
        go handleConnection(conn) // One goroutine per connection
    }
}

func handleConnection(conn net.Conn) {
    defer conn.Close()
    addr := conn.RemoteAddr().String()
    log.Printf("Client connected: %s", addr)

    scanner := bufio.NewScanner(conn)
    for scanner.Scan() {
        line := scanner.Text()
        log.Printf("[%s] Received: %s", addr, line)
        fmt.Fprintf(conn, "Echo: %s\n", line)
    }

    log.Printf("Client disconnected: %s", addr)
}
```

### 1.2 TCP Client

```go
func main() {
    conn, err := net.Dial("tcp", "localhost:9000")
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()

    // Send
    fmt.Fprintln(conn, "Hello, server!")

    // Receive
    scanner := bufio.NewScanner(conn)
    if scanner.Scan() {
        fmt.Println("Server:", scanner.Text())
    }
}
```

### 1.3 TCP with Context and Timeout

```go
func connectWithTimeout(addr string, timeout time.Duration) (net.Conn, error) {
    dialer := &net.Dialer{
        Timeout:   timeout,
        KeepAlive: 30 * time.Second,
    }
    return dialer.Dial("tcp", addr)
}

func handleWithDeadline(conn net.Conn) {
    defer conn.Close()

    for {
        // Set deadline for each read operation
        conn.SetReadDeadline(time.Now().Add(30 * time.Second))

        buf := make([]byte, 1024)
        n, err := conn.Read(buf)
        if err != nil {
            if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
                log.Println("Read timeout, closing connection")
            }
            return
        }

        conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
        conn.Write(buf[:n])
    }
}
```

---

## 2. UDP

### 2.1 UDP Server

```go
func main() {
    addr, _ := net.ResolveUDPAddr("udp", ":9001")
    conn, err := net.ListenUDP("udp", addr)
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()
    log.Println("UDP server listening on :9001")

    buf := make([]byte, 1024)
    for {
        n, remoteAddr, err := conn.ReadFromUDP(buf)
        if err != nil {
            log.Println("read error:", err)
            continue
        }

        msg := string(buf[:n])
        log.Printf("From %s: %s", remoteAddr, msg)

        // Reply
        response := fmt.Sprintf("Echo: %s", msg)
        conn.WriteToUDP([]byte(response), remoteAddr)
    }
}
```

### 2.2 UDP Client

```go
func main() {
    addr, _ := net.ResolveUDPAddr("udp", "localhost:9001")
    conn, err := net.DialUDP("udp", nil, addr)
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()

    conn.Write([]byte("Hello UDP!"))

    buf := make([]byte, 1024)
    conn.SetReadDeadline(time.Now().Add(5 * time.Second))
    n, err := conn.Read(buf)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("Response:", string(buf[:n]))
}
```

---

## 3. Custom Protocols

### 3.1 Length-Prefixed Protocol

```go
// Protocol: [4-byte length][payload]

func sendMessage(conn net.Conn, msg []byte) error {
    // Write length (4 bytes, big-endian)
    length := uint32(len(msg))
    if err := binary.Write(conn, binary.BigEndian, length); err != nil {
        return err
    }
    // Write payload
    _, err := conn.Write(msg)
    return err
}

func receiveMessage(conn net.Conn) ([]byte, error) {
    // Read length
    var length uint32
    if err := binary.Read(conn, binary.BigEndian, &length); err != nil {
        return nil, err
    }

    if length > 10*1024*1024 { // 10 MB limit
        return nil, fmt.Errorf("message too large: %d bytes", length)
    }

    // Read payload
    buf := make([]byte, length)
    _, err := io.ReadFull(conn, buf)
    return buf, err
}
```

### 3.2 JSON-RPC Style Protocol

```go
type Request struct {
    Method string          `json:"method"`
    ID     int             `json:"id"`
    Params json.RawMessage `json:"params"`
}

type Response struct {
    ID     int    `json:"id"`
    Result any    `json:"result,omitempty"`
    Error  string `json:"error,omitempty"`
}

func handleRPC(conn net.Conn) {
    defer conn.Close()
    decoder := json.NewDecoder(conn)
    encoder := json.NewEncoder(conn)

    for {
        var req Request
        if err := decoder.Decode(&req); err != nil {
            return
        }

        resp := processRPC(req)
        encoder.Encode(resp)
    }
}

func processRPC(req Request) Response {
    switch req.Method {
    case "echo":
        var msg string
        json.Unmarshal(req.Params, &msg)
        return Response{ID: req.ID, Result: msg}
    case "add":
        var nums [2]int
        json.Unmarshal(req.Params, &nums)
        return Response{ID: req.ID, Result: nums[0] + nums[1]}
    default:
        return Response{ID: req.ID, Error: "unknown method"}
    }
}
```

---

## 4. WebSocket

### 4.1 WebSocket Server

```go
import "golang.org/x/net/websocket"

func wsHandler(ws *websocket.Conn) {
    defer ws.Close()
    log.Printf("WebSocket connected: %s", ws.RemoteAddr())

    for {
        var msg string
        if err := websocket.Message.Receive(ws, &msg); err != nil {
            log.Println("Read error:", err)
            return
        }

        log.Printf("Received: %s", msg)
        reply := fmt.Sprintf("Echo: %s", msg)
        websocket.Message.Send(ws, reply)
    }
}

func main() {
    http.Handle("/ws", websocket.Handler(wsHandler))
    http.Handle("/", http.FileServer(http.Dir("static")))
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### 4.2 Chat Server with gorilla/websocket

```go
import "github.com/gorilla/websocket"

var upgrader = websocket.Upgrader{
    CheckOrigin: func(r *http.Request) bool { return true },
}

type Hub struct {
    clients    map[*websocket.Conn]bool
    broadcast  chan []byte
    register   chan *websocket.Conn
    unregister chan *websocket.Conn
    mu         sync.RWMutex
}

func (h *Hub) Run() {
    for {
        select {
        case conn := <-h.register:
            h.mu.Lock()
            h.clients[conn] = true
            h.mu.Unlock()

        case conn := <-h.unregister:
            h.mu.Lock()
            delete(h.clients, conn)
            h.mu.Unlock()

        case msg := <-h.broadcast:
            h.mu.RLock()
            for conn := range h.clients {
                if err := conn.WriteMessage(websocket.TextMessage, msg); err != nil {
                    conn.Close()
                    delete(h.clients, conn)
                }
            }
            h.mu.RUnlock()
        }
    }
}
```

---

## 5. gRPC

### 5.1 Protocol Buffer Definition

```protobuf
// proto/user.proto
syntax = "proto3";
package user;
option go_package = "./userpb";

service UserService {
    rpc GetUser(GetUserRequest) returns (User);
    rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);
    rpc CreateUser(CreateUserRequest) returns (User);
}

message User {
    int64 id = 1;
    string name = 2;
    string email = 3;
}

message GetUserRequest {
    int64 id = 1;
}

message ListUsersRequest {
    int32 page = 1;
    int32 per_page = 2;
}

message ListUsersResponse {
    repeated User users = 1;
    int32 total = 2;
}

message CreateUserRequest {
    string name = 1;
    string email = 2;
}
```

### 5.2 gRPC Server

```go
import (
    "google.golang.org/grpc"
    pb "github.com/user/project/userpb"
)

type userServer struct {
    pb.UnimplementedUserServiceServer
    store UserStore
}

func (s *userServer) GetUser(ctx context.Context, req *pb.GetUserRequest) (*pb.User, error) {
    user, err := s.store.Get(ctx, req.Id)
    if err != nil {
        return nil, status.Errorf(codes.NotFound, "user %d not found", req.Id)
    }
    return &pb.User{Id: user.ID, Name: user.Name, Email: user.Email}, nil
}

func main() {
    lis, _ := net.Listen("tcp", ":50051")
    srv := grpc.NewServer()
    pb.RegisterUserServiceServer(srv, &userServer{})
    log.Println("gRPC server on :50051")
    srv.Serve(lis)
}
```

---

## 6. Connection Management

### 6.1 Connection Pool

```go
type ConnPool struct {
    mu      sync.Mutex
    conns   chan net.Conn
    factory func() (net.Conn, error)
    maxSize int
}

func NewConnPool(maxSize int, factory func() (net.Conn, error)) *ConnPool {
    return &ConnPool{
        conns:   make(chan net.Conn, maxSize),
        factory: factory,
        maxSize: maxSize,
    }
}

func (p *ConnPool) Get() (net.Conn, error) {
    select {
    case conn := <-p.conns:
        return conn, nil
    default:
        return p.factory()
    }
}

func (p *ConnPool) Put(conn net.Conn) {
    select {
    case p.conns <- conn:
    default:
        conn.Close() // Pool full, discard
    }
}
```

---

## 7. Summary

### Key Takeaways

1. **Goroutine-per-connection** — Go's concurrency model makes this natural and efficient.
2. **TCP for reliable** — use for protocols requiring guaranteed delivery.
3. **UDP for speed** — use for real-time, lossy-tolerant applications.
4. **WebSocket for real-time** — bidirectional communication over HTTP.
5. **gRPC for services** — Protocol Buffers for schema, HTTP/2 for transport.
6. **Always set deadlines** — prevent hanging connections with `SetDeadline`.

---

## Exercises

### Exercise 1: Chat Server
Build a TCP chat server where clients can join rooms, send messages, and receive broadcasts.

### Exercise 2: File Transfer
Implement a file transfer protocol over TCP using length-prefixed messages with progress reporting.

### Exercise 3: DNS Resolver
Build a simple DNS resolver using UDP that sends DNS queries and parses responses.

### Exercise 4: gRPC Service
Define a gRPC service for a todo list. Implement server and client, including streaming for real-time updates.
