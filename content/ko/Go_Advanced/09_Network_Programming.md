# 20. 네트워크 프로그래밍

**이전**: [빌드와 배포](./08_Build_and_Deploy.md) | **다음**: [클라우드 네이티브 패턴](./10_Cloud_Native_Patterns.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있다:

1. `net` 패키지로 TCP 및 UDP 서버와 클라이언트를 구축한다
2. TCP 위에 커스텀 프로토콜을 구현한다
3. 실시간 통신을 위해 WebSocket을 사용한다
4. Protocol Buffers를 사용하여 gRPC 서비스를 구축한다
5. 연결 관리와 타임아웃을 처리한다

---

Go의 `net` 패키지는 저수준 네트워킹 프리미티브를 제공한다. 고루틴과 결합하면 동시 네트워크 프로그래밍이 자연스러워진다 — 각 연결이 자체 고루틴을 가지며, 런타임이 멀티플렉싱을 처리한다.

## 목차
1. [TCP 서버와 클라이언트](#1-tcp-서버와-클라이언트)
2. [UDP](#2-udp)
3. [커스텀 프로토콜](#3-커스텀-프로토콜)
4. [WebSocket](#4-websocket)
5. [gRPC](#5-grpc)
6. [연결 관리](#6-연결-관리)
7. [요약](#7-요약)

---

## 1. TCP 서버와 클라이언트

### 1.1 TCP 에코 서버

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

### 1.2 TCP 클라이언트

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

### 1.3 Context와 타임아웃이 있는 TCP

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

### 2.1 UDP 서버

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

### 2.2 UDP 클라이언트

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

## 3. 커스텀 프로토콜

### 3.1 길이 접두사 프로토콜

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

### 3.2 JSON-RPC 스타일 프로토콜

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

### 4.1 WebSocket 서버

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

### 4.2 gorilla/websocket을 사용한 채팅 서버

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

### 5.1 Protocol Buffer 정의

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

### 5.2 gRPC 서버

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

## 6. 연결 관리

### 6.1 연결 풀

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

## 7. 요약

### 핵심 포인트

1. **연결당 고루틴** — Go의 동시성 모델이 이를 자연스럽고 효율적으로 만든다.
2. **신뢰성에는 TCP** — 보장된 전달이 필요한 프로토콜에 사용한다.
3. **속도에는 UDP** — 실시간, 손실 허용 가능한 애플리케이션에 사용한다.
4. **실시간에는 WebSocket** — HTTP를 통한 양방향 통신을 제공한다.
5. **서비스에는 gRPC** — 스키마에 Protocol Buffers, 전송에 HTTP/2를 사용한다.
6. **항상 데드라인을 설정한다** — `SetDeadline`으로 중단된 연결을 방지한다.

---

## 연습 문제

### 연습 1: 채팅 서버
클라이언트가 방에 참가하고, 메시지를 보내고, 브로드캐스트를 수신할 수 있는 TCP 채팅 서버를 구축한다.

### 연습 2: 파일 전송
진행률 보고와 함께 길이 접두사 메시지를 사용하여 TCP를 통한 파일 전송 프로토콜을 구현한다.

### 연습 3: DNS 리졸버
UDP를 사용하여 DNS 쿼리를 보내고 응답을 파싱하는 간단한 DNS 리졸버를 만든다.

### 연습 4: gRPC 서비스
할 일 목록을 위한 gRPC 서비스를 정의한다. 실시간 업데이트를 위한 스트리밍을 포함하여 서버와 클라이언트를 구현한다.
