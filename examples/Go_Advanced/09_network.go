// 20_network.go — TCP echo server and client
//
// Run: go run 20_network.go

package main

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"time"
)

func main() {
	// Start server
	go startServer()
	time.Sleep(100 * time.Millisecond)

	// Run client
	fmt.Println("=== TCP Echo Client ===")
	runClient()
}

func startServer() {
	listener, err := net.Listen("tcp", ":9876")
	if err != nil {
		log.Fatal(err)
	}
	defer listener.Close()

	for {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		go handleConn(conn)
	}
}

func handleConn(conn net.Conn) {
	defer conn.Close()
	scanner := bufio.NewScanner(conn)
	for scanner.Scan() {
		msg := scanner.Text()
		fmt.Fprintf(conn, "Echo: %s\n", msg)
	}
}

func runClient() {
	conn, err := net.Dial("tcp", "localhost:9876")
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()

	messages := []string{"Hello", "Go", "Network"}
	for _, msg := range messages {
		fmt.Fprintf(conn, "%s\n", msg)
		reply, _ := bufio.NewReader(conn).ReadString('\n')
		fmt.Printf("Sent: %s → Got: %s", msg, reply)
	}
}
