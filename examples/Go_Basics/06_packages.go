// 06_packages.go — Package design and functional options pattern
//
// Run: go run 06_packages.go

package main

import (
	"fmt"
	"time"
)

// Server demonstrates the functional options pattern
type Server struct {
	host    string
	port    int
	timeout time.Duration
	tls     bool
}

type Option func(*Server)

func WithPort(port int) Option {
	return func(s *Server) { s.port = port }
}

func WithTimeout(d time.Duration) Option {
	return func(s *Server) { s.timeout = d }
}

func WithTLS(enable bool) Option {
	return func(s *Server) { s.tls = enable }
}

func NewServer(host string, opts ...Option) *Server {
	s := &Server{
		host:    host,
		port:    8080,
		timeout: 30 * time.Second,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

func (s *Server) String() string {
	protocol := "http"
	if s.tls {
		protocol = "https"
	}
	return fmt.Sprintf("%s://%s:%d (timeout=%v)", protocol, s.host, s.port, s.timeout)
}

func main() {
	fmt.Println("=== Functional Options Pattern ===")

	s1 := NewServer("localhost")
	fmt.Println("Default:", s1)

	s2 := NewServer("api.example.com",
		WithPort(443),
		WithTLS(true),
		WithTimeout(60*time.Second),
	)
	fmt.Println("Custom:", s2)
}
