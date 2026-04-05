// 19_build.go — Build info and embed demonstration
//
// Run: go run -ldflags "-X main.version=1.0.0" 19_build.go

package main

import (
	"fmt"
	"runtime"
)

var (
	version = "dev"
	commit  = "unknown"
	date    = "unknown"
)

func main() {
	fmt.Println("=== Build Information ===")
	fmt.Printf("Version:  %s\n", version)
	fmt.Printf("Commit:   %s\n", commit)
	fmt.Printf("Date:     %s\n", date)
	fmt.Printf("Go:       %s\n", runtime.Version())
	fmt.Printf("OS/Arch:  %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("Compiler: %s\n", runtime.Compiler)

	fmt.Println("\n=== Cross-Compilation Targets ===")
	targets := [][2]string{
		{"linux", "amd64"},
		{"linux", "arm64"},
		{"darwin", "amd64"},
		{"darwin", "arm64"},
		{"windows", "amd64"},
	}
	for _, t := range targets {
		ext := ""
		if t[0] == "windows" {
			ext = ".exe"
		}
		fmt.Printf("  GOOS=%s GOARCH=%s → myapp-%s-%s%s\n",
			t[0], t[1], t[0], t[1], ext)
	}
}
