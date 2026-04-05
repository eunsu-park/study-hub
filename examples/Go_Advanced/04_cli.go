// 15_cli.go — CLI tool with flags and subcommands
//
// Run: go run 15_cli.go -name Alice -count 3
// Run: go run 15_cli.go -verbose

package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"
)

func main() {
	name := flag.String("name", "World", "name to greet")
	count := flag.Int("count", 1, "number of greetings")
	verbose := flag.Bool("verbose", false, "enable verbose output")
	upper := flag.Bool("upper", false, "uppercase output")

	flag.Parse()

	if *verbose {
		fmt.Fprintf(os.Stderr, "[%s] Starting with name=%s count=%d\n",
			time.Now().Format("15:04:05"), *name, *count)
	}

	for i := 0; i < *count; i++ {
		msg := fmt.Sprintf("Hello, %s!", *name)
		if *upper {
			msg = strings.ToUpper(msg)
		}
		fmt.Println(msg)
	}

	// Show remaining args
	if flag.NArg() > 0 {
		fmt.Println("Extra args:", flag.Args())
	}

	if *verbose {
		fmt.Fprintf(os.Stderr, "[%s] Done\n", time.Now().Format("15:04:05"))
	}
}
