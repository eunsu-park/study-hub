# 17. Reflection and Code Generation

**Previous**: [Advanced Types](./05_Advanced_Types.md) | **Next**: [Performance Profiling](./07_Performance_Profiling.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Use the `reflect` package to inspect types and values at runtime
2. Read and parse struct tags programmatically
3. Build dynamic JSON/XML marshalers using reflection
4. Generate Go code with `go generate` and templates
5. Understand the performance implications of reflection

---

Reflection lets you inspect and manipulate types and values at runtime. Code generation lets you produce type-safe code at build time. Both are advanced tools — use reflection sparingly and prefer code generation when possible.

## Table of Contents
1. [reflect Package Basics](#1-reflect-package-basics)
2. [Inspecting Structs](#2-inspecting-structs)
3. [Struct Tags](#3-struct-tags)
4. [Dynamic Values](#4-dynamic-values)
5. [Code Generation](#5-code-generation)
6. [Practical Applications](#6-practical-applications)
7. [Summary](#7-summary)

---

## 1. reflect Package Basics

### 1.1 Type and Value

```go
package main

import (
    "fmt"
    "reflect"
)

func main() {
    x := 42
    s := "hello"
    f := 3.14

    // reflect.TypeOf — returns the type
    fmt.Println(reflect.TypeOf(x))  // int
    fmt.Println(reflect.TypeOf(s))  // string
    fmt.Println(reflect.TypeOf(f))  // float64

    // reflect.ValueOf — returns the value (wrapped)
    v := reflect.ValueOf(x)
    fmt.Println(v.Type())     // int
    fmt.Println(v.Kind())     // int
    fmt.Println(v.Int())      // 42
    fmt.Println(v.Interface()) // 42 (as any)

    // Kind vs Type
    type MyInt int
    var mi MyInt = 10
    t := reflect.TypeOf(mi)
    fmt.Println(t)        // main.MyInt (named type)
    fmt.Println(t.Kind()) // int (underlying kind)
}
```

### 1.2 Kinds

```go
func describeType(v any) {
    t := reflect.TypeOf(v)
    fmt.Printf("Type: %-20s Kind: %s\n", t, t.Kind())
}

func main() {
    describeType(42)                   // int, int
    describeType("hello")              // string, string
    describeType([]int{1, 2})          // []int, slice
    describeType(map[string]int{})     // map[string]int, map
    describeType(struct{ X int }{})    // struct { X int }, struct
    describeType(&struct{}{})          // *struct {}, ptr
    describeType(func() {})            // func(), func
    describeType(make(chan int))        // chan int, chan

    // All reflect.Kinds:
    // Bool, Int, Int8...Int64, Uint...Uint64, Uintptr
    // Float32, Float64, Complex64, Complex128
    // Array, Chan, Func, Interface, Map, Pointer, Slice, String, Struct
    // UnsafePointer
}
```

---

## 2. Inspecting Structs

### 2.1 Iterating Fields

```go
type User struct {
    ID        int       `json:"id" db:"id"`
    Name      string    `json:"name" db:"name"`
    Email     string    `json:"email" db:"email"`
    CreatedAt time.Time `json:"created_at" db:"created_at"`
    password  string    // unexported
}

func inspectStruct(v any) {
    t := reflect.TypeOf(v)
    if t.Kind() == reflect.Ptr {
        t = t.Elem()
    }

    fmt.Printf("Struct: %s (%d fields)\n", t.Name(), t.NumField())
    for i := 0; i < t.NumField(); i++ {
        field := t.Field(i)
        fmt.Printf("  %-15s %-10s exported=%-5t tag=%s\n",
            field.Name, field.Type, field.IsExported(), field.Tag)
    }
}

func main() {
    u := User{ID: 1, Name: "Alice", Email: "alice@example.com"}
    inspectStruct(u)
    // Struct: User (5 fields)
    //   ID              int        exported=true  tag=json:"id" db:"id"
    //   Name            string     exported=true  tag=json:"name" db:"name"
    //   ...
}
```

### 2.2 Reading Field Values

```go
func structToMap(v any) map[string]any {
    result := make(map[string]any)
    val := reflect.ValueOf(v)
    typ := val.Type()

    if typ.Kind() == reflect.Ptr {
        val = val.Elem()
        typ = val.Type()
    }

    for i := 0; i < typ.NumField(); i++ {
        field := typ.Field(i)
        if !field.IsExported() {
            continue
        }
        result[field.Name] = val.Field(i).Interface()
    }
    return result
}

func main() {
    u := User{ID: 1, Name: "Alice", Email: "alice@example.com"}
    m := structToMap(u)
    fmt.Println(m) // map[CreatedAt:0001-01-01... Email:alice@example.com ID:1 Name:Alice]
}
```

---

## 3. Struct Tags

### 3.1 Parsing Tags

```go
func getJSONFieldNames(v any) []string {
    t := reflect.TypeOf(v)
    if t.Kind() == reflect.Ptr {
        t = t.Elem()
    }

    var names []string
    for i := 0; i < t.NumField(); i++ {
        field := t.Field(i)
        tag := field.Tag.Get("json")
        if tag == "" || tag == "-" {
            continue
        }
        // Handle "name,omitempty" format
        name := strings.Split(tag, ",")[0]
        names = append(names, name)
    }
    return names
}

func main() {
    names := getJSONFieldNames(User{})
    fmt.Println(names) // [id name email created_at]
}
```

### 3.2 Custom Tags

```go
type Config struct {
    Host    string `env:"APP_HOST" default:"localhost"`
    Port    int    `env:"APP_PORT" default:"8080"`
    Debug   bool   `env:"APP_DEBUG" default:"false"`
    DBUrl   string `env:"DATABASE_URL" required:"true"`
}

func loadFromEnv(cfg any) error {
    v := reflect.ValueOf(cfg)
    if v.Kind() != reflect.Ptr || v.Elem().Kind() != reflect.Struct {
        return fmt.Errorf("expected pointer to struct")
    }
    v = v.Elem()
    t := v.Type()

    for i := 0; i < t.NumField(); i++ {
        field := t.Field(i)
        fieldVal := v.Field(i)

        envKey := field.Tag.Get("env")
        if envKey == "" {
            continue
        }

        envVal := os.Getenv(envKey)
        if envVal == "" {
            envVal = field.Tag.Get("default")
        }
        if envVal == "" && field.Tag.Get("required") == "true" {
            return fmt.Errorf("required env var %s not set", envKey)
        }

        switch field.Type.Kind() {
        case reflect.String:
            fieldVal.SetString(envVal)
        case reflect.Int:
            n, _ := strconv.Atoi(envVal)
            fieldVal.SetInt(int64(n))
        case reflect.Bool:
            b, _ := strconv.ParseBool(envVal)
            fieldVal.SetBool(b)
        }
    }
    return nil
}
```

---

## 4. Dynamic Values

### 4.1 Setting Values

```go
func main() {
    x := 42
    v := reflect.ValueOf(&x).Elem() // Must pass pointer and dereference
    fmt.Println(v.CanSet())          // true

    v.SetInt(100)
    fmt.Println(x) // 100

    // Cannot set non-addressable values
    v2 := reflect.ValueOf(42)
    fmt.Println(v2.CanSet()) // false — not a pointer
}
```

### 4.2 Creating Values Dynamically

```go
func main() {
    // Create a new struct dynamically
    userType := reflect.TypeOf(User{})
    newUser := reflect.New(userType) // Returns *User
    elem := newUser.Elem()

    elem.FieldByName("ID").SetInt(42)
    elem.FieldByName("Name").SetString("Dynamic")
    elem.FieldByName("Email").SetString("dyn@example.com")

    user := newUser.Interface().(*User)
    fmt.Printf("%+v\n", *user)

    // Create a slice dynamically
    sliceType := reflect.SliceOf(reflect.TypeOf(0))
    slice := reflect.MakeSlice(sliceType, 0, 10)
    slice = reflect.Append(slice, reflect.ValueOf(1))
    slice = reflect.Append(slice, reflect.ValueOf(2))
    fmt.Println(slice.Interface()) // [1 2]

    // Create a map dynamically
    mapType := reflect.MapOf(reflect.TypeOf(""), reflect.TypeOf(0))
    m := reflect.MakeMap(mapType)
    m.SetMapIndex(reflect.ValueOf("key"), reflect.ValueOf(42))
    fmt.Println(m.Interface()) // map[key:42]
}
```

---

## 5. Code Generation

### 5.1 go generate

```go
// In your source file, add a generate directive:
//go:generate stringer -type=Color

type Color int

const (
    Red Color = iota
    Green
    Blue
)

// Running `go generate ./...` creates color_string.go with:
// func (c Color) String() string { ... }
```

### 5.2 Template-Based Generation

```go
// gen/main.go — code generator
package main

import (
    "os"
    "text/template"
)

const tmpl = `// Code generated by gen; DO NOT EDIT.
package {{.Package}}

type {{.Name}}Set struct {
    items map[{{.Type}}]struct{}
}

func New{{.Name}}Set() *{{.Name}}Set {
    return &{{.Name}}Set{items: make(map[{{.Type}}]struct{})}
}

func (s *{{.Name}}Set) Add(item {{.Type}}) {
    s.items[item] = struct{}{}
}

func (s *{{.Name}}Set) Contains(item {{.Type}}) bool {
    _, ok := s.items[item]
    return ok
}

func (s *{{.Name}}Set) Len() int {
    return len(s.items)
}
`

type SetConfig struct {
    Package string
    Name    string
    Type    string
}

func main() {
    configs := []SetConfig{
        {"mypkg", "String", "string"},
        {"mypkg", "Int", "int"},
    }

    t := template.Must(template.New("set").Parse(tmpl))

    for _, cfg := range configs {
        filename := fmt.Sprintf("%s_set_gen.go", strings.ToLower(cfg.Name))
        f, _ := os.Create(filename)
        t.Execute(f, cfg)
        f.Close()
    }
}
```

### 5.3 Using go generate in Practice

```bash
# Add to source file:
# //go:generate go run gen/main.go

# Run all generators
go generate ./...

# Common generators:
# stringer — String() methods for enums
# mockgen — mock implementations for interfaces
# protoc-gen-go — Protocol Buffer code
# sqlc — type-safe SQL code
# ent — entity framework code
```

---

## 6. Practical Applications

### 6.1 Generic Struct Validator

```go
func Validate(v any) []string {
    val := reflect.ValueOf(v)
    typ := val.Type()
    var errors []string

    for i := 0; i < typ.NumField(); i++ {
        field := typ.Field(i)
        fieldVal := val.Field(i)

        if tag := field.Tag.Get("validate"); tag != "" {
            rules := strings.Split(tag, ",")
            for _, rule := range rules {
                switch {
                case rule == "required":
                    if fieldVal.IsZero() {
                        errors = append(errors, field.Name+" is required")
                    }
                case strings.HasPrefix(rule, "min="):
                    min, _ := strconv.Atoi(rule[4:])
                    if fieldVal.Kind() == reflect.String && len(fieldVal.String()) < min {
                        errors = append(errors, fmt.Sprintf("%s must be at least %d characters", field.Name, min))
                    }
                case strings.HasPrefix(rule, "max="):
                    max, _ := strconv.Atoi(rule[4:])
                    if fieldVal.Kind() == reflect.String && len(fieldVal.String()) > max {
                        errors = append(errors, fmt.Sprintf("%s must be at most %d characters", field.Name, max))
                    }
                }
            }
        }
    }
    return errors
}

type CreateUserInput struct {
    Name  string `validate:"required,min=2,max=100"`
    Email string `validate:"required"`
    Age   int    `validate:"required"`
}
```

### 6.2 Performance Considerations

```go
// Reflection is ~100x slower than direct access
// BenchmarkDirect-8     1000000000    0.25 ns/op
// BenchmarkReflect-8      20000000   25.0 ns/op

// Tips for performance-sensitive code:
// 1. Cache reflect.Type results
// 2. Use code generation instead of runtime reflection
// 3. Avoid reflect in hot paths
// 4. Use type switches instead of reflect when possible
```

---

## 7. Summary

### Key Takeaways

1. **`reflect.TypeOf` and `reflect.ValueOf`** — the entry points to reflection.
2. **Kind vs Type** — Kind is the underlying type category; Type is the specific named type.
3. **Struct tags are metadata** — accessible via `field.Tag.Get("key")`.
4. **Setting values requires pointers** — `reflect.ValueOf(&x).Elem().SetInt(42)`.
5. **Code generation over reflection** — prefer `go generate` for type-safe, fast code.
6. **Reflection is slow** — 10-100x overhead. Cache results and avoid hot paths.
7. **`//go:generate` directive** — run code generators as part of the build process.

---

## Exercises

### Exercise 1: Struct Diff
Write a function that compares two structs of the same type and returns a list of changed fields with old and new values.

### Exercise 2: Config Loader
Build a config loader that uses struct tags (`env`, `default`, `required`) to populate config structs from environment variables.

### Exercise 3: Code Generator
Create a code generator that reads a Go interface definition and generates a mock implementation.

### Exercise 4: ORM-Lite
Build a simple ORM that maps structs to SQL queries using struct tags for table/column names.
