// 17_reflection.go — Reflection and struct tags
//
// Run: go run 17_reflection.go

package main

import (
	"fmt"
	"reflect"
	"strings"
)

type User struct {
	ID    int    `json:"id" validate:"required"`
	Name  string `json:"name" validate:"required,min=2"`
	Email string `json:"email" validate:"required,email"`
	Age   int    `json:"age" validate:"min=0,max=150"`
}

func inspectStruct(v any) {
	t := reflect.TypeOf(v)
	val := reflect.ValueOf(v)

	fmt.Printf("Type: %s (%d fields)\n", t.Name(), t.NumField())
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		value := val.Field(i)
		jsonTag := field.Tag.Get("json")
		validateTag := field.Tag.Get("validate")
		fmt.Printf("  %-10s %-8s value=%-15v json=%s validate=%s\n",
			field.Name, field.Type, value.Interface(), jsonTag, validateTag)
	}
}

func getJSONFields(v any) []string {
	t := reflect.TypeOf(v)
	var names []string
	for i := 0; i < t.NumField(); i++ {
		tag := t.Field(i).Tag.Get("json")
		if tag != "" && tag != "-" {
			names = append(names, strings.Split(tag, ",")[0])
		}
	}
	return names
}

func structToMap(v any) map[string]any {
	result := make(map[string]any)
	val := reflect.ValueOf(v)
	typ := val.Type()
	for i := 0; i < typ.NumField(); i++ {
		field := typ.Field(i)
		if field.IsExported() {
			result[field.Name] = val.Field(i).Interface()
		}
	}
	return result
}

func main() {
	u := User{ID: 1, Name: "Alice", Email: "alice@example.com", Age: 30}

	fmt.Println("=== Struct Inspection ===")
	inspectStruct(u)

	fmt.Println("\n=== JSON Field Names ===")
	fmt.Println(getJSONFields(u))

	fmt.Println("\n=== Struct to Map ===")
	m := structToMap(u)
	for k, v := range m {
		fmt.Printf("  %s: %v\n", k, v)
	}
}
