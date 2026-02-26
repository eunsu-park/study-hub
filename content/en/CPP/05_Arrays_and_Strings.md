# Arrays and Strings

**Previous**: [Functions](./04_Functions.md) | **Next**: [Pointers and References](./06_Pointers_and_References.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Declare, initialize, and access elements in C-style arrays and multidimensional arrays
2. Compare C-style arrays with `std::array` and explain the safety advantages of bounds-checked access
3. Distinguish between C-style strings (`char[]`) and `std::string`, and identify when each is appropriate
4. Apply core `std::string` operations: concatenation, searching, substring extraction, insertion, and replacement
5. Implement string-to-number and number-to-string conversions using `std::stoi`, `std::stod`, and `std::to_string`
6. Design string-splitting logic using `std::stringstream`
7. Explain how `std::string_view` (C++17) avoids unnecessary copies when reading string data

---

Arrays and strings are the workhorses of data storage. Nearly every real-world program -- from a simple log parser to a high-frequency trading system -- spends much of its time creating, searching, and transforming sequences of values or characters. Knowing which container to reach for (raw array, `std::array`, `std::string`, or `std::string_view`) and understanding the performance trade-offs will pay dividends throughout your C++ career.

## 1. Array Basics

Arrays store multiple values of the same type in contiguous memory.

### Array Declaration and Initialization

```cpp
#include <iostream>

int main() {
    // Size specified
    int arr1[5];  // Uninitialized (garbage values)

    // Initializer list
    int arr2[5] = {1, 2, 3, 4, 5};

    // Partial initialization (rest are 0)
    int arr3[5] = {1, 2};  // {1, 2, 0, 0, 0}

    // Initialize all to 0
    int arr4[5] = {};  // {0, 0, 0, 0, 0}

    // Size auto-determined
    int arr5[] = {1, 2, 3};  // Size 3

    // Output
    for (int i = 0; i < 5; i++) {
        std::cout << arr2[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### Array Access

```cpp
#include <iostream>

int main() {
    int arr[5] = {10, 20, 30, 40, 50};

    // Read
    std::cout << "First: " << arr[0] << std::endl;  // 10
    std::cout << "Third: " << arr[2] << std::endl;  // 30

    // Write
    arr[1] = 200;
    std::cout << "After modification: " << arr[1] << std::endl;  // 200

    // Range-based for
    for (int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### Array Size

```cpp
#include <iostream>

int main() {
    int arr[5] = {1, 2, 3, 4, 5};

    // Calculate size with sizeof
    int size = sizeof(arr) / sizeof(arr[0]);
    std::cout << "Array size: " << size << std::endl;  // 5

    // C++17: std::size
    // #include <iterator>
    // std::cout << std::size(arr) << std::endl;

    return 0;
}
```

---

## 2. Multidimensional Arrays

### 2D Arrays

```cpp
#include <iostream>

int main() {
    // 3 rows, 4 columns
    int matrix[3][4] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };

    // Access
    std::cout << "matrix[1][2] = " << matrix[1][2] << std::endl;  // 7

    // Print all
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << matrix[i][j] << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}
```

### 3D Arrays

```cpp
#include <iostream>

int main() {
    int cube[2][3][4] = {
        {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12}
        },
        {
            {13, 14, 15, 16},
            {17, 18, 19, 20},
            {21, 22, 23, 24}
        }
    };

    std::cout << "cube[1][2][3] = " << cube[1][2][3] << std::endl;  // 24

    return 0;
}
```

---

## 3. std::array (C++11)

A safe, fixed-size array.

```cpp
#include <iostream>
#include <array>

int main() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};

    // Size
    std::cout << "Size: " << arr.size() << std::endl;

    // Access
    std::cout << "First: " << arr[0] << std::endl;
    std::cout << "Last: " << arr.back() << std::endl;

    // Bounds-checked access
    std::cout << "arr.at(2): " << arr.at(2) << std::endl;
    // arr.at(10);  // Throws exception!

    // Range-based for
    for (int num : arr) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    // Fill
    arr.fill(0);

    return 0;
}
```

### Array vs std::array

| Feature | C Array | std::array |
|---------|---------|------------|
| Size check | sizeof calculation | .size() |
| Bounds checking | None | .at() |
| Copy | Not allowed | Allowed |
| Function passing | Decays to pointer | Value/reference passing |

---

## 4. C-Style Strings

Represents strings as character arrays.

```cpp
#include <iostream>
#include <cstring>  // strlen, strcpy, etc.

int main() {
    // String literal
    char str1[] = "Hello";  // {'H', 'e', 'l', 'l', 'o', '\0'}
    char str2[10] = "World";

    // Length
    std::cout << "Length: " << strlen(str1) << std::endl;  // 5

    // Output
    std::cout << str1 << std::endl;

    // Character-by-character access
    for (int i = 0; str1[i] != '\0'; i++) {
        std::cout << str1[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### C String Functions

```cpp
#include <iostream>
#include <cstring>

int main() {
    char str1[20] = "Hello";
    char str2[20] = "World";
    char dest[40];

    // Copy
    strcpy(dest, str1);
    std::cout << "strcpy: " << dest << std::endl;  // Hello

    // Concatenate
    strcat(dest, " ");
    strcat(dest, str2);
    std::cout << "strcat: " << dest << std::endl;  // Hello World

    // Compare
    if (strcmp(str1, str2) < 0) {
        std::cout << str1 << " < " << str2 << std::endl;
    }

    // Find
    char* pos = strstr(dest, "World");
    if (pos != nullptr) {
        std::cout << "Found: " << pos << std::endl;  // World
    }

    return 0;
}
```

---

## 5. std::string

C++ string class.

### Basic Usage

```cpp
#include <iostream>
#include <string>

int main() {
    // Creation
    std::string s1 = "Hello";
    std::string s2("World");
    std::string s3(5, 'x');  // "xxxxx"

    // Output
    std::cout << s1 << " " << s2 << std::endl;

    // Length
    std::cout << "Length: " << s1.length() << std::endl;  // 5
    std::cout << "Size: " << s1.size() << std::endl;      // 5 (same)

    // Empty string check
    std::string empty;
    std::cout << "Is empty: " << empty.empty() << std::endl;  // true

    return 0;
}
```

### String Operations

```cpp
#include <iostream>
#include <string>

int main() {
    std::string s1 = "Hello";
    std::string s2 = "World";

    // Concatenation
    std::string s3 = s1 + " " + s2;
    std::cout << s3 << std::endl;  // Hello World

    // += operator
    s1 += "!";
    std::cout << s1 << std::endl;  // Hello!

    // append
    s1.append(" C++");
    std::cout << s1 << std::endl;  // Hello! C++

    // Comparison
    if (s1 == "Hello! C++") {
        std::cout << "Equal" << std::endl;
    }

    if (s1 < s2) {  // Lexicographic comparison
        std::cout << s1 << " < " << s2 << std::endl;
    }

    return 0;
}
```

### String Access

```cpp
#include <iostream>
#include <string>

int main() {
    std::string str = "Hello";

    // Index access
    std::cout << "First character: " << str[0] << std::endl;  // H
    std::cout << "Last: " << str.back() << std::endl;  // o

    // Bounds-checked access
    std::cout << "at(1): " << str.at(1) << std::endl;  // e

    // Modification
    str[0] = 'h';
    std::cout << str << std::endl;  // hello

    // Range-based for
    for (char c : str) {
        std::cout << c << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

### Substring

```cpp
#include <iostream>
#include <string>

int main() {
    std::string str = "Hello, World!";

    // Extract substring
    std::string sub = str.substr(7, 5);  // 5 characters from position 7
    std::cout << sub << std::endl;  // World

    // From position to end
    std::string rest = str.substr(7);
    std::cout << rest << std::endl;  // World!

    return 0;
}
```

### Searching

```cpp
#include <iostream>
#include <string>

int main() {
    std::string str = "Hello, World!";

    // Find
    size_t pos = str.find("World");
    if (pos != std::string::npos) {
        std::cout << "Position: " << pos << std::endl;  // 7
    }

    // Find character
    pos = str.find('o');
    std::cout << "First o: " << pos << std::endl;  // 4

    // Find from end
    pos = str.rfind('o');
    std::cout << "Last o: " << pos << std::endl;  // 8

    // Not found case
    pos = str.find("xyz");
    if (pos == std::string::npos) {
        std::cout << "Not found" << std::endl;
    }

    return 0;
}
```

### Modification

```cpp
#include <iostream>
#include <string>

int main() {
    std::string str = "Hello, World!";

    // Insert
    str.insert(7, "Beautiful ");
    std::cout << str << std::endl;  // Hello, Beautiful World!

    // Erase
    str.erase(7, 10);  // Erase 10 characters from position 7
    std::cout << str << std::endl;  // Hello, World!

    // Replace
    str.replace(7, 5, "C++");  // Replace World with C++
    std::cout << str << std::endl;  // Hello, C++!

    // Clear
    str.clear();
    std::cout << "Is empty: " << str.empty() << std::endl;

    return 0;
}
```

---

## 6. String Conversion

### Number <-> String

```cpp
#include <iostream>
#include <string>

int main() {
    // Number -> String
    int num = 42;
    std::string str1 = std::to_string(num);
    std::cout << "to_string: " << str1 << std::endl;

    double pi = 3.14159;
    std::string str2 = std::to_string(pi);
    std::cout << "to_string: " << str2 << std::endl;

    // String -> Number
    std::string s1 = "123";
    int n1 = std::stoi(s1);
    std::cout << "stoi: " << n1 << std::endl;

    std::string s2 = "3.14";
    double d1 = std::stod(s2);
    std::cout << "stod: " << d1 << std::endl;

    // Other conversion functions
    // std::stol - long
    // std::stoll - long long
    // std::stof - float

    return 0;
}
```

### Character Conversion

```cpp
#include <iostream>
#include <cctype>
#include <string>

int main() {
    char c = 'a';

    // Case conversion
    std::cout << "Uppercase: " << (char)std::toupper(c) << std::endl;  // A

    c = 'Z';
    std::cout << "Lowercase: " << (char)std::tolower(c) << std::endl;  // z

    // Character checking
    std::cout << std::boolalpha;
    std::cout << "isalpha('A'): " << (bool)std::isalpha('A') << std::endl;  // true
    std::cout << "isdigit('5'): " << (bool)std::isdigit('5') << std::endl;  // true
    std::cout << "isspace(' '): " << (bool)std::isspace(' ') << std::endl;  // true

    // Convert entire string to uppercase
    std::string str = "Hello World";
    for (char& c : str) {
        c = std::toupper(c);
    }
    std::cout << str << std::endl;  // HELLO WORLD

    return 0;
}
```

---

## 7. String Input

```cpp
#include <iostream>
#include <string>

int main() {
    std::string word;
    std::string line;

    // Word input (until whitespace)
    std::cout << "Enter a word: ";
    std::cin >> word;
    std::cout << "Input: " << word << std::endl;

    // Clear buffer
    std::cin.ignore();

    // Entire line input
    std::cout << "Enter a sentence: ";
    std::getline(std::cin, line);
    std::cout << "Input: " << line << std::endl;

    return 0;
}
```

---

## 8. String Splitting

```cpp
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

int main() {
    std::string str = "apple,banana,cherry,date";

    // Using stringstream
    std::stringstream ss(str);
    std::string token;
    std::vector<std::string> tokens;

    while (std::getline(ss, token, ',')) {
        tokens.push_back(token);
    }

    for (const auto& t : tokens) {
        std::cout << t << std::endl;
    }

    return 0;
}
```

---

## 9. string_view (C++17)

References a string without copying.

```cpp
#include <iostream>
#include <string>
#include <string_view>

void print(std::string_view sv) {
    std::cout << sv << std::endl;
}

int main() {
    std::string str = "Hello, World!";
    const char* cstr = "C-style string";

    // Can accept various string types
    print(str);
    print(cstr);
    print("Literal");

    // Substring without copy
    std::string_view sv = str;
    std::cout << sv.substr(0, 5) << std::endl;  // Hello

    return 0;
}
```

---

## 10. Practice Examples

### String Reversal

```cpp
#include <iostream>
#include <string>
#include <algorithm>

int main() {
    std::string str = "Hello";

    // Method 1: reverse function
    std::reverse(str.begin(), str.end());
    std::cout << str << std::endl;  // olleH

    // Method 2: Manual implementation
    str = "World";
    int len = str.length();
    for (int i = 0; i < len / 2; i++) {
        std::swap(str[i], str[len - 1 - i]);
    }
    std::cout << str << std::endl;  // dlroW

    return 0;
}
```

### Palindrome Check

```cpp
#include <iostream>
#include <string>
#include <algorithm>

bool isPalindrome(const std::string& str) {
    std::string reversed = str;
    std::reverse(reversed.begin(), reversed.end());
    return str == reversed;
}

int main() {
    std::cout << std::boolalpha;
    std::cout << isPalindrome("radar") << std::endl;  // true
    std::cout << isPalindrome("hello") << std::endl;  // false
    return 0;
}
```

### Word Count

```cpp
#include <iostream>
#include <string>
#include <sstream>

int countWords(const std::string& str) {
    std::stringstream ss(str);
    std::string word;
    int count = 0;

    while (ss >> word) {
        count++;
    }

    return count;
}

int main() {
    std::string text = "Hello World this is C++";
    std::cout << "Word count: " << countWords(text) << std::endl;  // 5
    return 0;
}
```

---

## 11. Summary

| Type | Features |
|------|----------|
| C array `T[]` | Fixed size, no bounds checking |
| `std::array<T, N>` | Fixed size, safe |
| C string `char[]` | Null-terminated, manual management |
| `std::string` | Dynamic size, automatic management |
| `std::string_view` | Read-only reference |

| std::string Methods | Description |
|---------------------|-------------|
| `length()`, `size()` | Length |
| `empty()` | Is empty |
| `substr(pos, len)` | Substring |
| `find(str)` | Search |
| `replace(pos, len, str)` | Replace |
| `insert(pos, str)` | Insert |
| `erase(pos, len)` | Erase |

---

## Next Step

Let's learn about pointers and references in [06_Pointers_and_References.md](./06_Pointers_and_References.md)!
