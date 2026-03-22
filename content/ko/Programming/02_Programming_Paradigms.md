# 프로그래밍 패러다임

> **토픽**: Programming
> **레슨**: 2 of 16
> **선수 지식**: What Is Programming
> **목표**: 다양한 프로그래밍 패러다임, 그 원칙, 각 접근법을 사용하는 시기를 이해합니다.

---

## 프로그래밍 패러다임이란 무엇인가?

**프로그래밍 패러다임(Programming Paradigm)**은 프로그래밍의 기본적인 스타일이나 접근법입니다. 다음을 정의합니다:
- 코드를 구조화하는 방법
- 문제에 대해 생각하는 방법
- 사용하는 개념과 추상화

패러다임을 코드로 문제를 해결하는 다양한 **철학**으로 생각하세요.

**비유**: 건축 양식(고딕, 모더니즘, 브루탈리즘)이 다양하듯이, 프로그래밍 스타일도 다양합니다. 각각은 장점, 절충안, 적절한 사용 사례가 있습니다.

---

## 명령형 프로그래밍(Imperative Programming)

### 핵심 아이디어

컴퓨터에게 **어떻게(HOW)** 하는지를 단계별로 알려줍니다. 명시적인 명령을 통해 **상태 변경**에 집중합니다.

### 특징

- 명시적인 문장 시퀀스
- 시간에 따라 변하는 변수 (가변 상태)
- 제어 흐름: 루프, 조건문
- 생각: "이것을 하고, 그 다음 저것을 하고, 그 다음 다른 것을 해라"

### 예시: 배열 요소 합계

**Python:**
```python
def sum_array(numbers):
    total = 0  # Initial state
    for num in numbers:
        total = total + num  # Mutate state
    return total

result = sum_array([1, 2, 3, 4, 5])
print(result)  # Output: 15
```

**JavaScript:**
```javascript
function sumArray(numbers) {
    let total = 0;  // Initial state
    for (let i = 0; i < numbers.length; i++) {
        total = total + numbers[i];  // Mutate state
    }
    return total;
}

console.log(sumArray([1, 2, 3, 4, 5]));  // Output: 15
```

**Java:**
```java
public static int sumArray(int[] numbers) {
    int total = 0;  // Initial state
    for (int num : numbers) {
        total = total + num;  // Mutate state
    }
    return total;
}
```

**C++:**
```cpp
int sumArray(const std::vector<int>& numbers) {
    int total = 0;  // Initial state
    for (int num : numbers) {
        total = total + num;  // Mutate state
    }
    return total;
}
```

**초점**: `total` 변수를 명시적으로 관리하며, 각 반복마다 업데이트합니다.

---

## 절차적 프로그래밍(Procedural Programming)

### 핵심 아이디어

명령형 코드를 **프로시저(procedure)**(함수나 서브루틴이라고도 함)로 구성합니다. 프로그램을 서로를 호출하는 프로시저의 모음으로 구조화합니다.

### 특징

- 하향식 설계: 문제를 프로시저로 나눔
- 각 프로시저는 특정 작업 수행
- 프로시저는 다른 프로시저를 호출할 수 있음
- 데이터와 프로시저는 분리됨
- **모듈성(modularity)**과 **재사용성(reusability)** 강조

### 예시: 함수로 구조화된 프로그램

**Python:**
```python
def get_numbers():
    """Get numbers from user input"""
    return [1, 2, 3, 4, 5]

def calculate_sum(numbers):
    """Calculate sum of numbers"""
    total = 0
    for num in numbers:
        total += num
    return total

def calculate_average(numbers):
    """Calculate average of numbers"""
    return calculate_sum(numbers) / len(numbers)

def display_results(numbers):
    """Display results"""
    total = calculate_sum(numbers)
    avg = calculate_average(numbers)
    print(f"Sum: {total}")
    print(f"Average: {avg}")

def main():
    """Main procedure that orchestrates the program"""
    numbers = get_numbers()
    display_results(numbers)

# Entry point
main()
```

**C:**
```c
#include <stdio.h>

// Procedures as separate functions
int calculate_sum(int numbers[], int size) {
    int total = 0;
    for (int i = 0; i < size; i++) {
        total += numbers[i];
    }
    return total;
}

double calculate_average(int numbers[], int size) {
    return (double)calculate_sum(numbers, size) / size;
}

void display_results(int numbers[], int size) {
    int total = calculate_sum(numbers, size);
    double avg = calculate_average(numbers, size);
    printf("Sum: %d\n", total);
    printf("Average: %.2f\n", avg);
}

int main() {
    int numbers[] = {1, 2, 3, 4, 5};
    int size = 5;
    display_results(numbers, size);
    return 0;
}
```

**이점**: 코드가 구조화되고, 재사용 가능하며, 테스트하기 쉽습니다. 각 프로시저는 단일 책임을 갖습니다.

---

## 객체지향 프로그래밍(Object-Oriented Programming, OOP)

### 핵심 아이디어

**객체(object)** — 데이터(속성)와 동작(메서드)의 묶음 — 중심으로 코드를 구성합니다. 현실 세계 개체를 상호작용하는 객체로 모델링합니다.

### 핵심 개념

1. **캡슐화(Encapsulation)**: 데이터와 메서드를 함께 묶음; 내부 세부사항 숨김
2. **상속(Inheritance)**: 기존 클래스를 기반으로 새 클래스 생성
3. **다형성(Polymorphism)**: 같은 인터페이스, 다른 구현
4. **추상화(Abstraction)**: 구현을 지정하지 않고 인터페이스 정의

### 예시: 은행 계좌 모델링

**Python:**
```python
class BankAccount:
    """Represents a bank account"""

    def __init__(self, owner, balance=0):
        self.owner = owner
        self.__balance = balance  # Private attribute (encapsulation)

    def deposit(self, amount):
        """Deposit money"""
        if amount > 0:
            self.__balance += amount
            return True
        return False

    def withdraw(self, amount):
        """Withdraw money"""
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            return True
        return False

    def get_balance(self):
        """Get current balance"""
        return self.__balance


class SavingsAccount(BankAccount):
    """Savings account with interest (inheritance)"""

    def __init__(self, owner, balance=0, interest_rate=0.02):
        super().__init__(owner, balance)
        self.interest_rate = interest_rate

    def apply_interest(self):
        """Apply interest to balance"""
        interest = self.get_balance() * self.interest_rate
        self.deposit(interest)


# Usage
account = SavingsAccount("Alice", 1000)
account.deposit(500)
account.apply_interest()
print(f"Balance: ${account.get_balance():.2f}")  # Output: Balance: $1530.00
```

**Java:**
```java
public class BankAccount {
    private String owner;
    private double balance;  // Encapsulation: private field

    public BankAccount(String owner, double balance) {
        this.owner = owner;
        this.balance = balance;
    }

    public boolean deposit(double amount) {
        if (amount > 0) {
            this.balance += amount;
            return true;
        }
        return false;
    }

    public boolean withdraw(double amount) {
        if (amount > 0 && amount <= this.balance) {
            this.balance -= amount;
            return true;
        }
        return false;
    }

    public double getBalance() {
        return this.balance;
    }
}

// Inheritance
public class SavingsAccount extends BankAccount {
    private double interestRate;

    public SavingsAccount(String owner, double balance, double interestRate) {
        super(owner, balance);
        this.interestRate = interestRate;
    }

    public void applyInterest() {
        double interest = getBalance() * interestRate;
        deposit(interest);
    }
}
```

**C++:**
```cpp
class BankAccount {
private:
    std::string owner;
    double balance;  // Encapsulation

public:
    BankAccount(const std::string& owner, double balance = 0)
        : owner(owner), balance(balance) {}

    bool deposit(double amount) {
        if (amount > 0) {
            balance += amount;
            return true;
        }
        return false;
    }

    bool withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }

    double getBalance() const {
        return balance;
    }
};

// Inheritance
class SavingsAccount : public BankAccount {
private:
    double interestRate;

public:
    SavingsAccount(const std::string& owner, double balance = 0, double interestRate = 0.02)
        : BankAccount(owner, balance), interestRate(interestRate) {}

    void applyInterest() {
        double interest = getBalance() * interestRate;
        deposit(interest);
    }
};
```

**이점**: 현실 세계 개체를 모델링하고, 상속을 통한 코드 재사용을 촉진하며, 복잡성을 캡슐화합니다.

---

## 함수형 프로그래밍(Functional Programming, FP)

### 핵심 아이디어

계산을 수학적 함수의 평가로 취급합니다. 상태 변경과 가변 데이터를 피합니다.

### 핵심 원칙

1. **순수 함수(Pure Functions)**: 같은 입력은 항상 같은 출력을 생성; 부작용 없음
2. **불변성(Immutability)**: 데이터는 생성 후 변경될 수 없음
3. **일급 함수(First-Class Functions)**: 함수는 값 (전달되고 반환될 수 있음)
4. **고차 함수(Higher-Order Functions)**: 다른 함수를 받거나 반환하는 함수
5. **선언적(Declarative)**: **무엇을(WHAT)** 계산할지 표현, **어떻게(HOW)**가 아닌

### 예시: 명령형 vs 함수형

**명령형 접근법 (Python):**
```python
# Mutable state, explicit loops
def get_even_squares(numbers):
    result = []
    for num in numbers:
        if num % 2 == 0:
            result.append(num ** 2)
    return result

print(get_even_squares([1, 2, 3, 4, 5, 6]))  # [4, 16, 36]
```

**함수형 접근법 (Python):**
```python
# Immutable, declarative, higher-order functions
def get_even_squares(numbers):
    return list(
        map(lambda x: x ** 2,
            filter(lambda x: x % 2 == 0, numbers))
    )

# Or with list comprehension (more Pythonic)
def get_even_squares(numbers):
    return [x ** 2 for x in numbers if x % 2 == 0]

print(get_even_squares([1, 2, 3, 4, 5, 6]))  # [4, 16, 36]
```

**JavaScript (함수형 스타일):**
```javascript
const getEvenSquares = (numbers) =>
    numbers
        .filter(x => x % 2 === 0)  // Keep even numbers
        .map(x => x ** 2);         // Square them

console.log(getEvenSquares([1, 2, 3, 4, 5, 6]));  // [4, 16, 36]
```

### 순수 함수

**불순한 함수 (부작용 있음):**
```python
total = 0  # External state

def add_to_total(x):
    global total
    total += x  # Modifies external state (side effect)
    return total
```

**순수 함수 (부작용 없음):**
```python
def add(x, y):
    return x + y  # Only depends on inputs, no external state
```

### FP의 이점

- **예측 가능**: 순수 함수는 같은 입력에 대해 항상 같은 출력 반환
- **테스트 가능**: 테스트하기 쉬움 (숨겨진 상태 없음)
- **병렬화 가능**: 공유 상태 없어 안전한 동시성
- **조합 가능**: 작은 함수를 결합해 복잡한 동작 구축

---

## 선언형 프로그래밍(Declarative Programming)

### 핵심 아이디어

**무엇을(WHAT)** 원하는지 설명하고, **어떻게(HOW)** 달성할지는 설명하지 않습니다. 시스템이 구현을 파악합니다.

### 예시

**SQL (선언적 쿼리):**
```sql
-- You describe WHAT you want
SELECT name, age
FROM users
WHERE age > 18
ORDER BY name;

-- You don't specify HOW to:
-- - Scan the table
-- - Apply the filter
-- - Sort the results
-- The database engine decides the execution plan
```

**HTML (선언적 구조):**
```html
<!-- You describe WHAT the page should look like -->
<html>
  <body>
    <h1>Welcome</h1>
    <p>This is a paragraph.</p>
  </body>
</html>

<!-- You don't specify HOW to:
     - Render pixels
     - Apply default styles
     - Handle layout
     The browser does that
-->
```

**CSS (선언적 스타일링):**
```css
/* Describe WHAT styles you want */
.button {
    background-color: blue;
    color: white;
    padding: 10px;
}

/* Browser handles HOW to apply styles */
```

**명령형과 대비 (JavaScript):**
```javascript
// Imperative: HOW to create and style a button
const button = document.createElement('button');
button.style.backgroundColor = 'blue';
button.style.color = 'white';
button.style.padding = '10px';
button.textContent = 'Click me';
document.body.appendChild(button);
```

---

## 논리 프로그래밍(Logic Programming)

### 핵심 아이디어

논리를 사실과 규칙으로 표현합니다. 시스템이 논리적 추론을 수행하여 쿼리에 답합니다.

### 예시: Prolog

```prolog
% Facts
parent(tom, bob).
parent(tom, liz).
parent(bob, ann).
parent(bob, pat).
parent(pat, jim).

% Rules
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

% Queries
?- grandparent(tom, ann).  % true
?- ancestor(tom, jim).     % true
```

**무엇이 참인지** 정의하면, Prolog가 백트래킹과 단일화를 사용해 **어떻게 답을 찾을지** 파악합니다.

**사용 사례**: 전문가 시스템, 자연어 처리, 정리 증명.

---

## 이벤트 주도 프로그래밍(Event-Driven Programming)

### 핵심 아이디어

프로그램 흐름은 **이벤트**(사용자 액션, 메시지, 센서 입력)에 의해 결정됩니다. 시스템은 콜백이나 이벤트 핸들러를 통해 이벤트에 응답합니다.

### 특징

- **이벤트 루프(Event loop)**: 지속적으로 이벤트 확인
- **콜백/핸들러(Callbacks/Handlers)**: 이벤트에 대한 응답으로 실행되는 함수
- **비동기(Asynchronous)**: 이벤트는 언제든 발생 가능

### 예시: GUI 프로그래밍

**JavaScript (브라우저):**
```javascript
// Event handler
function handleClick(event) {
    console.log('Button clicked!');
    console.log('Mouse position:', event.clientX, event.clientY);
}

// Register event listener
const button = document.getElementById('myButton');
button.addEventListener('click', handleClick);

// The event loop waits for events and calls handlers
```

**Python (tkinter 사용):**
```python
import tkinter as tk

def on_button_click():
    print("Button clicked!")

# Create window
root = tk.Tk()

# Create button with event handler
button = tk.Button(root, text="Click Me", command=on_button_click)
button.pack()

# Start event loop
root.mainloop()  # Waits for events (clicks, key presses, etc.)
```

**사용 사례**: GUI, 웹 애플리케이션, 게임 엔진, IoT 시스템.

---

## 반응형 프로그래밍(Reactive Programming)

### 핵심 아이디어

**비동기 데이터 스트림**으로 프로그래밍합니다. 시간에 걸친 변화에 반응합니다.

### 특징

- **Observables**: 시간에 걸친 데이터/이벤트 스트림
- **Observers**: Observables를 구독하고 방출에 반응
- **Operators**: 스트림 변환, 필터링, 결합

### 예시: RxJS (JavaScript)

```javascript
import { fromEvent, interval } from 'rxjs';
import { map, filter, debounceTime } from 'rxjs/operators';

// Stream of click events
const clicks$ = fromEvent(document, 'click');

// Transform stream: get click positions
clicks$
    .pipe(
        map(event => ({ x: event.clientX, y: event.clientY }))
    )
    .subscribe(pos => console.log('Clicked at:', pos));

// Stream of time ticks
const ticks$ = interval(1000);  // Every second

ticks$
    .pipe(
        filter(n => n % 2 === 0)  // Only even numbers
    )
    .subscribe(n => console.log('Even tick:', n));

// Search input with debounce
const searchInput = document.getElementById('search');
const search$ = fromEvent(searchInput, 'input');

search$
    .pipe(
        map(event => event.target.value),
        debounceTime(300)  // Wait for 300ms pause
    )
    .subscribe(query => console.log('Search:', query));
```

**사용 사례**: 실시간 데이터 (주가 시세, 채팅), 사용자 입력 처리, 복잡한 비동기 워크플로우.

---

## 다중 패러다임 언어(Multi-Paradigm Languages)

대부분의 현대 언어는 **여러 패러다임**을 지원하여, 각 문제에 최적의 접근법을 선택할 수 있는 유연성을 제공합니다.

### Python: 명령형, OOP, 함수형

```python
# Imperative
total = 0
for i in range(10):
    total += i

# Object-Oriented
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

# Functional
from functools import reduce
total = reduce(lambda acc, x: acc + x, range(10), 0)
```

### JavaScript: 명령형, OOP, 함수형, 이벤트 주도

```javascript
// Imperative
let total = 0;
for (let i = 0; i < 10; i++) {
    total += i;
}

// Object-Oriented (class syntax)
class Counter {
    constructor() {
        this.count = 0;
    }
    increment() {
        this.count++;
    }
}

// Functional
const total = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    .reduce((acc, x) => acc + x, 0);

// Event-Driven
button.addEventListener('click', () => console.log('Clicked'));
```

### Scala: OOP, 함수형

```scala
// Object-Oriented
class BankAccount(var balance: Double) {
    def deposit(amount: Double): Unit = {
        balance += amount
    }
}

// Functional
val numbers = List(1, 2, 3, 4, 5)
val doubled = numbers.map(_ * 2)
val sum = numbers.reduce(_ + _)
```

---

## 패러다임 비교: 언제 어떤 것을 사용할까?

### 명령형/절차적

**언제:**
- 성능이 중요한 코드
- 저수준 시스템 프로그래밍
- 자연스럽게 상태 변경을 포함하는 알고리즘

**예시**: 디바이스 드라이버, 게임 엔진, 임베디드 시스템

### 객체지향

**언제:**
- 많은 개체가 있는 복잡한 도메인 모델링
- 상속을 통한 코드 재사용
- 대규모 팀과 코드베이스 (캡슐화가 복잡성 관리에 도움)

**예시**: 엔터프라이즈 애플리케이션, GUI 프레임워크, 시뮬레이션

### 함수형

**언제:**
- 데이터 변환
- 동시/병렬 처리 (공유 상태 없음)
- 수학적 계산
- 예측 가능하고 테스트 가능한 코드

**예시**: 데이터 파이프라인, 스트림 처리, 컴파일러, 금융 시스템

### 선언형

**언제:**
- 구현 세부사항 없이 의도를 표현하고 싶을 때
- 프레임워크/라이브러리가 효율적인 구현을 제공할 때

**예시**: 데이터베이스 쿼리 (SQL), UI 마크업 (HTML/CSS), 설정 (YAML)

### 이벤트 주도

**언제:**
- 사용자 인터페이스
- 비동기 I/O
- 실시간 시스템

**예시**: 웹 앱, 모바일 앱, IoT, 네트워크 서버

---

## 절충안

| 패러다임       | 장점                                      | 단점                                   |
|----------------|------------------------------------------------|----------------------------------------------|
| 명령형     | 명시적 제어, 성능                  | 장황함, 상태에 대한 추론 어려움        |
| OOP            | 현실 세계 모델링, 캡슐화, 재사용        | 과도한 설계 가능, 상속 문제   |
| 함수형     | 예측 가능, 테스트 가능, 동시성 안전        | 학습 곡선, 성능 오버헤드 (불변성) |
| 선언형    | 간결함, 명확한 의도                          | 실행에 대한 제어 적음                  |
| 이벤트 주도   | 반응성, 비동기                       | 디버깅 어려움, "콜백 지옥"        |

---

## 연습 문제

### 연습 문제 1: 패러다임 인식

각 코드 스니펫에서 사용된 패러다임을 식별하세요:

**A)**
```python
numbers = [1, 2, 3, 4, 5]
result = sum(filter(lambda x: x % 2 == 0, numbers))
```

**B)**
```java
public class Car {
    private int speed;
    public void accelerate() {
        speed += 10;
    }
}
```

**C)**
```javascript
document.getElementById('btn').addEventListener('click', function() {
    alert('Clicked!');
});
```

### 연습 문제 2: 명령형에서 함수형으로

이 명령형 코드를 함수형 스타일로 다시 작성하세요:

```python
def process_data(numbers):
    result = []
    for num in numbers:
        if num > 0:
            result.append(num * 2)
    return result
```

### 연습 문제 3: OOP 설계

간단한 도서관 시스템을 위한 클래스 설계:
- `Book`: 제목, 저자, ISBN
- `Member`: 이름, 회원 ID, 대출한 책
- `Library`: 책 컬렉션, 회원, 대출/반납 메서드

선택한 언어로 구현하세요. 캡슐화를 적용하고, 상속을 고려하세요 (예: `EBook`이 `Book`을 확장?).

### 연습 문제 4: 함수형 vs OOP

다음 문제를 OOP와 함수형 접근법 모두로 해결하세요:

**문제**: 숫자 리스트에 대한 통계 (합계, 평균, 최댓값, 최솟값) 계산

- **OOP**: `Statistics` 클래스 생성
- **함수형**: 순수 함수와 고차 함수 사용

어떤 접근법이 더 자연스럽게 느껴지나요? 왜 그런가요?

### 연습 문제 5: 다중 패러다임

다음을 수행하는 프로그램 작성:
1. `Person` 클래스 정의 (OOP)
2. `Person` 객체 리스트 사용
3. 함수형 프로그래밍을 사용해 성인 (나이 >= 18) 필터링
4. 내장 함수를 사용해 이름으로 정렬
5. 루프로 결과를 명령형으로 출력

### 연습 문제 6: 이벤트 주도

간단한 이벤트 주도 프로그램 작성 (웹 또는 GUI):
- 텍스트 입력과 버튼 생성
- 버튼 클릭 시, 입력 텍스트를 alert/label에 표시
- (보너스) 각 클릭마다 증가하는 카운터 추가

---

## 요약

프로그래밍 패러다임은 코드에 대한 다양한 **사고 방식**입니다:

- **명령형**: 단계별 명령, 가변 상태
- **절차적**: 명령형 코드를 함수로 구성
- **객체지향**: 객체로 모델링, 캡슐화, 상속
- **함수형**: 순수 함수, 불변성, 조합
- **선언형**: 무엇을(WHAT) 설명, 어떻게(HOW)가 아닌
- **논리**: 사실, 규칙, 쿼리, 추론
- **이벤트 주도**: 비동기적으로 이벤트에 응답
- **반응형**: 데이터 스트림으로 프로그래밍

**핵심 통찰**: 어떤 패러다임도 "최고"가 아닙니다 — 문제, 팀, 생태계에 따라 선택하세요. 현대 언어는 여러 패러다임을 지원하여 유연성을 제공합니다.

---

## 탐색

[← 이전: What Is Programming](01_What_Is_Programming.md) | [다음: Data Types & Abstraction →](03_Data_Types_and_Abstraction.md)
