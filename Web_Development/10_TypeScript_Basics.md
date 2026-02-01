# 10. TypeScript 기초 (TypeScript Fundamentals)

## 학습 목표
- TypeScript의 장점과 JavaScript와의 관계 이해
- 기본 타입 시스템 마스터
- 인터페이스와 타입 별칭 활용
- 제네릭을 통한 재사용 가능한 코드 작성
- 유틸리티 타입과 고급 타입 기능 이해

## 목차
1. [TypeScript 소개](#1-typescript-소개)
2. [기본 타입](#2-기본-타입)
3. [인터페이스와 타입](#3-인터페이스와-타입)
4. [함수 타입](#4-함수-타입)
5. [제네릭](#5-제네릭)
6. [유틸리티 타입](#6-유틸리티-타입)
7. [연습 문제](#7-연습-문제)

---

## 1. TypeScript 소개

### 1.1 TypeScript란?

```
┌─────────────────────────────────────────────────────────────────┐
│                    TypeScript 개요                               │
│                                                                 │
│   TypeScript = JavaScript + 정적 타입                           │
│                                                                 │
│   특징:                                                         │
│   - Microsoft에서 개발                                          │
│   - JavaScript의 상위 집합 (Superset)                           │
│   - 컴파일 시 타입 검사                                          │
│   - 모든 JavaScript 코드는 유효한 TypeScript                    │
│                                                                 │
│   장점:                                                         │
│   - 런타임 전 오류 발견                                          │
│   - IDE 지원 향상 (자동완성, 리팩토링)                           │
│   - 코드 가독성 및 문서화                                        │
│   - 대규모 프로젝트 유지보수 용이                                │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 설치 및 설정

```bash
# TypeScript 설치
npm install -g typescript

# 버전 확인
tsc --version

# 프로젝트 초기화
npm init -y
npm install typescript --save-dev

# tsconfig.json 생성
npx tsc --init
```

```json
// tsconfig.json 기본 설정
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "declaration": true,
    "moduleResolution": "node"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### 1.3 컴파일과 실행

```bash
# 단일 파일 컴파일
tsc hello.ts

# 프로젝트 전체 컴파일
tsc

# 감시 모드 (파일 변경 시 자동 컴파일)
tsc --watch

# ts-node로 직접 실행 (개발용)
npm install -g ts-node
ts-node hello.ts
```

---

## 2. 기본 타입

### 2.1 원시 타입

```typescript
// 문자열
let name: string = "TypeScript";
let greeting: string = `Hello, ${name}!`;

// 숫자
let age: number = 25;
let price: number = 99.99;
let hex: number = 0xf00d;

// 불리언
let isActive: boolean = true;
let hasError: boolean = false;

// null과 undefined
let nothing: null = null;
let notDefined: undefined = undefined;

// BigInt (ES2020+)
let bigNumber: bigint = 9007199254740991n;

// Symbol
let sym: symbol = Symbol("unique");
```

### 2.2 배열과 튜플

```typescript
// 배열 타입 (두 가지 방식)
let numbers: number[] = [1, 2, 3, 4, 5];
let strings: Array<string> = ["a", "b", "c"];

// 다차원 배열
let matrix: number[][] = [
  [1, 2, 3],
  [4, 5, 6],
];

// 튜플 (고정 길이, 고정 타입 배열)
let tuple: [string, number] = ["Alice", 30];
let rgb: [number, number, number] = [255, 128, 0];

// 튜플 요소 접근
const [userName, userAge] = tuple;
console.log(userName); // "Alice"

// 명명된 튜플 (가독성 향상)
type Point = [x: number, y: number];
const point: Point = [10, 20];
```

### 2.3 객체 타입

```typescript
// 기본 객체 타입
let person: { name: string; age: number } = {
  name: "Bob",
  age: 25,
};

// 선택적 속성 (?)
let config: { host: string; port?: number } = {
  host: "localhost",
  // port는 선택적
};

// 읽기 전용 속성
let user: { readonly id: number; name: string } = {
  id: 1,
  name: "Alice",
};
// user.id = 2;  // 오류! readonly

// 인덱스 시그니처
let dictionary: { [key: string]: number } = {
  apple: 1,
  banana: 2,
};
```

### 2.4 특수 타입

```typescript
// any - 모든 타입 허용 (사용 자제)
let anything: any = "hello";
anything = 42;
anything = { foo: "bar" };

// unknown - any보다 안전한 대안
let unknownValue: unknown = "hello";
// unknownValue.toUpperCase();  // 오류!
if (typeof unknownValue === "string") {
  unknownValue.toUpperCase(); // OK - 타입 가드 후
}

// void - 반환값 없음
function logMessage(msg: string): void {
  console.log(msg);
}

// never - 절대 반환하지 않음
function throwError(message: string): never {
  throw new Error(message);
}

function infiniteLoop(): never {
  while (true) {}
}
```

### 2.5 Union과 Intersection

```typescript
// Union 타입 (|) - 여러 타입 중 하나
let id: string | number;
id = "abc";
id = 123;

type Status = "pending" | "approved" | "rejected";
let orderStatus: Status = "pending";

// Intersection 타입 (&) - 모든 타입 결합
type Name = { name: string };
type Age = { age: number };
type Person = Name & Age;

const person: Person = {
  name: "Alice",
  age: 30,
};
```

### 2.6 타입 추론과 타입 단언

```typescript
// 타입 추론 - TypeScript가 자동으로 타입 결정
let message = "Hello"; // string으로 추론
let count = 10; // number로 추론

// 타입 단언 (Type Assertion)
let someValue: unknown = "this is a string";

// 방법 1: as 문법 (권장)
let strLength1: number = (someValue as string).length;

// 방법 2: angle-bracket 문법 (JSX와 충돌)
let strLength2: number = (<string>someValue).length;

// const 단언
let colors = ["red", "green", "blue"] as const;
// readonly ["red", "green", "blue"] 타입

// Non-null 단언 (!)
function getLength(str: string | null): number {
  return str!.length; // null이 아님을 단언
}
```

---

## 3. 인터페이스와 타입

### 3.1 인터페이스 기본

```typescript
// 인터페이스 정의
interface User {
  id: number;
  name: string;
  email: string;
  age?: number; // 선택적
  readonly createdAt: Date; // 읽기 전용
}

// 인터페이스 사용
const user: User = {
  id: 1,
  name: "Alice",
  email: "alice@example.com",
  createdAt: new Date(),
};

// 함수 타입 인터페이스
interface Calculator {
  (a: number, b: number): number;
}

const add: Calculator = (a, b) => a + b;
```

### 3.2 인터페이스 확장

```typescript
// 인터페이스 상속
interface Animal {
  name: string;
  age: number;
}

interface Dog extends Animal {
  breed: string;
  bark(): void;
}

const myDog: Dog = {
  name: "Buddy",
  age: 3,
  breed: "Labrador",
  bark() {
    console.log("Woof!");
  },
};

// 다중 상속
interface Pet extends Animal {
  owner: string;
}

interface ServiceDog extends Dog, Pet {
  certificationId: string;
}
```

### 3.3 타입 별칭 (Type Alias)

```typescript
// 타입 별칭 정의
type ID = string | number;
type Point = { x: number; y: number };
type Callback = (data: string) => void;

// 사용
let userId: ID = "user_123";
let position: Point = { x: 10, y: 20 };

// 유니온 타입에 유용
type Result<T> = { success: true; data: T } | { success: false; error: string };

function fetchData(): Result<User> {
  return { success: true, data: { id: 1, name: "Alice", email: "a@b.com", createdAt: new Date() } };
}
```

### 3.4 인터페이스 vs 타입

```typescript
// 인터페이스 - 선언 병합 가능
interface Window {
  title: string;
}

interface Window {
  size: number; // 자동 병합됨
}

// 타입 - 병합 불가, 더 유연
type StringOrNumber = string | number; // 유니온
type Point = [number, number]; // 튜플

// 권장사항:
// - 객체 형태 정의: interface 사용
// - 유니온, 튜플, 원시 타입 별칭: type 사용
// - 라이브러리 API: interface (확장 가능)
```

---

## 4. 함수 타입

### 4.1 함수 타입 정의

```typescript
// 함수 선언
function add(a: number, b: number): number {
  return a + b;
}

// 화살표 함수
const multiply = (a: number, b: number): number => a * b;

// 함수 타입 별칭
type MathOperation = (a: number, b: number) => number;

const divide: MathOperation = (a, b) => a / b;

// 함수 타입 인터페이스
interface MathFunc {
  (a: number, b: number): number;
  description?: string;
}
```

### 4.2 매개변수 옵션

```typescript
// 선택적 매개변수 (?)
function greet(name: string, greeting?: string): string {
  return `${greeting || "Hello"}, ${name}!`;
}

// 기본값 매개변수
function greetWithDefault(name: string, greeting: string = "Hello"): string {
  return `${greeting}, ${name}!`;
}

// 나머지 매개변수
function sum(...numbers: number[]): number {
  return numbers.reduce((acc, n) => acc + n, 0);
}

console.log(sum(1, 2, 3, 4, 5)); // 15
```

### 4.3 함수 오버로딩

```typescript
// 함수 오버로딩 시그니처
function process(x: string): string;
function process(x: number): number;
function process(x: string | number): string | number {
  if (typeof x === "string") {
    return x.toUpperCase();
  }
  return x * 2;
}

console.log(process("hello")); // "HELLO"
console.log(process(5)); // 10
```

### 4.4 this 타입

```typescript
interface Button {
  label: string;
  click(this: Button): void;
}

const button: Button = {
  label: "Submit",
  click() {
    console.log(`Clicked: ${this.label}`);
  },
};

button.click(); // OK
// const handler = button.click;
// handler();  // 오류! this 컨텍스트 손실
```

---

## 5. 제네릭

### 5.1 제네릭 기본

```typescript
// 제네릭 함수
function identity<T>(arg: T): T {
  return arg;
}

// 사용
let output1 = identity<string>("hello");
let output2 = identity<number>(42);
let output3 = identity("auto"); // 타입 추론

// 제네릭 배열
function firstElement<T>(arr: T[]): T | undefined {
  return arr[0];
}

const first = firstElement([1, 2, 3]); // number | undefined
```

### 5.2 제네릭 인터페이스와 타입

```typescript
// 제네릭 인터페이스
interface Box<T> {
  value: T;
}

const stringBox: Box<string> = { value: "hello" };
const numberBox: Box<number> = { value: 42 };

// 제네릭 타입 별칭
type Result<T> = {
  success: boolean;
  data: T;
};

type Pair<K, V> = {
  key: K;
  value: V;
};

const pair: Pair<string, number> = { key: "age", value: 30 };
```

### 5.3 제네릭 제약조건

```typescript
// extends로 제약 추가
interface Lengthwise {
  length: number;
}

function logLength<T extends Lengthwise>(arg: T): T {
  console.log(arg.length);
  return arg;
}

logLength("hello"); // OK - string has length
logLength([1, 2, 3]); // OK - array has length
// logLength(123);    // 오류! number has no length

// keyof 제약조건
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

const person = { name: "Alice", age: 30 };
const name = getProperty(person, "name"); // string
const age = getProperty(person, "age"); // number
// getProperty(person, "email");  // 오류!
```

### 5.4 제네릭 클래스

```typescript
class Queue<T> {
  private items: T[] = [];

  enqueue(item: T): void {
    this.items.push(item);
  }

  dequeue(): T | undefined {
    return this.items.shift();
  }

  peek(): T | undefined {
    return this.items[0];
  }

  get length(): number {
    return this.items.length;
  }
}

const numberQueue = new Queue<number>();
numberQueue.enqueue(1);
numberQueue.enqueue(2);
console.log(numberQueue.dequeue()); // 1
```

---

## 6. 유틸리티 타입

### 6.1 기본 유틸리티 타입

```typescript
interface User {
  id: number;
  name: string;
  email: string;
  age?: number;
}

// Partial<T> - 모든 속성 선택적으로
type PartialUser = Partial<User>;
// { id?: number; name?: string; email?: string; age?: number }

// Required<T> - 모든 속성 필수로
type RequiredUser = Required<User>;
// { id: number; name: string; email: string; age: number }

// Readonly<T> - 모든 속성 읽기 전용
type ReadonlyUser = Readonly<User>;

// Pick<T, K> - 특정 속성만 선택
type UserBasic = Pick<User, "id" | "name">;
// { id: number; name: string }

// Omit<T, K> - 특정 속성 제외
type UserWithoutEmail = Omit<User, "email">;
// { id: number; name: string; age?: number }
```

### 6.2 레코드와 맵핑

```typescript
// Record<K, T> - 키-값 맵핑
type UserRole = "admin" | "user" | "guest";
type RolePermissions = Record<UserRole, string[]>;

const permissions: RolePermissions = {
  admin: ["read", "write", "delete"],
  user: ["read", "write"],
  guest: ["read"],
};

// 사용 예시
type PageInfo = {
  title: string;
  url: string;
};

type Pages = Record<"home" | "about" | "contact", PageInfo>;
```

### 6.3 조건부 타입

```typescript
// Exclude<T, U> - T에서 U 제외
type Numbers = 1 | 2 | 3 | 4 | 5;
type SmallNumbers = Exclude<Numbers, 4 | 5>; // 1 | 2 | 3

// Extract<T, U> - T와 U의 공통 타입
type Common = Extract<"a" | "b" | "c", "a" | "c" | "d">; // "a" | "c"

// NonNullable<T> - null, undefined 제외
type MaybeString = string | null | undefined;
type DefinitelyString = NonNullable<MaybeString>; // string

// ReturnType<T> - 함수 반환 타입
function getUser() {
  return { id: 1, name: "Alice" };
}
type UserReturn = ReturnType<typeof getUser>;
// { id: number; name: string }

// Parameters<T> - 함수 매개변수 타입
type UserParams = Parameters<typeof getUser>; // []
```

### 6.4 템플릿 리터럴 타입

```typescript
// 문자열 리터럴 조합
type Color = "red" | "green" | "blue";
type Size = "small" | "medium" | "large";

type ClassName = `${Size}-${Color}`;
// "small-red" | "small-green" | ... | "large-blue"

// 이벤트 이름 생성
type EventName<T extends string> = `on${Capitalize<T>}`;
type ClickEvent = EventName<"click">; // "onClick"
```

---

## 7. 연습 문제

### 연습 1: 타입 정의
다음 데이터 구조에 대한 타입을 정의하세요.

```typescript
// 예시 답안
interface Product {
  id: number;
  name: string;
  price: number;
  category: string;
  inStock: boolean;
  tags?: string[];
}

interface CartItem {
  product: Product;
  quantity: number;
}

interface ShoppingCart {
  items: CartItem[];
  total: number;
  couponCode?: string;
}
```

### 연습 2: 제네릭 함수
배열에서 조건에 맞는 첫 번째 요소를 찾는 제네릭 함수를 작성하세요.

```typescript
// 예시 답안
function find<T>(arr: T[], predicate: (item: T) => boolean): T | undefined {
  for (const item of arr) {
    if (predicate(item)) {
      return item;
    }
  }
  return undefined;
}

// 사용
const numbers = [1, 2, 3, 4, 5];
const firstEven = find(numbers, (n) => n % 2 === 0); // 2

const users = [{ name: "Alice" }, { name: "Bob" }];
const alice = find(users, (u) => u.name === "Alice");
```

### 연습 3: 유틸리티 타입 활용
API 응답 타입을 정의하세요.

```typescript
// 예시 답안
interface ApiResponse<T> {
  success: boolean;
  data: T;
  error?: string;
  timestamp: number;
}

type User = {
  id: number;
  name: string;
  email: string;
};

type UserResponse = ApiResponse<User>;
type UsersResponse = ApiResponse<User[]>;
type DeleteResponse = ApiResponse<{ deleted: boolean }>;

// Partial을 활용한 업데이트 타입
type UserUpdate = Partial<Omit<User, "id">>;
```

---

## 다음 단계
- [11. 웹 접근성](./11_Web_Accessibility.md)
- [12. SEO 기초](./12_SEO_Basics.md)

## 참고 자료
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/)
- [TypeScript Playground](https://www.typescriptlang.org/play)
- [DefinitelyTyped](https://github.com/DefinitelyTyped/DefinitelyTyped)
- [TypeScript Deep Dive](https://basarat.gitbook.io/typescript/)
