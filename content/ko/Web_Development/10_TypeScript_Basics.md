# 10. TypeScript 기초 (TypeScript Fundamentals)

**이전**: [실전 프로젝트](./09_Practical_Projects.md) | **다음**: [웹 접근성](./11_Web_Accessibility.md)

## 학습 목표(Learning Objectives)

이 레슨을 마치면 다음을 할 수 있습니다:

1. 대규모 프로젝트에서 TypeScript가 순수 JavaScript보다 유리한 점을 설명한다
2. TypeScript의 기본 타입 시스템(type system)을 사용해 변수, 함수, 반환 타입에 어노테이션을 추가한다
3. 인터페이스(interface)로 객체 형태를 정의하고, 타입 별칭(type alias)을 언제 사용할지 구분한다
4. 제네릭(generics)을 사용하여 재사용 가능하고 타입 안전한 함수와 클래스를 구현한다
5. `extends`와 `keyof`를 사용한 제네릭 제약(generic constraints)으로 타입 관계를 강제한다
6. `Partial`, `Pick`, `Omit`, `Record` 등 내장 유틸리티 타입(utility types)을 활용한다
7. `tsconfig.json`으로 TypeScript 프로젝트를 설정하고 JavaScript로 컴파일한다

---

JavaScript의 유연성은 그것의 가장 큰 강점인 동시에 규모가 커질수록 가장 큰 약점이 되기도 합니다. TypeScript는 정적 타입 레이어를 추가하여 컴파일 시점에 오류를 잡아냅니다 — 사용자에게 배포되기 전에 말이죠. TypeScript를 채택한다고 해서 JavaScript를 포기하는 것이 아닙니다. 유효한 JavaScript 프로그램은 이미 유효한 TypeScript입니다. 이 레슨은 현대 웹 개발에서 TypeScript를 없어서는 안 될 도구로 만드는 핵심 타입 시스템 기능들을 익히게 합니다.

## 목차

참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. TypeScript는 JavaScript 위에 *구조적(structural)* 타입 시스템을 더하며 런타임에는 지워집니다(erase). 일상의 안전 메커니즘은 타입 좁히기(narrowing), 매개변수화된 타입은 제네릭(generics), 그리고 `tsc`는 런타임 동작을 바꾸지 않는 빌드 타임 전용 컴파일러입니다.

1. [TypeScript 소개](#1-typescript-소개)
2. [기본 타입](#2-기본-타입)
3. [인터페이스와 타입](#3-인터페이스와-타입)
4. [함수 타입](#4-함수-타입)
5. [제네릭](#5-제네릭)
6. [유틸리티 타입](#6-유틸리티-타입)
7. [연습 문제](#7-연습-문제)

---

## 이론과 원리

TypeScript는 어노테이션이 더해진 JavaScript처럼 보이지만, 그 아래의 설계는 결과적입니다. 세 속성이 거의 모든 TypeScript 결정의 모양을 잡습니다 — 타입은 **구조적(structural)** 이며(이름이 아니라 *모양* 으로 호환), 타입은 컴파일 시점에 **지워지고(erased)** (런타임 비용 0, 런타임 보장 0), 컴파일러는 코드 진행에 따라 타입을 좁히는 정교한 **흐름 분석(flow analysis)** 을 실행합니다. 문법 투어를 읽기 전에 이 셋에 이름을 붙이면, 다른 모든 것이 어휘 목록이 아니라 자명한 결과처럼 느껴집니다.

### A. 구조적 타이핑: 모양으로 호환

*명목적(nominal)* 타입 시스템(Java, C#)에서는 같은 멤버를 가진 두 타입도 이름이 다르면 호환되지 않습니다 — `class Dog { name: string }`과 `class Cat { name: string }`은 서로 대체될 수 없습니다. TypeScript는 *구조적* 입니다 — 값이 적절한 모양을 가지고 있다면, 어디서 왔는지에 관계없이 어떤 타입에 할당 가능합니다.

```ts
interface Named { name: string }

function greet(n: Named) { console.log(n.name); }

class Dog { constructor(public name: string) {} }
greet(new Dog("Rex"));      // OK
greet({ name: "Anonymous" }); // OK — 모양이 맞는 평범한 객체
```

기억할 만한 두 결과:

1. **객체 리터럴은 *과잉 속성 검사(excess property check)* 를 받습니다.** `{ name: "x", color: "red" }`를 `greet`에 직접 넘기면, 그 값이 구조적으로 `Named`임에도 불구하고 인식되지 않는 `color` 필드에서 오류가 납니다. 변수에 먼저 보관하면 오류가 사라집니다 — 변수의 추론된 타입에 이미 `color`가 포함되어 있기 때문입니다. 이는 리터럴 자리의 오타를 잡기 위한 의도적 설계 선택입니다.
2. **`unknown`과 `any`는 다릅니다.** `any`는 타입 검사에서 완전히 빠집니다 — 어떤 것이든 받고 어디든 할당 가능합니다. `unknown`은 어떤 것이든 받지만, 먼저 좁히지 않으면 *어디에도* 할당 불가능합니다. 경계 입력(JSON, DOM 이벤트)에는 `unknown`을 사용하고, `any`는 의도적인 비상구로만 사용하세요.

### B. 타입 소거: 컴파일 타임만

TypeScript 타입은 런타임에 존재하지 않습니다. 컴파일러는 `.ts`를 읽고 타입을 검사한 뒤, 모든 타입 어노테이션이 제거된 `.js`를 emit합니다. 인터페이스에 대한 `instanceof` 테스트도, 제네릭 매개변수의 리플렉션도, 런타임 브랜딩도 *없습니다*. 이는 즉각적으로 세 함의를 가집니다.

1. **타입만으로는 들어오는 데이터를 타입 검사할 수 없습니다.** `data as User`는 모양을 검증하지 않고, 컴파일러에게 "믿어 줘"라고 말할 뿐입니다. 경계(fetch에서 받은 JSON, 쿼리 스트링 파싱, DOM 입력)에서의 진짜 검증을 위해서는, 타입화된 값을 반환 *하면서도* 검사하는 런타임 검증기(Zod, Valibot, ArkType)와 TypeScript를 짝지으세요.
2. **타입 오류는 컴파일러가 컴파일을 거부하는 경고입니다.** `// @ts-ignore`로 무시하면, JavaScript는 여전히 실행됩니다 — 안전망을 잃은 것뿐입니다. 그래서 CI 설정은 `strict: true`를 두고 타입 오류를 빌드 실패로 간주합니다.
3. **일부 구문은 소거에서 살아남습니다.** `enum`, `class`, `namespace`, 매개변수 속성(`constructor(public x: number)`)은 JavaScript 출력을 emit합니다. 순수 타입(`interface`, `type`, 제네릭 매개변수)은 아무것도 emit하지 않습니다. 이식성을 위해 두 번째 범주를 선호하세요.

### C. 타입 좁히기(Narrowing): 흐름 민감 분석

TypeScript를 즐겁게 만드는 단일 기능이 **좁히기(narrowing)** 입니다 — 컴파일러는 각 제어 흐름 분기가 변수의 타입을 어떻게 정제하는지 추적합니다.

```ts
function format(x: string | number) {
  if (typeof x === "string") {
    return x.toUpperCase(); // 여기서 x는 string으로 좁혀짐
  }
  return x.toFixed(2);      // 여기서 x는 number로 좁혀짐
}
```

컴파일러가 이해하는 좁히기 연산자에는 다음이 있습니다.

- **`typeof x === "..."`** — 원시에 대해.
- **`x instanceof Class`** — 클래스에 대해(런타임에 존재).
- **`"key" in obj`** — 구조로 union 멤버를 구별.
- 리터럴에 대한 **동등성**(`x === "loading"`).
- **사용자 정의 타입 가드(custom type guard)** — 반환 타입이 `x is T`인 함수. `true`를 반환한 분기 안에서 컴파일러가 그것을 신뢰합니다.
- **`!` 비-null 단언(non-null assertion)** — `x!`는 컴파일러에게 "이것이 null/undefined 아님을 안다"라고 말합니다. 드물게 사용하고, 좁히기를 선호하세요.

이것이 가능하게 하는 가장 유용한 패턴이 **판별된 union(discriminated union)** 입니다 — 리터럴 필드를 공유하는 객체 타입의 union으로, 그 필드가 판별자로 사용됩니다.

```ts
type State =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; data: User };

function render(s: State) {
  switch (s.status) {
    case "loading": ...
    case "error":  s.message  // 좁혀짐: error 변형은 message를 가짐
    case "success": s.data    // 좁혀짐: success 변형은 data를 가짐
  }
}
```

이는 정확히 이전 레슨의 §C 패턴이 타입 시스템에 인코딩된 것입니다. 새 상태를 추가하면 모든 `switch`가 그것을 처리하도록 강제됩니다(`never` 망라성 검사로).

### D. 제네릭(Generics): 매개변수화된 타입

**제네릭 매개변수(generic parameter)** 를 가진 함수나 타입은 *호출자* 가 타입을 채워 넣게 하고, 컴파일러는 그것을 본문 전체에 전파합니다. 정전인 예:

```ts
function first<T>(arr: T[]): T | undefined {
  return arr[0];
}

const n = first([1, 2, 3]);          // T가 number로 추론 → n: number | undefined
const s = first(["a", "b"]);         // T가 string으로 추론 → s: string | undefined
```

제네릭은 두 실패 모드를 피합니다 — `any`를 반환(다운스트림의 모든 타입 정보를 잃음)하고 N개 타입에 대해 같은 함수의 N개 사본을 쓰는 것.

반복적으로 등장하는 두 정제 메커니즘:

- **`extends`로 제약(constraint).** `<T extends { id: number }>`는 "T는 어떤 타입이든 될 수 있되, `id: number` 필드를 가져야 한다"라고 말합니다. 함수 안에서 `obj.id`는 이제 안전하게 읽을 수 있습니다.
- **`keyof T`.** "T의 속성 이름의 union" 타입. 인덱스 접근(`T[K]`)과 결합하면, "T의 속성 K의 타입을 줘"를 표현하며, 이것이 `Pick`, `Omit`, `Record`가 지어지는 방식입니다.

유틸리티 타입은 이 프리미티브 위의 레시피로 출하됩니다. `Partial<T>`는 모든 속성을 옵셔널로, `Required<T>`는 모두 필수로, `Pick<T, K>`는 명명된 속성만 유지, `Omit<T, K>`는 그것들을 떨어뜨림, `Record<K, V>`는 키 K와 값 V의 객체 타입을 짓고, `ReturnType<F>`는 함수의 반환 타입을 추출하며, `Awaited<P>`는 Promise의 resolve된 타입을 풀어 냅니다. `lib.es5.d.ts`의 정의를 읽을 수 있게 되면, 자신의 것도 쓸 수 있습니다.

### 이론에서 아래 참조로

- **TypeScript 소개**(섹션 1)는 §B의 컴파일 타임 전용 모델과 `tsc` 워크플로우를 다룹니다.
- **기본 타입**(섹션 2)은 `string`, `number`, `boolean`, `unknown`, `any`, `never`, 그리고 배열과 튜플을 소개합니다 — 모두 §A의 구조적 규칙에 지배됩니다.
- **인터페이스와 타입**(섹션 3)은 문법으로의 §A입니다 — 모양 선언, `interface`(확장 가능)와 `type`(합성 가능) 사이의 선택.
- **함수 타입**(섹션 4)은 매개변수 타입, 반환 타입, 오버로드 — 그리고 매개변수 반변(contravariant) 규칙 — 을 다룹니다.
- **제네릭**(섹션 5)은 §D입니다 — 매개변수, 제약, `keyof`, `extends`.
- **유틸리티 타입**(섹션 6)은 §D 위의 표준 라이브러리입니다 — `Partial`, `Pick`, `Omit`, `Record`, `Awaited`, `ReturnType`.

레슨의 나머지를, 모든 어노테이션이 컴파일러가 소거 전에 검사할 제약이며 — 좁히기가 느슨한 union 타입을 구체적 타입으로 바꾸는 일상의 메커니즘이라는 점을 알고 읽으세요.

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
    "strict": true,          // strictNullChecks, noImplicitAny 등 10가지 이상의 검사를 활성화 — 일반 JS에서 런타임 오류가 될 버그를 컴파일 시점에 포착
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
// T는 인수 타입에서 추론됨 — 수동 캐스팅 없이 TypeScript가 arg와 반환값이 같은 타입임을 보장
function identity<T>(arg: T): T {
  return arg;
}

// 사용
let output1 = identity<string>("hello");
let output2 = identity<number>(42);
let output3 = identity("auto"); // 타입 추론 — 인수로부터 T = string을 자동으로 추론

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

// Partial<T> - 모든 User 필드를 선택적으로 만듦 — 일부 필드만 변경하는 업데이트 함수에 유용
type PartialUser = Partial<User>;
// { id?: number; name?: string; email?: string; age?: number }

// Required<T> - 모든 속성 필수로
type RequiredUser = Required<User>;
// { id: number; name: string; email: string; age: number }

// Readonly<T> - 모든 속성 읽기 전용
type ReadonlyUser = Readonly<User>;

// Pick<T, K> - 필요한 필드만 선택; 공개 타입에서 민감한 필드(예: password, token) 노출 방지에 활용
type UserBasic = Pick<User, "id" | "name">;
// { id: number; name: string }

// Omit<T, K> - 원하지 않는 필드 제외; 서버 생성 필드(id, createdAt 등)를 제거한 "입력(input)" 타입 생성에 유용
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
