# 레슨 11: 디자인 패턴 입문

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:
1. 디자인 패턴이 무엇이고 왜 중요한지 설명할 수 있다
2. 단일 인스턴스 클래스를 위한 싱글턴 패턴을 구현할 수 있다
3. 객체 생성과 사용을 분리하는 팩토리 패턴을 사용할 수 있다
4. 이벤트 기반 통신을 위한 옵저버 패턴을 적용할 수 있다
5. 교체 가능한 알고리즘을 위한 전략 패턴을 구현할 수 있다
6. 패턴을 생성, 구조, 행동 범주로 분류할 수 있다
7. 일반적인 설계 문제에 적합한 패턴을 선택할 수 있다

## 디자인 패턴이란?

디자인 패턴은 소프트웨어 설계에서 **자주 발생하는 문제에 대한 재사용 가능한 해결책**입니다.

```
┌─────────────────────────────────────────────────┐
│            디자인 패턴 범주                      │
├─────────────────┬───────────────┬───────────────┤
│  생성 패턴      │  구조 패턴    │  행동 패턴    │
│  (객체 생성)    │  (객체 합성)  │  (객체 상호   │
│                 │               │   작용)       │
├─────────────────┼───────────────┼───────────────┤
│ Singleton       │ Adapter       │ Observer      │
│ Factory Method  │ Decorator     │ Strategy      │
│ Abstract Factory│ Facade        │ Command       │
│ Builder         │ Proxy         │ Iterator      │
└─────────────────┴───────────────┴───────────────┘
```

## 패턴 1: 싱글턴 (Singleton)

**의도**: 클래스가 **하나의 인스턴스만** 가지도록 보장하고 전역 접근점을 제공합니다.

```python
class DatabaseConnection:
    """싱글턴 데이터베이스 연결."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, host="localhost", port=5432):
        if self._initialized:
            return
        self.host = host
        self.port = port
        self.connected = False
        self._initialized = True

    def connect(self):
        self.connected = True
        return f"{self.host}:{self.port}에 연결됨"

db1 = DatabaseConnection("prod-server", 5432)
db2 = DatabaseConnection()
print(db1 is db2)  # True — 같은 인스턴스
```

## 패턴 2: 팩토리 (Factory)

**의도**: 객체 생성을 위한 인터페이스를 정의하되, 어떤 클래스를 인스턴스화할지는 런타임 조건에 따라 결정합니다.

```python
class Animal:
    def __init__(self, name):
        self.name = name
    def speak(self):
        raise NotImplementedError

class Dog(Animal):
    def speak(self):
        return f"{self.name}: 멍멍!"

class Cat(Animal):
    def speak(self):
        return f"{self.name}: 야옹!"

class AnimalFactory:
    """타입 문자열을 기반으로 동물을 생성하는 팩토리."""

    _registry = {"dog": Dog, "cat": Cat}

    @classmethod
    def create(cls, animal_type: str, name: str) -> Animal:
        animal_class = cls._registry.get(animal_type.lower())
        if animal_class is None:
            raise ValueError(f"알 수 없는 동물 유형: {animal_type}")
        return animal_class(name)

    @classmethod
    def register(cls, type_name: str, animal_class):
        """새 동물 유형 등록 (확장 가능!)."""
        cls._registry[type_name.lower()] = animal_class

# 클라이언트는 특정 클래스를 알 필요가 없음
dog = AnimalFactory.create("dog", "Rex")
cat = AnimalFactory.create("cat", "나비")

# 팩토리 코드를 수정하지 않고 확장 (OCP!)
class Fish(Animal):
    def speak(self):
        return f"{self.name}: 뻐끔!"

AnimalFactory.register("fish", Fish)
```

## 패턴 3: 옵저버 (Observer)

**의도**: 한 객체의 상태가 변경될 때 모든 종속 객체가 **자동으로 통지**되는 일대다 의존 관계를 정의합니다.

```
┌──────────────┐     통지       ┌──────────────┐
│   Subject    │────────────────▶│  Observer 1  │
│  (발행자)    │                 └──────────────┘
│              │     통지       ┌──────────────┐
│  - observers │────────────────▶│  Observer 2  │
│  + attach()  │                 └──────────────┘
│  + notify()  │     통지       ┌──────────────┐
│              │────────────────▶│  Observer 3  │
└──────────────┘                 └──────────────┘
```

```python
from abc import ABC, abstractmethod

class Observer(ABC):
    @abstractmethod
    def update(self, event: str, data: dict) -> None:
        pass

class EventEmitter:
    """옵저버를 관리하고 이벤트를 발행하는 주체."""

    def __init__(self):
        self._observers: dict[str, list[Observer]] = {}

    def on(self, event: str, observer: Observer):
        if event not in self._observers:
            self._observers[event] = []
        self._observers[event].append(observer)

    def emit(self, event: str, data: dict = None):
        for observer in self._observers.get(event, []):
            observer.update(event, data or {})

class Logger(Observer):
    def update(self, event, data):
        print(f"[LOG] {event}: {data}")

class EmailNotifier(Observer):
    def update(self, event, data):
        if event == "user_registered":
            print(f"[EMAIL] {data.get('email')}에 환영 메일 발송")

app = EventEmitter()
app.on("user_registered", Logger())
app.on("user_registered", EmailNotifier())
app.emit("user_registered", {"email": "alice@example.com"})
```

## 패턴 4: 전략 (Strategy)

**의도**: 알고리즘 군을 정의하고, 각각을 캡슐화하여 런타임에 **교체 가능**하게 만듭니다.

```python
from abc import ABC, abstractmethod

class CompressionStrategy(ABC):
    @abstractmethod
    def compress(self, data: str) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

class ZipCompression(CompressionStrategy):
    @property
    def name(self):
        return "ZIP"
    def compress(self, data):
        return f"[ZIP 압축: {len(data)} -> {len(data)//2} 바이트]"

class GzipCompression(CompressionStrategy):
    @property
    def name(self):
        return "GZIP"
    def compress(self, data):
        return f"[GZIP 압축: {len(data)} -> {len(data)//3} 바이트]"

class FileArchiver:
    """압축 전략을 사용하는 컨텍스트."""

    def __init__(self, strategy: CompressionStrategy):
        self._strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, new_strategy):
        print(f"압축 방식 전환: {self._strategy.name} -> {new_strategy.name}")
        self._strategy = new_strategy

    def archive(self, filename: str, data: str):
        compressed = self._strategy.compress(data)
        print(f"{filename}을 {self._strategy.name}으로 압축: {compressed}")

archiver = FileArchiver(ZipCompression())
archiver.archive("report.txt", "A" * 1000)

archiver.strategy = GzipCompression()  # 런타임에 교체!
archiver.archive("data.csv", "B" * 5000)
```

## 패턴 비교

```
┌────────────┬────────────────────┬──────────────────────┐
│  패턴      │  해결하는 문제     │  해결 방법           │
├────────────┼────────────────────┼──────────────────────┤
│ 싱글턴     │ 전역적으로 정확히  │ __new__에서 인스턴스 │
│            │ 하나의 인스턴스 필요│ 생성 제어            │
├────────────┼────────────────────┼──────────────────────┤
│ 팩토리     │ 런타임까지 어떤    │ 팩토리 메서드/클래스 │
│            │ 클래스를 만들지 모름│ 에 생성 위임         │
├────────────┼────────────────────┼──────────────────────┤
│ 옵저버     │ 객체들이 다른      │ 구독/발행 이벤트     │
│            │ 객체의 변경에 반응  │ 시스템               │
├────────────┼────────────────────┼──────────────────────┤
│ 전략       │ 런타임에 알고리즘  │ 각 알고리즘을 교체   │
│            │ 교체 필요          │ 가능한 객체로 캡슐화 │
└────────────┴────────────────────┴──────────────────────┘
```

## 요약

- 디자인 패턴은 일반적인 OOP 문제에 대한 재사용 가능한 해결책입니다
- **싱글턴**: 하나의 인스턴스 보장; 공유 자원에 사용 (DB 연결, 설정)
- **팩토리**: 생성과 사용을 분리; 등록을 통해 확장 가능
- **옵저버**: 발행/구독 이벤트 시스템; 이벤트 소스와 핸들러를 분리
- **전략**: 합성을 통한 교체 가능한 알고리즘; 조건 로직 제거
- 패턴은 가이드라인이지 엄격한 규칙이 아닙니다 — 실제 문제를 해결할 때 적용하세요

## 다음 단계

[레슨 12: 매직 메서드](12_Magic_Methods.md)에서 내장 연산자와 함수에서 객체의 동작을 커스터마이즈하는 Python의 특수 메서드를 탐구합니다.
