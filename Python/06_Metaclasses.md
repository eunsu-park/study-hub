# 메타클래스 (Metaclasses)

## 1. 클래스도 객체다

파이썬에서 모든 것은 객체입니다. 클래스도 예외가 아닙니다.

```python
class MyClass:
    pass

# 클래스 자체가 객체
print(type(MyClass))        # <class 'type'>
print(isinstance(MyClass, type))  # True

# 인스턴스
obj = MyClass()
print(type(obj))            # <class '__main__.MyClass'>
```

### 클래스 계층 구조

```
┌─────────────────────────────────────────┐
│              type (메타클래스)           │
│  • 모든 클래스의 클래스                  │
│  • 클래스를 생성하는 역할                │
└─────────────────────────────────────────┘
          │ (인스턴스)
          ▼
┌─────────────────────────────────────────┐
│            MyClass (클래스)              │
│  • type의 인스턴스                       │
│  • 객체를 생성하는 역할                  │
└─────────────────────────────────────────┘
          │ (인스턴스)
          ▼
┌─────────────────────────────────────────┐
│            obj (인스턴스)                │
│  • MyClass의 인스턴스                    │
└─────────────────────────────────────────┘
```

---

## 2. type()으로 클래스 생성

`type()`은 클래스를 동적으로 생성할 수 있습니다.

### type(name, bases, dict)

```python
# 일반적인 클래스 정의
class Dog:
    species = "Canis familiaris"

    def bark(self):
        return "Woof!"

# type()으로 동일한 클래스 생성
Dog = type(
    "Dog",                              # 클래스 이름
    (),                                 # 부모 클래스 튜플
    {                                   # 속성과 메서드
        "species": "Canis familiaris",
        "bark": lambda self: "Woof!"
    }
)

dog = Dog()
print(dog.species)  # Canis familiaris
print(dog.bark())   # Woof!
```

### 상속 포함

```python
class Animal:
    def breathe(self):
        return "Breathing"

# Cat 클래스를 type()으로 생성
Cat = type(
    "Cat",
    (Animal,),  # Animal 상속
    {
        "meow": lambda self: "Meow!",
        "species": "Felis catus"
    }
)

cat = Cat()
print(cat.breathe())  # Breathing
print(cat.meow())     # Meow!
```

---

## 3. 메타클래스 정의

메타클래스는 클래스를 생성하는 클래스입니다.

### 기본 메타클래스 구조

```python
class MyMeta(type):
    def __new__(mcs, name, bases, namespace):
        """클래스 객체 생성"""
        print(f"클래스 생성: {name}")
        return super().__new__(mcs, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        """클래스 객체 초기화"""
        print(f"클래스 초기화: {name}")
        super().__init__(name, bases, namespace)

class MyClass(metaclass=MyMeta):
    pass

# 출력:
# 클래스 생성: MyClass
# 클래스 초기화: MyClass
```

### __new__ vs __init__

| 메서드 | 호출 시점 | 역할 |
|--------|----------|------|
| `__new__` | 클래스 객체 생성 전 | 클래스 생성 및 수정 |
| `__init__` | 클래스 객체 생성 후 | 클래스 초기화 |

```python
class LoggingMeta(type):
    def __new__(mcs, name, bases, namespace):
        # 클래스 생성 전에 namespace 수정 가능
        namespace['created_by'] = 'LoggingMeta'
        return super().__new__(mcs, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        # 클래스 생성 후 추가 설정
        cls.initialized = True
        super().__init__(name, bases, namespace)

class MyClass(metaclass=LoggingMeta):
    pass

print(MyClass.created_by)   # LoggingMeta
print(MyClass.initialized)  # True
```

---

## 4. 메타클래스 활용 패턴

### 싱글톤 패턴

```python
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        print("데이터베이스 연결 생성")

db1 = Database()  # 데이터베이스 연결 생성
db2 = Database()  # (출력 없음)
print(db1 is db2)  # True
```

### 클래스 레지스트리

```python
class PluginMeta(type):
    plugins = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if name != 'Plugin':  # 기본 클래스 제외
            mcs.plugins[name] = cls
        return cls

class Plugin(metaclass=PluginMeta):
    pass

class JSONPlugin(Plugin):
    def process(self):
        return "JSON 처리"

class XMLPlugin(Plugin):
    def process(self):
        return "XML 처리"

# 등록된 플러그인 확인
print(PluginMeta.plugins)
# {'JSONPlugin': <class 'JSONPlugin'>, 'XMLPlugin': <class 'XMLPlugin'>}

# 플러그인 동적 사용
plugin = PluginMeta.plugins['JSONPlugin']()
print(plugin.process())  # JSON 처리
```

### 속성 검증

```python
class ValidatedMeta(type):
    def __new__(mcs, name, bases, namespace):
        # required_fields 속성 검증
        if 'required_fields' in namespace:
            for field in namespace['required_fields']:
                if field not in namespace:
                    raise TypeError(f"필수 필드 누락: {field}")
        return super().__new__(mcs, name, bases, namespace)

class Model(metaclass=ValidatedMeta):
    pass

class User(Model):
    required_fields = ['name', 'email']
    name = "default"
    email = "default@example.com"

# class InvalidUser(Model):
#     required_fields = ['name', 'email']
#     name = "default"
#     # TypeError: 필수 필드 누락: email
```

### 자동 메서드 추가

```python
class AutoReprMeta(type):
    def __new__(mcs, name, bases, namespace):
        # __repr__ 자동 생성
        if '__repr__' not in namespace:
            def auto_repr(self):
                attrs = ', '.join(
                    f"{k}={v!r}"
                    for k, v in vars(self).items()
                )
                return f"{name}({attrs})"
            namespace['__repr__'] = auto_repr

        return super().__new__(mcs, name, bases, namespace)

class Point(metaclass=AutoReprMeta):
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(3, 4)
print(p)  # Point(x=3, y=4)
```

---

## 5. __init_subclass__ (Python 3.6+)

메타클래스 없이 서브클래스 생성을 가로챌 수 있습니다.

```python
class Plugin:
    plugins = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # 서브클래스가 생성될 때 호출됨
        cls.plugins[cls.__name__] = cls

class JSONPlugin(Plugin):
    pass

class XMLPlugin(Plugin):
    pass

print(Plugin.plugins)
# {'JSONPlugin': <class 'JSONPlugin'>, 'XMLPlugin': <class 'XMLPlugin'>}
```

### 키워드 인자 받기

```python
class Serializer:
    def __init_subclass__(cls, format_type=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.format_type = format_type

class JSONSerializer(Serializer, format_type="json"):
    pass

class XMLSerializer(Serializer, format_type="xml"):
    pass

print(JSONSerializer.format_type)  # json
print(XMLSerializer.format_type)   # xml
```

---

## 6. __class_getitem__ (Python 3.9+)

제네릭 문법을 지원하게 합니다.

```python
class Container:
    def __class_getitem__(cls, item):
        return f"Container[{item.__name__}]"

# 제네릭 문법 사용 가능
print(Container[int])    # Container[int]
print(Container[str])    # Container[str]
```

### 실제 제네릭 구현

```python
from typing import Generic, TypeVar

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self):
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

# 타입 힌트용
stack: Stack[int] = Stack()
stack.push(1)
stack.push(2)
```

---

## 7. 메타클래스 상속

메타클래스도 상속됩니다.

```python
class BaseMeta(type):
    def __new__(mcs, name, bases, namespace):
        namespace['meta_info'] = f"Created by {mcs.__name__}"
        return super().__new__(mcs, name, bases, namespace)

class Base(metaclass=BaseMeta):
    pass

class Child(Base):  # BaseMeta 상속됨
    pass

print(Child.meta_info)  # Created by BaseMeta
```

### 메타클래스 충돌 해결

```python
class Meta1(type):
    pass

class Meta2(type):
    pass

class Base1(metaclass=Meta1):
    pass

class Base2(metaclass=Meta2):
    pass

# 메타클래스 충돌!
# class Child(Base1, Base2):  # TypeError!
#     pass

# 해결: 공통 메타클래스 생성
class CombinedMeta(Meta1, Meta2):
    pass

class Child(Base1, Base2, metaclass=CombinedMeta):
    pass
```

---

## 8. __call__ 메서드

인스턴스 생성을 제어합니다.

```python
class LimitedInstancesMeta(type):
    """인스턴스 수를 제한하는 메타클래스"""
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls._instances = []
        cls._max_instances = namespace.get('max_instances', 3)

    def __call__(cls, *args, **kwargs):
        if len(cls._instances) >= cls._max_instances:
            raise RuntimeError(f"최대 {cls._max_instances}개 인스턴스만 허용")
        instance = super().__call__(*args, **kwargs)
        cls._instances.append(instance)
        return instance

class LimitedClass(metaclass=LimitedInstancesMeta):
    max_instances = 2

obj1 = LimitedClass()  # OK
obj2 = LimitedClass()  # OK
# obj3 = LimitedClass()  # RuntimeError!
```

---

## 9. 언제 메타클래스를 사용하는가?

### 사용해야 할 때

1. **ORM 프레임워크** - 모델 클래스 자동 등록
2. **플러그인 시스템** - 플러그인 자동 발견
3. **API 프레임워크** - 엔드포인트 자동 등록
4. **검증 프레임워크** - 클래스 정의 시 검증

### 사용하지 말아야 할 때

> "메타클래스가 필요한지 고민된다면, 필요 없는 것이다."
> — Tim Peters

- 간단한 로직은 데코레이터로 충분
- `__init_subclass__`로 해결되는 경우
- 코드 복잡성이 크게 증가하는 경우

### 대안들

| 방법 | 사용 시점 |
|------|----------|
| 클래스 데코레이터 | 단일 클래스 수정 |
| `__init_subclass__` | 서브클래스 생성 시 로직 |
| 믹스인 클래스 | 공통 기능 추가 |
| 메타클래스 | 클래스 생성 자체를 제어 |

---

## 10. 실제 사용 예시: Django 모델

Django ORM은 메타클래스를 사용합니다.

```python
# Django 스타일 (간소화)
class ModelMeta(type):
    def __new__(mcs, name, bases, namespace):
        # 필드 수집
        fields = {}
        for key, value in namespace.items():
            if isinstance(value, Field):
                fields[key] = value

        namespace['_fields'] = fields
        return super().__new__(mcs, name, bases, namespace)

class Field:
    def __init__(self, field_type):
        self.field_type = field_type

class Model(metaclass=ModelMeta):
    pass

class User(Model):
    name = Field("string")
    age = Field("integer")

print(User._fields)
# {'name': <Field>, 'age': <Field>}
```

---

## 11. 요약

| 개념 | 설명 |
|------|------|
| 메타클래스 | 클래스를 생성하는 클래스 |
| `type` | 기본 메타클래스 |
| `__new__` | 클래스 객체 생성 |
| `__init__` | 클래스 객체 초기화 |
| `__call__` | 인스턴스 생성 제어 |
| `__init_subclass__` | 메타클래스 없이 서브클래스 훅 |
| `__class_getitem__` | 제네릭 문법 지원 |

---

## 12. 연습 문제

### 연습 1: 추상 메서드 강제

추상 메서드가 구현되지 않으면 에러를 발생시키는 메타클래스를 작성하세요.

### 연습 2: 속성 변환

모든 메서드를 자동으로 로깅하는 메타클래스를 작성하세요.

### 연습 3: 불변 클래스

인스턴스 생성 후 속성 변경을 금지하는 메타클래스를 작성하세요.

---

## 다음 단계

[07_Descriptors.md](./07_Descriptors.md)에서 속성 접근을 제어하는 디스크립터를 배워봅시다!
