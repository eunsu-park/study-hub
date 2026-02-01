# 디스크립터 (Descriptors)

## 1. 디스크립터란?

디스크립터는 속성 접근을 커스터마이징하는 객체입니다. `__get__`, `__set__`, `__delete__` 중 하나 이상을 구현합니다.

```python
class Descriptor:
    def __get__(self, obj, objtype=None):
        """속성 읽기"""
        pass

    def __set__(self, obj, value):
        """속성 쓰기"""
        pass

    def __delete__(self, obj):
        """속성 삭제"""
        pass
```

### 속성 접근 흐름

```
obj.attr 접근 시:
    │
    ▼
┌─────────────────────────────────────────┐
│ 1. 데이터 디스크립터 확인 (type(obj).__dict__)│
│    → __get__과 __set__ 모두 있으면        │
└─────────────────────────────────────────┘
    │ (없으면)
    ▼
┌─────────────────────────────────────────┐
│ 2. 인스턴스 __dict__ 확인 (obj.__dict__)   │
└─────────────────────────────────────────┘
    │ (없으면)
    ▼
┌─────────────────────────────────────────┐
│ 3. 비데이터 디스크립터 확인                 │
│    → __get__만 있으면                     │
└─────────────────────────────────────────┘
    │ (없으면)
    ▼
┌─────────────────────────────────────────┐
│ 4. __getattr__ 호출                       │
└─────────────────────────────────────────┘
```

---

## 2. 데이터 디스크립터 vs 비데이터 디스크립터

| 종류 | 메서드 | 우선순위 |
|------|--------|----------|
| 데이터 디스크립터 | `__get__` + `__set__` | 높음 (인스턴스 __dict__ 보다 우선) |
| 비데이터 디스크립터 | `__get__`만 | 낮음 (인스턴스 __dict__ 보다 후순위) |

### 비데이터 디스크립터 예제

```python
class NonDataDescriptor:
    def __get__(self, obj, objtype=None):
        print("__get__ 호출")
        return "descriptor value"

class MyClass:
    attr = NonDataDescriptor()

obj = MyClass()
print(obj.attr)  # __get__ 호출, "descriptor value"

# 인스턴스 __dict__에 직접 저장하면 우선됨
obj.__dict__['attr'] = "instance value"
print(obj.attr)  # "instance value" (디스크립터 무시)
```

### 데이터 디스크립터 예제

```python
class DataDescriptor:
    def __get__(self, obj, objtype=None):
        print("__get__ 호출")
        return obj.__dict__.get('_attr')

    def __set__(self, obj, value):
        print("__set__ 호출")
        obj.__dict__['_attr'] = value

class MyClass:
    attr = DataDescriptor()

obj = MyClass()
obj.attr = "test"     # __set__ 호출
print(obj.attr)       # __get__ 호출, "test"

# 인스턴스 __dict__에 직접 저장해도 디스크립터가 우선
obj.__dict__['attr'] = "direct"
print(obj.attr)       # __get__ 호출, "test" (디스크립터 우선!)
```

---

## 3. __get__ 메서드 상세

```python
def __get__(self, obj, objtype=None):
    """
    obj: 디스크립터를 접근하는 인스턴스 (클래스로 접근 시 None)
    objtype: 디스크립터를 소유한 클래스
    """
```

### 클래스 vs 인스턴스 접근

```python
class Verbose:
    def __get__(self, obj, objtype=None):
        if obj is None:
            # 클래스로 접근
            return f"클래스 {objtype.__name__}의 디스크립터"
        else:
            # 인스턴스로 접근
            return f"인스턴스 {obj}의 디스크립터"

class MyClass:
    attr = Verbose()

print(MyClass.attr)        # 클래스 MyClass의 디스크립터
print(MyClass().attr)      # 인스턴스 <MyClass>의 디스크립터
```

---

## 4. property의 내부 구현

`@property`는 디스크립터로 구현되어 있습니다.

### property 직접 구현

```python
class Property:
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("읽을 수 없는 속성")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("쓸 수 없는 속성")
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("삭제할 수 없는 속성")
        self.fdel(obj)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)
```

### 사용 예제

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @Property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("반지름은 양수여야 합니다")
        self._radius = value

c = Circle(5)
print(c.radius)    # 5
c.radius = 10
print(c.radius)    # 10
```

---

## 5. 속성 검증 디스크립터

### 타입 검증

```python
class Typed:
    def __init__(self, expected_type):
        self.expected_type = expected_type
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.name}은(는) {self.expected_type.__name__} 타입이어야 합니다"
            )
        obj.__dict__[self.name] = value

class Person:
    name = Typed(str)
    age = Typed(int)

    def __init__(self, name, age):
        self.name = name
        self.age = age

p = Person("Alice", 30)
print(p.name)  # Alice
print(p.age)   # 30

# p.age = "thirty"  # TypeError!
```

### 범위 검증

```python
class Bounded:
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)

    def __set__(self, obj, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name}은(는) {self.min_value} 이상이어야 합니다")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name}은(는) {self.max_value} 이하여야 합니다")
        obj.__dict__[self.name] = value

class Product:
    price = Bounded(min_value=0)
    quantity = Bounded(min_value=0, max_value=1000)

    def __init__(self, price, quantity):
        self.price = price
        self.quantity = quantity

p = Product(1000, 50)
# p.price = -100     # ValueError!
# p.quantity = 2000  # ValueError!
```

---

## 6. __set_name__ (Python 3.6+)

디스크립터가 클래스에 할당될 때 이름을 자동으로 받습니다.

```python
class Descriptor:
    def __set_name__(self, owner, name):
        """
        owner: 디스크립터를 소유한 클래스
        name: 디스크립터가 할당된 속성 이름
        """
        self.name = name
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)

class MyClass:
    x = Descriptor()  # __set_name__(MyClass, 'x') 호출됨
    y = Descriptor()  # __set_name__(MyClass, 'y') 호출됨

obj = MyClass()
obj.x = 10
obj.y = 20
print(obj.x, obj.y)  # 10 20
print(obj.__dict__)  # {'_x': 10, '_y': 20}
```

---

## 7. ORM 스타일 필드 구현

### 기본 필드 클래스

```python
class Field:
    def __init__(self, default=None):
        self.default = default
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name, self.default)

    def __set__(self, obj, value):
        obj.__dict__[self.name] = value

class StringField(Field):
    def __set__(self, obj, value):
        if not isinstance(value, str):
            raise TypeError(f"{self.name}은(는) 문자열이어야 합니다")
        super().__set__(obj, value)

class IntegerField(Field):
    def __set__(self, obj, value):
        if not isinstance(value, int):
            raise TypeError(f"{self.name}은(는) 정수여야 합니다")
        super().__set__(obj, value)

class BooleanField(Field):
    def __set__(self, obj, value):
        if not isinstance(value, bool):
            raise TypeError(f"{self.name}은(는) 불리언이어야 합니다")
        super().__set__(obj, value)
```

### 모델 클래스

```python
class Model:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        fields = ', '.join(
            f"{k}={v!r}"
            for k, v in self.__dict__.items()
        )
        return f"{self.__class__.__name__}({fields})"

class User(Model):
    name = StringField()
    age = IntegerField()
    is_active = BooleanField(default=True)

user = User(name="Alice", age=30)
print(user)  # User(name='Alice', age=30)
print(user.is_active)  # True (기본값)
```

---

## 8. 지연 계산 (Lazy Evaluation)

### cached_property 구현

```python
class CachedProperty:
    """처음 접근 시 계산하고 캐싱"""
    def __init__(self, func):
        self.func = func
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # 캐시된 값이 있으면 반환
        if self.name in obj.__dict__:
            return obj.__dict__[self.name]
        # 없으면 계산 후 캐싱
        value = self.func(obj)
        obj.__dict__[self.name] = value
        return value

class DataAnalyzer:
    def __init__(self, data):
        self.data = data

    @CachedProperty
    def statistics(self):
        print("통계 계산 중...")  # 한 번만 출력
        return {
            "sum": sum(self.data),
            "avg": sum(self.data) / len(self.data),
            "max": max(self.data),
            "min": min(self.data),
        }

analyzer = DataAnalyzer([1, 2, 3, 4, 5])
print(analyzer.statistics)  # 통계 계산 중... (계산됨)
print(analyzer.statistics)  # (캐시에서 반환)
```

**참고**: Python 3.8+에서는 `functools.cached_property` 사용 가능

```python
from functools import cached_property

class DataAnalyzer:
    @cached_property
    def statistics(self):
        # ...
```

---

## 9. 메서드도 디스크립터

함수는 비데이터 디스크립터입니다.

```python
def func(self):
    pass

# 함수의 __get__ 메서드
print(hasattr(func, '__get__'))  # True
```

### 바운드 메서드 생성 원리

```python
class MyClass:
    def method(self):
        return "Hello"

obj = MyClass()

# 클래스로 접근: 함수 반환
print(MyClass.method)  # <function MyClass.method>

# 인스턴스로 접근: 바운드 메서드 반환
print(obj.method)      # <bound method MyClass.method>

# 실제로 일어나는 일
print(MyClass.__dict__['method'].__get__(obj, MyClass))
# <bound method MyClass.method>
```

---

## 10. staticmethod와 classmethod

### staticmethod 구현

```python
class StaticMethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        return self.func  # 함수 그대로 반환

class MyClass:
    @StaticMethod
    def static_func():
        return "static"

print(MyClass.static_func())    # static
print(MyClass().static_func())  # static
```

### classmethod 구현

```python
class ClassMethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if objtype is None:
            objtype = type(obj)
        # 클래스를 첫 번째 인자로 바인딩
        def method(*args, **kwargs):
            return self.func(objtype, *args, **kwargs)
        return method

class MyClass:
    @ClassMethod
    def class_func(cls):
        return f"class: {cls.__name__}"

print(MyClass.class_func())    # class: MyClass
print(MyClass().class_func())  # class: MyClass
```

---

## 11. 요약

| 개념 | 설명 |
|------|------|
| 디스크립터 | 속성 접근을 커스터마이징하는 객체 |
| `__get__` | 속성 읽기 |
| `__set__` | 속성 쓰기 |
| `__delete__` | 속성 삭제 |
| `__set_name__` | 속성 이름 자동 설정 (3.6+) |
| 데이터 디스크립터 | `__get__` + `__set__` |
| 비데이터 디스크립터 | `__get__`만 |

---

## 12. 연습 문제

### 연습 1: 읽기 전용 속성

한 번 설정되면 수정할 수 없는 디스크립터를 작성하세요.

### 연습 2: 로깅 디스크립터

속성 접근/수정을 모두 로깅하는 디스크립터를 작성하세요.

### 연습 3: 단위 변환

저장은 기본 단위로, 표시는 다른 단위로 하는 디스크립터를 작성하세요.
(예: 미터로 저장, 킬로미터로 표시)

---

## 다음 단계

[08_Async_Programming.md](./08_Async_Programming.md)에서 async/await를 배워봅시다!
