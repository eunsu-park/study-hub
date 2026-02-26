# Descriptors

**Previous**: [Metaclasses](./06_Metaclasses.md) | **Next**: [Async Programming](./08_Async_Programming.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the descriptor protocol (`__get__`, `__set__`, `__delete__`) and its role in attribute access
2. Distinguish between data descriptors and non-data descriptors and describe their priority in the attribute lookup chain
3. Implement the `__get__` method and handle both class-level and instance-level access
4. Reimplement `property` from scratch using the descriptor protocol
5. Write attribute validation descriptors for type checking and range enforcement
6. Use `__set_name__` to automatically capture the attribute name assigned to a descriptor
7. Build ORM-style field classes (`StringField`, `IntegerField`) and a `CachedProperty` descriptor
8. Explain how Python's methods, `staticmethod`, and `classmethod` are implemented as descriptors

---

Descriptors are the hidden mechanism behind some of Python's most familiar features: `property`, `classmethod`, `staticmethod`, and even ordinary method binding. When you write `obj.attr`, Python does not simply look up a dictionary key -- it invokes a sophisticated lookup chain in which descriptors can intercept, validate, or transform the access. Mastering descriptors gives you the power to build reusable attribute-level behaviors (validation, lazy computation, access control) that work cleanly across any class, much like ORM fields in Django or SQLAlchemy.

## 1. What are Descriptors?

Descriptors are objects that customize attribute access. They implement one or more of `__get__`, `__set__`, and `__delete__`.

```python
class Descriptor:
    def __get__(self, obj, objtype=None):
        """Read attribute"""
        pass

    def __set__(self, obj, value):
        """Write attribute"""
        pass

    def __delete__(self, obj):
        """Delete attribute"""
        pass
```

### Attribute Access Flow

```
When accessing obj.attr:
    │
    ▼
┌─────────────────────────────────────────┐
│ 1. Check data descriptor (type(obj).__dict__)│
│    → Has both __get__ and __set__        │
└─────────────────────────────────────────┘
    │ (if not found)
    ▼
┌─────────────────────────────────────────┐
│ 2. Check instance __dict__ (obj.__dict__)│
└─────────────────────────────────────────┘
    │ (if not found)
    ▼
┌─────────────────────────────────────────┐
│ 3. Check non-data descriptor             │
│    → Has only __get__                    │
└─────────────────────────────────────────┘
    │ (if not found)
    ▼
┌─────────────────────────────────────────┐
│ 4. Call __getattr__                      │
└─────────────────────────────────────────┘
```

---

## 2. Data Descriptor vs Non-Data Descriptor

| Type | Methods | Priority |
|------|---------|----------|
| Data Descriptor | `__get__` + `__set__` | High (takes precedence over instance __dict__) |
| Non-Data Descriptor | `__get__` only | Low (after instance __dict__) |

### Non-Data Descriptor Example

```python
class NonDataDescriptor:
    def __get__(self, obj, objtype=None):
        print("__get__ called")
        return "descriptor value"

class MyClass:
    attr = NonDataDescriptor()

obj = MyClass()
print(obj.attr)  # __get__ called, "descriptor value"

# Instance __dict__ takes precedence
obj.__dict__['attr'] = "instance value"
print(obj.attr)  # "instance value" (descriptor ignored)
```

### Data Descriptor Example

```python
class DataDescriptor:
    def __get__(self, obj, objtype=None):
        print("__get__ called")
        return obj.__dict__.get('_attr')

    def __set__(self, obj, value):
        print("__set__ called")
        obj.__dict__['_attr'] = value

class MyClass:
    attr = DataDescriptor()

obj = MyClass()
obj.attr = "test"     # __set__ called
print(obj.attr)       # __get__ called, "test"

# Descriptor takes precedence even with direct __dict__ assignment
obj.__dict__['attr'] = "direct"
print(obj.attr)       # __get__ called, "test" (descriptor wins!)
```

---

## 3. __get__ Method Details

```python
def __get__(self, obj, objtype=None):
    """
    obj: Instance accessing the descriptor (None if accessed via class)
    objtype: Class owning the descriptor
    """
```

### Class vs Instance Access

```python
class Verbose:
    def __get__(self, obj, objtype=None):
        if obj is None:
            # Accessed via class
            return f"Descriptor of class {objtype.__name__}"
        else:
            # Accessed via instance
            return f"Descriptor of instance {obj}"

class MyClass:
    attr = Verbose()

print(MyClass.attr)        # Descriptor of class MyClass
print(MyClass().attr)      # Descriptor of instance <MyClass>
```

---

## 4. Internal Implementation of property

`@property` is implemented using descriptors.

### Implementing property from Scratch

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
            raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None:
            raise AttributeError("can't delete attribute")
        self.fdel(obj)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        return type(self)(self.fget, self.fset, fdel, self.__doc__)
```

### Usage Example

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
            raise ValueError("Radius must be positive")
        self._radius = value

c = Circle(5)
print(c.radius)    # 5
c.radius = 10
print(c.radius)    # 10
```

---

## 5. Attribute Validation Descriptors

### Type Validation

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
                f"{self.name} must be of type {self.expected_type.__name__}"
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

### Range Validation

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
            raise ValueError(f"{self.name} must be at least {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be at most {self.max_value}")
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

Automatically receives the name when assigned to a class.

```python
class Descriptor:
    def __set_name__(self, owner, name):
        """
        owner: Class owning the descriptor
        name: Attribute name assigned to descriptor
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
    x = Descriptor()  # __set_name__(MyClass, 'x') is called
    y = Descriptor()  # __set_name__(MyClass, 'y') is called

obj = MyClass()
obj.x = 10
obj.y = 20
print(obj.x, obj.y)  # 10 20
print(obj.__dict__)  # {'_x': 10, '_y': 20}
```

---

## 7. ORM-Style Field Implementation

### Basic Field Class

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
            raise TypeError(f"{self.name} must be a string")
        super().__set__(obj, value)

class IntegerField(Field):
    def __set__(self, obj, value):
        if not isinstance(value, int):
            raise TypeError(f"{self.name} must be an integer")
        super().__set__(obj, value)

class BooleanField(Field):
    def __set__(self, obj, value):
        if not isinstance(value, bool):
            raise TypeError(f"{self.name} must be a boolean")
        super().__set__(obj, value)
```

### Model Class

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
print(user.is_active)  # True (default value)
```

---

## 8. Lazy Evaluation

### Implementing cached_property

```python
class CachedProperty:
    """Compute on first access and cache"""
    def __init__(self, func):
        self.func = func
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # Return cached value if exists
        if self.name in obj.__dict__:
            return obj.__dict__[self.name]
        # Compute and cache
        value = self.func(obj)
        obj.__dict__[self.name] = value
        return value

class DataAnalyzer:
    def __init__(self, data):
        self.data = data

    @CachedProperty
    def statistics(self):
        print("Computing statistics...")  # Only printed once
        return {
            "sum": sum(self.data),
            "avg": sum(self.data) / len(self.data),
            "max": max(self.data),
            "min": min(self.data),
        }

analyzer = DataAnalyzer([1, 2, 3, 4, 5])
print(analyzer.statistics)  # Computing statistics... (computed)
print(analyzer.statistics)  # (returned from cache)
```

**Note**: Python 3.8+ includes `functools.cached_property`

```python
from functools import cached_property

class DataAnalyzer:
    @cached_property
    def statistics(self):
        # ...
```

---

## 9. Methods are Descriptors

Functions are non-data descriptors.

```python
def func(self):
    pass

# Functions have __get__ method
print(hasattr(func, '__get__'))  # True
```

### How Bound Methods are Created

```python
class MyClass:
    def method(self):
        return "Hello"

obj = MyClass()

# Access via class: returns function
print(MyClass.method)  # <function MyClass.method>

# Access via instance: returns bound method
print(obj.method)      # <bound method MyClass.method>

# What actually happens
print(MyClass.__dict__['method'].__get__(obj, MyClass))
# <bound method MyClass.method>
```

---

## 10. staticmethod and classmethod

### Implementing staticmethod

```python
class StaticMethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        return self.func  # Return function as-is

class MyClass:
    @StaticMethod
    def static_func():
        return "static"

print(MyClass.static_func())    # static
print(MyClass().static_func())  # static
```

### Implementing classmethod

```python
class ClassMethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if objtype is None:
            objtype = type(obj)
        # Bind class as first argument
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

## 11. Summary

| Concept | Description |
|---------|-------------|
| Descriptor | Object that customizes attribute access |
| `__get__` | Read attribute |
| `__set__` | Write attribute |
| `__delete__` | Delete attribute |
| `__set_name__` | Automatically set attribute name (3.6+) |
| Data Descriptor | `__get__` + `__set__` |
| Non-Data Descriptor | `__get__` only |

---

## 12. Practice Problems

### Exercise 1: Read-Only Attribute

Write a descriptor that allows setting once but prevents modification.

### Exercise 2: Logging Descriptor

Write a descriptor that logs all attribute access and modifications.

### Exercise 3: Unit Conversion

Write a descriptor that stores in base units but displays in different units.
(e.g., store in meters, display in kilometers)

---

**Previous**: [Metaclasses](./06_Metaclasses.md) | **Next**: [Async Programming](./08_Async_Programming.md)
