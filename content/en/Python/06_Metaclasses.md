# Metaclasses

**Previous**: [Closures & Scope](./05_Closures_and_Scope.md) | **Next**: [Descriptors](./07_Descriptors.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why classes in Python are themselves objects and identify `type` as the default metaclass
2. Create classes dynamically using the three-argument form of `type(name, bases, dict)`
3. Implement a custom metaclass by subclassing `type` and overriding `__new__` and `__init__`
4. Distinguish between `__new__` (class creation) and `__init__` (class initialization) in a metaclass
5. Apply metaclass patterns: singleton, class registry, attribute validation, and automatic method generation
6. Use `__init_subclass__` as a simpler alternative to metaclasses for subclass hooks
7. Resolve metaclass conflicts when using multiple inheritance
8. Identify when metaclasses are appropriate versus simpler alternatives (decorators, mixins, `__init_subclass__`)

---

Most Python developers never need to write a metaclass, yet metaclasses power some of the most widely used frameworks in the ecosystem -- Django's ORM, SQLAlchemy's declarative models, and Python's own `abc.ABCMeta`. Understanding how classes are created gives you insight into the deepest layer of Python's object model and equips you to read framework source code, build plugin architectures, and enforce invariants at class definition time rather than at runtime.

## 1. Classes are Objects Too

In Python, everything is an object. Classes are no exception.

```python
class MyClass:
    pass

# The class itself is an object
print(type(MyClass))        # <class 'type'>
print(isinstance(MyClass, type))  # True

# Instance
obj = MyClass()
print(type(obj))            # <class '__main__.MyClass'>
```

### Class Hierarchy

```
┌─────────────────────────────────────────┐
│          type (metaclass)                │
│  • The class of all classes             │
│  • Responsible for creating classes     │
└─────────────────────────────────────────┘
          │ (instance)
          ▼
┌─────────────────────────────────────────┐
│          MyClass (class)                 │
│  • Instance of type                      │
│  • Responsible for creating objects      │
└─────────────────────────────────────────┘
          │ (instance)
          ▼
┌─────────────────────────────────────────┐
│          obj (instance)                  │
│  • Instance of MyClass                   │
└─────────────────────────────────────────┘
```

---

## 2. Creating Classes with type()

`type()` can dynamically create classes.

### type(name, bases, dict)

```python
# Normal class definition
class Dog:
    species = "Canis familiaris"

    def bark(self):
        return "Woof!"

# Creating the same class with type()
Dog = type(
    "Dog",                              # Class name
    (),                                 # Parent class tuple
    {                                   # Attributes and methods
        "species": "Canis familiaris",
        "bark": lambda self: "Woof!"
    }
)

dog = Dog()
print(dog.species)  # Canis familiaris
print(dog.bark())   # Woof!
```

### Including Inheritance

```python
class Animal:
    def breathe(self):
        return "Breathing"

# Create Cat class with type()
Cat = type(
    "Cat",
    (Animal,),  # Inherit from Animal
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

## 3. Defining Metaclasses

> **Analogy:** a regular class is a cookie cutter that stamps out instances (cookies). A metaclass is the machine that manufactures the cookie cutters themselves. You rarely need to build the machine, but when you do, you control the shape of every cutter it produces.

A metaclass is a class that creates classes.

### Basic Metaclass Structure

```python
class MyMeta(type):
    def __new__(mcs, name, bases, namespace):
        """Create class object"""
        print(f"Creating class: {name}")
        return super().__new__(mcs, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        """Initialize class object"""
        print(f"Initializing class: {name}")
        super().__init__(name, bases, namespace)

class MyClass(metaclass=MyMeta):
    pass

# Output:
# Creating class: MyClass
# Initializing class: MyClass
```

### __new__ vs __init__

| Method | Called When | Role |
|--------|------------|------|
| `__new__` | Before class object creation | Create and modify class |
| `__init__` | After class object creation | Initialize class |

```python
class LoggingMeta(type):
    def __new__(mcs, name, bases, namespace):
        # Pre-creation namespace modification lets the metaclass inject default attributes
        # or validate class structure before any instances exist — changes here become
        # part of the class object itself, not just instance state
        namespace['created_by'] = 'LoggingMeta'
        return super().__new__(mcs, name, bases, namespace)

    def __init__(cls, name, bases, namespace):
        # Additional setup after class creation
        cls.initialized = True
        super().__init__(name, bases, namespace)

class MyClass(metaclass=LoggingMeta):
    pass

print(MyClass.created_by)   # LoggingMeta
print(MyClass.initialized)  # True
```

---

## 4. Metaclass Usage Patterns

### Singleton Pattern

```python
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        print("Creating database connection")

db1 = Database()  # Creating database connection
db2 = Database()  # (no output)
print(db1 is db2)  # True
```

### Class Registry

```python
class PluginMeta(type):
    plugins = {}

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if name != 'Plugin':  # Exclude base class — it defines the interface but shouldn't
                              # appear in the registry; only concrete implementations should
                              # be discoverable by plugin consumers
            mcs.plugins[name] = cls
        return cls

class Plugin(metaclass=PluginMeta):
    pass

class JSONPlugin(Plugin):
    def process(self):
        return "Processing JSON"

class XMLPlugin(Plugin):
    def process(self):
        return "Processing XML"

# Check registered plugins
print(PluginMeta.plugins)
# {'JSONPlugin': <class 'JSONPlugin'>, 'XMLPlugin': <class 'XMLPlugin'>}

# Use plugin dynamically
plugin = PluginMeta.plugins['JSONPlugin']()
print(plugin.process())  # Processing JSON
```

### Attribute Validation

```python
class ValidatedMeta(type):
    def __new__(mcs, name, bases, namespace):
        # Validate required_fields attribute
        if 'required_fields' in namespace:
            for field in namespace['required_fields']:
                if field not in namespace:
                    raise TypeError(f"Missing required field: {field}")
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
#     # TypeError: Missing required field: email
```

### Automatic Method Addition

```python
class AutoReprMeta(type):
    def __new__(mcs, name, bases, namespace):
        # Automatically generate __repr__
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

Intercept subclass creation without a metaclass.

```python
class Plugin:
    plugins = {}

    def __init_subclass__(cls, **kwargs):
        # Simpler than a metaclass — no need to subclass type; preferred for most
        # subclass hooks since Python 3.6+.  Use a metaclass only when you need
        # to intercept __new__ or control the class object construction itself.
        super().__init_subclass__(**kwargs)
        # Called when subclass is created
        cls.plugins[cls.__name__] = cls

class JSONPlugin(Plugin):
    pass

class XMLPlugin(Plugin):
    pass

print(Plugin.plugins)
# {'JSONPlugin': <class 'JSONPlugin'>, 'XMLPlugin': <class 'XMLPlugin'>}
```

### Accepting Keyword Arguments

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

Support generic syntax.

```python
class Container:
    def __class_getitem__(cls, item):
        return f"Container[{item.__name__}]"

# Can use generic syntax
print(Container[int])    # Container[int]
print(Container[str])    # Container[str]
```

### Actual Generic Implementation

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

# For type hints
stack: Stack[int] = Stack()
stack.push(1)
stack.push(2)
```

---

## 7. Metaclass Inheritance

Metaclasses are also inherited.

```python
class BaseMeta(type):
    def __new__(mcs, name, bases, namespace):
        namespace['meta_info'] = f"Created by {mcs.__name__}"
        return super().__new__(mcs, name, bases, namespace)

class Base(metaclass=BaseMeta):
    pass

class Child(Base):  # BaseMeta is inherited
    pass

print(Child.meta_info)  # Created by BaseMeta
```

### Resolving Metaclass Conflicts

```python
class Meta1(type):
    pass

class Meta2(type):
    pass

class Base1(metaclass=Meta1):
    pass

class Base2(metaclass=Meta2):
    pass

# Metaclass conflict!
# class Child(Base1, Base2):  # TypeError!
#     pass

# Solution: Create a common metaclass
class CombinedMeta(Meta1, Meta2):
    pass

class Child(Base1, Base2, metaclass=CombinedMeta):
    pass
```

---

## 8. __call__ Method

Controls instance creation.

```python
class LimitedInstancesMeta(type):
    """Metaclass that limits the number of instances"""
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls._instances = []
        cls._max_instances = namespace.get('max_instances', 3)

    def __call__(cls, *args, **kwargs):
        if len(cls._instances) >= cls._max_instances:
            raise RuntimeError(f"Maximum {cls._max_instances} instances allowed")
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

## 9. When to Use Metaclasses?

### When to Use

1. **ORM Frameworks** - Automatic model class registration
2. **Plugin Systems** - Automatic plugin discovery
3. **API Frameworks** - Automatic endpoint registration
4. **Validation Frameworks** - Validation during class definition

### When NOT to Use

> "If you wonder whether you need metaclasses, you don't."
> — Tim Peters

- Simple logic is better with decorators
- When `__init_subclass__` solves the problem
- When code complexity increases significantly

### Alternatives

| Method | When to Use |
|--------|------------|
| Class Decorator | Modify a single class |
| `__init_subclass__` | Logic during subclass creation |
| Mixin Classes | Add common functionality |
| Metaclass | Control class creation itself |

---

## 10. Real-World Example: Django Models

Django ORM uses metaclasses.

```python
# Django style (simplified)
class ModelMeta(type):
    def __new__(mcs, name, bases, namespace):
        # Iterates namespace instead of __annotations__ to also capture fields defined
        # as class-level assignments without type hints — this mirrors Django's approach
        # where Field instances are the source of truth, not the type annotation
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

## 11. Summary

| Concept | Description |
|---------|-------------|
| Metaclass | A class that creates classes |
| `type` | Default metaclass |
| `__new__` | Create class object |
| `__init__` | Initialize class object |
| `__call__` | Control instance creation |
| `__init_subclass__` | Subclass hook without metaclass |
| `__class_getitem__` | Support generic syntax |

---

## 12. Practice Problems

### Exercise 1: Enforce Abstract Methods

Write a metaclass that raises an error if abstract methods are not implemented.

### Exercise 2: Attribute Transformation

Write a metaclass that automatically logs all methods.

### Exercise 3: Immutable Class

Write a metaclass that prohibits attribute changes after instance creation.

---

**Previous**: [Closures & Scope](./05_Closures_and_Scope.md) | **Next**: [Descriptors](./07_Descriptors.md)
