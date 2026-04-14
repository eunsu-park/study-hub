# Logging

**Previous**: [Debugging Strategy](./05_Debugging_Strategy.md) | **Next**: [Testing Basics](./07_Testing_Basics.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why logging is superior to print debugging for production code
2. Use the `logging` module with appropriate log levels (DEBUG through CRITICAL)
3. Configure log output format with timestamps, source locations, and custom fields
4. Direct log output to files, console, and multiple destinations using handlers
5. Create and use named loggers for modular applications
6. Apply `logging.basicConfig()` and dictionary-based configuration
7. Use structured logging patterns for machine-readable output
8. Avoid common logging pitfalls (performance, security, formatting)

---

While print debugging works for quick investigations, production code needs something more robust. The `logging` module is Python's built-in solution for producing diagnostic output that can be filtered by severity, routed to different destinations, and left in place permanently. Unlike print statements, log calls can remain in your code and be turned on or off without code changes.

> **Rule of Thumb:** If you reach for `print()` to diagnose a problem in code that runs in production, use `logging` instead. Print is for development; logging is for operations.

---

## 1. Why Logging, Not Print

| Feature | `print()` | `logging` |
|---------|-----------|-----------|
| Severity levels | No | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| Enable/disable without code change | No | Yes (configuration) |
| Timestamps | Manual | Automatic |
| Output destination | stdout only | Files, stderr, network, email, etc. |
| Source location | Manual | Automatic (file, line, function) |
| Thread safety | No | Yes |
| Performance control | No | Level filtering before formatting |
| Production-ready | No | Yes |

---

## 2. Quick Start

### 2.1 The Simplest Example

```python
import logging

logging.basicConfig(level=logging.DEBUG)

logging.debug("Starting calculation")
logging.info("Processing 100 records")
logging.warning("Disk space below 10%")
logging.error("Failed to connect to database")
logging.critical("System memory exhausted, shutting down")
```

Output:
```
DEBUG:root:Starting calculation
INFO:root:Processing 100 records
WARNING:root:Disk space below 10%
ERROR:root:Failed to connect to database
CRITICAL:root:System memory exhausted, shutting down
```

### 2.2 Default Behavior

Without `basicConfig()`, only WARNING and above are shown:

```python
import logging
logging.debug("Not shown")
logging.info("Not shown")
logging.warning("This IS shown")
```

---

## 3. Log Levels

```
┌──────────────────────────────────────────────────────┐
│  Level      Value   When to Use                      │
├──────────────────────────────────────────────────────┤
│  DEBUG       10     Detailed diagnostic information  │
│                     (variable values, flow tracing)  │
│  INFO        20     Confirmation that things work    │
│                     ("Server started on port 8080")  │
│  WARNING     30     Something unexpected but not     │
│                     broken ("Retry attempt 3 of 5")  │
│  ERROR       40     Something failed, but program    │
│                     continues ("DB query failed")    │
│  CRITICAL    50     Program cannot continue          │
│                     ("Out of memory, shutting down")  │
└──────────────────────────────────────────────────────┘

     Increasing severity →→→→→→→→→→→→→→→→→→→→→
```

### Choosing the Right Level

```python
import logging

logger = logging.getLogger(__name__)

def process_order(order):
    logger.debug(f"Processing order: {order}")           # Dev details
    
    if order["total"] > 10000:
        logger.info(f"Large order: #{order['id']}")      # Business event
    
    if order["stock"] < order["quantity"]:
        logger.warning(f"Low stock for item {order['item']}")  # Concerning
    
    try:
        charge_payment(order)
    except PaymentError as e:
        logger.error(f"Payment failed for order #{order['id']}: {e}")  # Failure
    
    if not db.is_connected():
        logger.critical("Database connection lost!")      # System failure
```

---

## 4. Log Formatting

### 4.1 Format String

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("Server started")
# 2024-01-15 10:30:45 [INFO    ] root:8 - Server started
```

### 4.2 Common Format Fields

| Field | Description | Example |
|-------|-------------|---------|
| `%(asctime)s` | Timestamp | `2024-01-15 10:30:45,123` |
| `%(levelname)s` | Log level name | `INFO` |
| `%(name)s` | Logger name | `my_module` |
| `%(filename)s` | Source file name | `app.py` |
| `%(lineno)d` | Line number | `42` |
| `%(funcName)s` | Function name | `process_order` |
| `%(message)s` | Log message | `Processing complete` |
| `%(module)s` | Module name | `app` |
| `%(process)d` | Process ID | `12345` |
| `%(thread)d` | Thread ID | `140234` |

### 4.3 Useful Format Templates

```python
# Development (verbose)
DEV_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"

# Production (concise)
PROD_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"

# JSON-like (for log aggregation tools)
JSON_FORMAT = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
```

---

## 5. Handlers: Where Logs Go

### 5.1 Console Handler

```python
import logging

logger = logging.getLogger("myapp")
logger.setLevel(logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.INFO)       # Console only shows INFO+
console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
logger.addHandler(console)
```

### 5.2 File Handler

```python
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)  # File captures everything
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
))
logger.addHandler(file_handler)
```

### 5.3 Multiple Handlers (Common Pattern)

```python
import logging

def setup_logging():
    logger = logging.getLogger("myapp")
    logger.setLevel(logging.DEBUG)
    
    # Console: INFO and above
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    
    # File: everything (DEBUG and above)
    file_handler = logging.FileHandler("debug.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s"
    ))
    
    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    return logger
```

### 5.4 Rotating File Handler

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "app.log",
    maxBytes=5_000_000,     # 5 MB per file
    backupCount=3,          # Keep 3 backup files
)
```

### 5.5 Handler Architecture

```
              Logger (myapp)
              level=DEBUG
                   │
          ┌────────┴────────┐
          ▼                 ▼
    StreamHandler     FileHandler
    level=INFO        level=DEBUG
          │                 │
          ▼                 ▼
    Console (stderr)   app.log
    INFO+ messages     ALL messages
```

---

## 6. Named Loggers

### 6.1 Module-Level Loggers

Each module should have its own logger:

```python
# file: database.py
import logging
logger = logging.getLogger(__name__)   # logger name = "database"

def connect():
    logger.info("Connecting to database...")
    ...

# file: api.py
import logging
logger = logging.getLogger(__name__)   # logger name = "api"

def handle_request():
    logger.debug("Received request")
    ...
```

### 6.2 Logger Hierarchy

Loggers form a hierarchy based on dot-separated names:

```
root
├── myapp                  # logging.getLogger("myapp")
│   ├── myapp.database     # logging.getLogger("myapp.database")
│   ├── myapp.api          # logging.getLogger("myapp.api")
│   └── myapp.api.auth     # logging.getLogger("myapp.api.auth")
```

Child loggers propagate messages to parent loggers by default.

### 6.3 Selective Logging

```python
# Show only database-related debug logs, silence everything else
logging.getLogger("myapp").setLevel(logging.WARNING)
logging.getLogger("myapp.database").setLevel(logging.DEBUG)
```

---

## 7. Configuration Methods

### 7.1 `basicConfig()` (Simple Scripts)

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="app.log",       # Log to file (omit for console)
    filemode="a",             # Append (default) or "w" for overwrite
)
```

### 7.2 Dictionary Configuration (Applications)

```python
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": "app.log",
        },
    },
    "loggers": {
        "myapp": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        },
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
```

---

## 8. Logging Best Practices

### 8.1 Use Lazy Formatting

```python
# BAD: String is always formatted, even if level is filtered
logger.debug(f"Processing {len(large_list)} items: {large_list}")

# GOOD: String is only formatted if DEBUG level is enabled
logger.debug("Processing %d items: %s", len(large_list), large_list)
```

### 8.2 Log Exceptions Properly

```python
# BAD: Loses stack trace
try:
    process()
except Exception as e:
    logger.error(f"Failed: {e}")

# GOOD: Includes full traceback
try:
    process()
except Exception:
    logger.exception("Failed to process")  # Automatically adds traceback

# ALTERNATIVE: exc_info=True
try:
    process()
except Exception:
    logger.error("Failed to process", exc_info=True)
```

### 8.3 Don't Log Sensitive Data

```python
# BAD: Password in logs!
logger.info(f"User login: {username}, password: {password}")

# GOOD: Redact sensitive fields
logger.info(f"User login: {username}")
```

### 8.4 Use Structured Context

```python
# Include relevant context for filtering/searching
logger.info(
    "Order processed",
    extra={"order_id": order.id, "amount": order.total, "user": order.user_id},
)
```

---

## 9. Logging vs Print: Decision Guide

```
Is the diagnostic output...
│
├─ Temporary (remove before commit)?
│   └─ Use print() or debug_print()
│
├─ Permanent (stays in codebase)?
│   └─ Use logging
│
├─ Needed in production?
│   └─ Use logging
│
├─ For a quick one-off script?
│   └─ print() is fine
│
└─ Part of the program's actual output?
    └─ Use print() (to stdout)
        Diagnostics go to logging (to stderr/file)
```

---

## 10. Practical Example: Logging in a Data Pipeline

```python
import logging

logger = logging.getLogger(__name__)

def run_pipeline(input_path, output_path):
    logger.info("Pipeline started: input=%s, output=%s", input_path, output_path)
    
    # Step 1: Load
    logger.debug("Loading data from %s", input_path)
    try:
        data = load_data(input_path)
    except FileNotFoundError:
        logger.error("Input file not found: %s", input_path)
        return False
    logger.info("Loaded %d records", len(data))
    
    # Step 2: Validate
    valid, invalid = validate(data)
    if invalid:
        logger.warning("Found %d invalid records (skipping)", len(invalid))
        for i, record in enumerate(invalid[:5]):
            logger.debug("Invalid record %d: %s", i, record)
    
    # Step 3: Transform
    logger.debug("Transforming %d valid records", len(valid))
    results = transform(valid)
    
    # Step 4: Save
    try:
        save_data(results, output_path)
    except IOError:
        logger.exception("Failed to save results to %s", output_path)
        return False
    
    logger.info("Pipeline complete: %d records written to %s", len(results), output_path)
    return True
```

---

## Summary

- Use `logging` instead of `print()` for any code that persists beyond a quick investigation
- Choose the right level: DEBUG for details, INFO for milestones, WARNING for concerns, ERROR for failures, CRITICAL for catastrophes
- Format logs with timestamps, source locations, and logger names
- Use handlers to direct output to console, files, or both
- Create named loggers per module using `logging.getLogger(__name__)`
- Use lazy formatting (`%s` style) for performance
- Log exceptions with `logger.exception()` to capture tracebacks
- Never log sensitive data (passwords, tokens, personal information)

---

## Exercises

1. Set up basic logging with a custom format showing timestamps and line numbers
2. Create a multi-handler setup: console (INFO+) and file (DEBUG+)
3. Add logging to a data processing function using appropriate levels
4. Configure logging from a dictionary configuration

**Previous**: [Debugging Strategy](./05_Debugging_Strategy.md) | **Next**: [Testing Basics](./07_Testing_Basics.md)
