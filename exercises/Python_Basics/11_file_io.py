"""
Exercise 11: File I/O

Practice reading/writing CSV, JSON, and parsing log files.
All exercises use in-memory strings via io.StringIO to avoid actual file ops.
"""

import csv
import json
import io
from collections import defaultdict


SAMPLE_CSV = """name,age,city
Alice,30,Seoul
Bob,25,Tokyo
Charlie,35,London
Diana,28,Seoul
Eve,32,Tokyo"""

SAMPLE_LOG = """2024-01-15 10:23:45 INFO  Server started
2024-01-15 10:24:01 ERROR Database connection failed
2024-01-15 10:24:05 INFO  Retrying connection
2024-01-15 10:24:06 INFO  Database connected
2024-01-15 10:25:30 WARN  High memory usage detected
2024-01-15 10:26:00 ERROR Timeout on request /api/users"""


def parse_csv_data(csv_string):
    """Parse CSV string and return a list of dicts.

    Each dict maps column headers to values.
    Convert 'age' values to integers.

    Args:
        csv_string: CSV formatted string with headers.

    Returns:
        List of dicts, one per row.
    """
    # TODO: Implement using csv.DictReader and io.StringIO
    pass


def filter_by_city(records, city):
    """Filter a list of record dicts by city.

    Args:
        records: List of dicts with a "city" key.
        city: City name to filter by.

    Returns:
        List of records matching the city.
    """
    # TODO: Implement this
    pass


def records_to_json(records, indent=2):
    """Convert a list of record dicts to a JSON string.

    Args:
        records: List of dicts.
        indent: JSON indentation level.

    Returns:
        JSON formatted string.
    """
    # TODO: Implement using json.dumps
    pass


def parse_log_entries(log_string):
    """Parse log entries and return structured data.

    Each log line format: "YYYY-MM-DD HH:MM:SS LEVEL  Message"

    Return a list of dicts with keys: "timestamp", "level", "message"

    Args:
        log_string: Multi-line log string.

    Returns:
        List of log entry dicts.
    """
    # TODO: Implement this
    pass


def count_log_levels(entries):
    """Count occurrences of each log level.

    Args:
        entries: List of log entry dicts (from parse_log_entries).

    Returns:
        Dict mapping level name to count. E.g., {"INFO": 3, "ERROR": 2}
    """
    # TODO: Implement this
    pass


def create_config(app_name, port, debug, features):
    """Create a JSON config string from parameters.

    The JSON structure should be:
    {
        "app": {"name": app_name, "port": port},
        "debug": debug,
        "features": features  (list of strings)
    }

    Args:
        app_name: Application name string.
        port: Port number (int).
        debug: Debug flag (bool).
        features: List of feature name strings.

    Returns:
        JSON string with indent=2.
    """
    # TODO: Implement this
    pass


# === Tests ===

# CSV parsing
records = parse_csv_data(SAMPLE_CSV)
assert len(records) == 5, "5 records"
assert records[0]["name"] == "Alice", "First name"
assert records[0]["age"] == 30, "Age is int"
assert records[1]["city"] == "Tokyo", "Bob's city"

# Filter by city
seoul = filter_by_city(records, "Seoul")
assert len(seoul) == 2, "2 in Seoul"
assert all(r["city"] == "Seoul" for r in seoul), "All Seoul"

# JSON conversion
json_str = records_to_json(records)
parsed_back = json.loads(json_str)
assert len(parsed_back) == 5, "JSON roundtrip"
assert parsed_back[0]["name"] == "Alice", "JSON content"

# Log parsing
entries = parse_log_entries(SAMPLE_LOG)
assert len(entries) == 6, "6 log entries"
assert entries[0]["level"] == "INFO", "First level"
assert entries[1]["level"] == "ERROR", "Second level"
assert entries[1]["message"] == "Database connection failed", "Error message"

# Log level counts
counts = count_log_levels(entries)
assert counts["INFO"] == 3, "3 INFO"
assert counts["ERROR"] == 2, "2 ERROR"
assert counts["WARN"] == 1, "1 WARN"

# Config creation
config_str = create_config("myapp", 8080, True, ["auth", "logging"])
config = json.loads(config_str)
assert config["app"]["name"] == "myapp", "App name"
assert config["app"]["port"] == 8080, "Port"
assert config["debug"] is True, "Debug"
assert config["features"] == ["auth", "logging"], "Features"

print("All tests passed!")
