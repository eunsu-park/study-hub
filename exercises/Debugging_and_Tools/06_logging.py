"""
Exercise 06: Logging

Practice setting up and using Python's logging module
with proper levels, formatters, and handlers.
"""
import logging
import os
import tempfile


def setup_basic_logging():
    """Set up basic logging with a custom format.

    Configure logging with:
    - Level: DEBUG
    - Format: "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s"
    - Date format: "%Y-%m-%d %H:%M:%S"

    Returns:
        logging.Logger: A configured logger named "exercise".
    """
    # TODO: Configure basicConfig and return a named logger
    pass


def setup_multi_handler(log_file_path):
    """Set up logging with both console and file handlers.

    Create a logger with:
    - Console handler: INFO level, format "[%(levelname)s] %(message)s"
    - File handler: DEBUG level, format with timestamp and line number

    Args:
        log_file_path: Path to the log file.

    Returns:
        logging.Logger: A configured logger named "multi_handler".
    """
    # TODO: Create logger with two handlers
    pass


def add_logging_to_processor(data):
    """Add appropriate logging to this data processor.

    Process a list of numbers:
    - Log INFO when starting with the count of items
    - Log DEBUG for each item being processed
    - Log WARNING if any item is negative (skip it)
    - Log ERROR if any item is not a number (skip it)
    - Log INFO when complete with the result

    Args:
        data: A list of values (may contain non-numbers).

    Returns:
        float: Sum of all valid non-negative numbers.
    """
    # TODO: Add logging at appropriate levels
    total = 0
    for item in data:
        if isinstance(item, (int, float)) and item >= 0:
            total += item
    return total


def configure_from_dict():
    """Configure logging using a dictionary configuration.

    Create a dict config with:
    - A "standard" formatter with timestamp
    - A console handler at INFO level
    - A logger named "dictconfig_exercise" at DEBUG level

    Returns:
        logging.Logger: The configured logger.
    """
    # TODO: Create and apply a dictionary configuration
    pass


if __name__ == "__main__":
    # Test setup_basic_logging
    logger = setup_basic_logging()
    assert logger is not None, "Should return a logger"
    assert logger.name == "exercise"
    logger.info("Basic logging test")
    print("setup_basic_logging: PASSED")

    # Test setup_multi_handler
    log_file = os.path.join(tempfile.gettempdir(), "test_exercise.log")
    logger = setup_multi_handler(log_file)
    assert logger is not None
    logger.debug("Debug message (file only)")
    logger.info("Info message (both)")
    logger.error("Error message (both)")
    assert os.path.exists(log_file), "Log file should exist"
    with open(log_file) as f:
        content = f.read()
    assert "Debug message" in content, "File should contain debug messages"
    os.unlink(log_file)
    print("setup_multi_handler: PASSED")

    # Test add_logging_to_processor
    data = [10, 20, -5, "bad", 30, None, 15]
    result = add_logging_to_processor(data)
    assert result == 75, f"Expected 75, got {result}"
    print("add_logging_to_processor: PASSED")

    # Test configure_from_dict
    logger = configure_from_dict()
    assert logger is not None
    assert logger.name == "dictconfig_exercise"
    logger.info("Dict config test")
    print("configure_from_dict: PASSED")
