"""
06 Logging
==========
Demonstrates Python's logging module: levels, formatters,
handlers, named loggers, and configuration.
"""
import logging
import logging.config
import tempfile
import os


def basic_logging():
    """Show basic logging with different levels."""
    print("=== Basic Logging ===")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.debug("Detailed diagnostic information")
    logging.info("Confirmation that things work")
    logging.warning("Something unexpected happened")
    logging.error("Something failed")
    logging.critical("Program cannot continue")
    print()


def named_loggers():
    """Show module-level named loggers."""
    print("=== Named Loggers ===")
    db_logger = logging.getLogger("myapp.database")
    api_logger = logging.getLogger("myapp.api")

    db_logger.info("Connecting to database...")
    api_logger.debug("Request received: GET /users")
    db_logger.warning("Slow query detected (2.5s)")
    api_logger.error("Request timeout after 30s")
    print()


def multi_handler_setup():
    """Demonstrate console + file handler setup."""
    print("=== Multi-Handler Setup ===")
    logger = logging.getLogger("demo_app")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler (INFO+)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # File handler (DEBUG+)
    log_file = os.path.join(tempfile.gettempdir(), "demo_app.log")
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s"
    ))

    logger.addHandler(console)
    logger.addHandler(file_handler)

    logger.debug("This goes to file only")
    logger.info("This goes to both console and file")
    logger.warning("This also goes to both")
    logger.error("And this too")

    print(f"\nLog file written to: {log_file}")
    with open(log_file) as f:
        print("File contents:")
        for line in f:
            print(f"  {line.rstrip()}")
    os.unlink(log_file)
    print()


def exception_logging():
    """Show proper exception logging."""
    print("=== Exception Logging ===")
    logger = logging.getLogger("exception_demo")

    # Bad way
    try:
        result = 1 / 0
    except ZeroDivisionError as e:
        logger.error(f"Bad: {e}")  # Loses traceback

    # Good way
    try:
        result = 1 / 0
    except ZeroDivisionError:
        logger.exception("Good: includes full traceback")
    print()


def lazy_formatting():
    """Show lazy vs eager formatting for performance."""
    print("=== Lazy Formatting ===")
    logger = logging.getLogger("perf_demo")
    logger.setLevel(logging.WARNING)  # DEBUG messages won't be shown

    large_data = list(range(10000))

    # Eager (bad): string is always formatted
    # logger.debug(f"Data: {large_data}")  # Formats even when filtered

    # Lazy (good): only formatted if level is enabled
    logger.debug("Data has %d items", len(large_data))

    print("When level=WARNING, debug messages are filtered.")
    print("Lazy formatting avoids formatting the string at all.")
    print()


def dict_config_demo():
    """Show dictionary-based logging configuration."""
    print("=== Dictionary Config ===")
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {"format": "[%(levelname)s] %(name)s: %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "simple",
            },
        },
        "loggers": {
            "dictconfig_app": {
                "level": "DEBUG",
                "handlers": ["console"],
            },
        },
    }
    logging.config.dictConfig(config)
    logger = logging.getLogger("dictconfig_app")
    logger.info("Configured via dictionary")
    logger.debug("All levels work")
    print()


def practical_pipeline():
    """Show logging in a realistic data pipeline."""
    print("=== Practical Pipeline ===")
    logger = logging.getLogger("pipeline")

    def run_pipeline(data):
        logger.info("Pipeline started with %d records", len(data))

        # Validate
        valid = [d for d in data if isinstance(d, (int, float)) and d >= 0]
        invalid_count = len(data) - len(valid)
        if invalid_count:
            logger.warning("Skipped %d invalid records", invalid_count)

        # Transform
        logger.debug("Transforming %d valid records", len(valid))
        results = [x ** 0.5 for x in valid]

        logger.info("Pipeline complete: %d results", len(results))
        return results

    data = [4, 9, 16, -1, "bad", 25, None, 36]
    results = run_pipeline(data)
    print(f"Results: {[f'{r:.1f}' for r in results]}")
    print()


if __name__ == "__main__":
    basic_logging()
    named_loggers()
    multi_handler_setup()
    exception_logging()
    lazy_formatting()
    dict_config_demo()
    practical_pipeline()
