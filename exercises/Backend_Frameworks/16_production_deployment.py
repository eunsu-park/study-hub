# Exercise: Production Deployment
# Practice with Dockerfile generation, health checks, config management, and process supervision.


# Exercise 1: Dockerfile Builder
class DockerfileBuilder:
    """Fluent builder for generating Dockerfiles."""

    def __init__(self):
        self._instructions = []

    def from_image(self, image: str) -> "DockerfileBuilder":
        """Set base image (FROM instruction)."""
        # TODO: Implement
        pass

    def workdir(self, path: str) -> "DockerfileBuilder":
        """Set working directory."""
        # TODO: Implement
        pass

    def copy(self, src: str, dst: str) -> "DockerfileBuilder":
        """Add COPY instruction."""
        # TODO: Implement
        pass

    def run(self, command: str) -> "DockerfileBuilder":
        """Add RUN instruction."""
        # TODO: Implement
        pass

    def expose(self, port: int) -> "DockerfileBuilder":
        """Add EXPOSE instruction."""
        # TODO: Implement
        pass

    def env(self, key: str, value: str) -> "DockerfileBuilder":
        """Add ENV instruction."""
        # TODO: Implement
        pass

    def cmd(self, command: list[str]) -> "DockerfileBuilder":
        """Set CMD instruction (JSON form)."""
        # TODO: Implement
        pass

    def healthcheck(self, command: str, interval: str = "30s") -> "DockerfileBuilder":
        """Add HEALTHCHECK instruction."""
        # TODO: Implement
        pass

    def build(self) -> str:
        """Return the complete Dockerfile as a string."""
        # TODO: Implement
        pass


# Test
# df = (DockerfileBuilder()
#     .from_image("python:3.12-slim")
#     .workdir("/app")
#     .copy("requirements.txt", ".")
#     .run("pip install --no-cache-dir -r requirements.txt")
#     .copy(".", ".")
#     .expose(8000)
#     .cmd(["uvicorn", "main:app", "--host", "0.0.0.0"]))
# text = df.build()
# assert "FROM python:3.12-slim" in text
# assert "EXPOSE 8000" in text


# Exercise 2: Environment Config Loader
class EnvConfig:
    """Load and validate configuration from environment-like dict.

    Supports type casting, defaults, required fields, and prefixes.
    """

    def __init__(self, env: dict, prefix: str = ""):
        self._env = env
        self._prefix = prefix

    def get_str(self, key: str, default: str | None = None) -> str:
        """Get string value. Raise KeyError if required (default=None) and missing."""
        # TODO: Implement (prepend prefix to key)
        pass

    def get_int(self, key: str, default: int | None = None) -> int:
        """Get integer value. Raise ValueError on bad cast."""
        # TODO: Implement
        pass

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean. Truthy values: "true", "1", "yes" (case-insensitive)."""
        # TODO: Implement
        pass

    def get_list(self, key: str, separator: str = ",", default: list | None = None) -> list[str]:
        """Get list by splitting string value."""
        # TODO: Implement
        pass


# Test
# env = {"APP_HOST": "0.0.0.0", "APP_PORT": "8000", "APP_DEBUG": "true", "APP_CORS": "a.com,b.com"}
# cfg = EnvConfig(env, prefix="APP_")
# assert cfg.get_str("HOST") == "0.0.0.0"
# assert cfg.get_int("PORT") == 8000
# assert cfg.get_bool("DEBUG") is True
# assert cfg.get_list("CORS") == ["a.com", "b.com"]


# Exercise 3: Health Check Aggregator
class HealthAggregator:
    """Aggregate multiple health checks into a single status.

    Overall status is "healthy" only if ALL checks pass.
    """

    def __init__(self):
        self._checks = {}  # name -> callable returning (bool, str)

    def register(self, name: str, check_fn):
        """Register a health check function.

        check_fn() -> (is_healthy: bool, detail: str)
        """
        # TODO: Implement
        pass

    def check_all(self) -> dict:
        """Run all checks and return aggregated result.

        Returns: {
            "status": "healthy" | "unhealthy",
            "checks": {
                "name": {"healthy": bool, "detail": str, "duration_ms": float},
                ...
            }
        }
        """
        # TODO: Implement
        pass


# Test
# agg = HealthAggregator()
# agg.register("db", lambda: (True, "connected"))
# agg.register("cache", lambda: (False, "connection refused"))
# result = agg.check_all()
# assert result["status"] == "unhealthy"
# assert result["checks"]["db"]["healthy"] is True
# assert result["checks"]["cache"]["healthy"] is False


# Exercise 4: Graceful Shutdown Coordinator
class ShutdownCoordinator:
    """Coordinate graceful shutdown of multiple services."""

    def __init__(self):
        self._handlers = []  # (priority, name, func) — lower priority runs first

    def register(self, name: str, handler, priority: int = 10):
        """Register a shutdown handler with priority (lower = earlier)."""
        # TODO: Implement
        pass

    def shutdown(self) -> list[dict]:
        """Execute all handlers in priority order.

        Returns: [{"name": str, "success": bool, "error": str|None}, ...]
        """
        # TODO: Implement
        pass


# Test
# coord = ShutdownCoordinator()
# order = []
# coord.register("drain", lambda: order.append("drain"), priority=1)
# coord.register("close_db", lambda: order.append("db"), priority=5)
# coord.register("flush_logs", lambda: order.append("logs"), priority=10)
# results = coord.shutdown()
# assert order == ["drain", "db", "logs"]
# assert all(r["success"] for r in results)


if __name__ == "__main__":
    print("Production Deployment Exercise")
    print("Implement each class/function and verify with the test cases.")
