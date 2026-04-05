# Exercise: Django Advanced
# Practice with Django Channels, Celery tasks, signals, and custom middleware.

import json
from datetime import datetime


# Exercise 1: Celery Task Retry Logic
# Implement a function that simulates Celery retry behavior with exponential backoff.

def retry_task(
    func,
    args: tuple = (),
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict:
    """Execute func with retry logic and exponential backoff.

    Args:
        func: Callable that may raise exceptions
        args: Arguments to pass to func
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds (actual delay = base_delay * 2^attempt)

    Returns:
        {
            "success": bool,
            "result": <return value or None>,
            "attempts": int,
            "errors": [str, ...],  # error messages from each failed attempt
        }
    """
    # TODO: Implement
    pass


# Test
def flaky_fn(threshold=2, _state={"calls": 0}):
    _state["calls"] += 1
    if _state["calls"] < threshold:
        raise ConnectionError("DB unavailable")
    return "done"


# result = retry_task(flaky_fn, args=(2,), max_retries=3, base_delay=0.01)
# assert result["success"] is True
# assert result["attempts"] == 2


# Exercise 2: Django Signal Dispatcher
# Implement a minimal signal/receiver system (like django.dispatch.Signal).

class Signal:
    """Minimal signal dispatcher."""

    def __init__(self):
        self._receivers = []

    def connect(self, receiver, sender=None):
        """Register a receiver function, optionally filtering by sender.

        receiver signature: receiver(sender, **kwargs)
        """
        # TODO: Implement
        pass

    def disconnect(self, receiver):
        """Remove a receiver."""
        # TODO: Implement
        pass

    def send(self, sender, **kwargs) -> list:
        """Call all connected receivers. Return list of (receiver, result) tuples.

        Only call receivers where sender matches (or receiver has sender=None).
        """
        # TODO: Implement
        pass


# Test
# post_save = Signal()
# results = []
# def on_save(sender, **kwargs):
#     results.append(f"{sender}:{kwargs.get('instance')}")
# post_save.connect(on_save, sender="User")
# post_save.send("User", instance="alice")
# post_save.send("Post", instance="hello")  # should not trigger
# assert results == ["User:alice"]


# Exercise 3: Custom Middleware Pipeline
# Implement a WSGI-like middleware chain.

class MiddlewarePipeline:
    """Chain middleware functions that process request/response."""

    def __init__(self):
        self._middlewares = []

    def add(self, middleware_fn):
        """Add middleware. Signature: middleware_fn(request, call_next) -> response.

        call_next(request) invokes the next middleware or final handler.
        """
        # TODO: Implement
        pass

    def execute(self, request: dict, handler) -> dict:
        """Execute the middleware chain ending with handler(request) -> response."""
        # TODO: Implement
        pass


# Test
# pipeline = MiddlewarePipeline()
# def timing_mw(req, call_next):
#     import time; start = time.time()
#     resp = call_next(req)
#     resp["duration_ms"] = round((time.time() - start) * 1000)
#     return resp
# def auth_mw(req, call_next):
#     if not req.get("user"):
#         return {"status": 401, "body": "Unauthorized"}
#     return call_next(req)
# pipeline.add(timing_mw)
# pipeline.add(auth_mw)
# resp = pipeline.execute({"user": "alice"}, lambda r: {"status": 200, "body": "OK"})
# assert resp["status"] == 200
# assert "duration_ms" in resp


# Exercise 4: WebSocket Message Router
# Implement a message type router for Django Channels-style WebSocket handling.

class WebSocketRouter:
    """Route WebSocket messages by 'type' field."""

    def __init__(self):
        self._handlers = {}

    def on(self, message_type: str):
        """Decorator to register a handler for a message type.

        Handler signature: handler(message: dict) -> dict
        """
        # TODO: Implement
        pass

    def dispatch(self, raw_message: str) -> dict:
        """Parse JSON message, route to handler, return response.

        Returns {"error": "..."} if type is unknown or JSON is invalid.
        """
        # TODO: Implement
        pass


# Test
# ws = WebSocketRouter()
# @ws.on("chat.message")
# def handle_chat(msg):
#     return {"type": "chat.response", "text": f"Echo: {msg['text']}"}
# resp = ws.dispatch('{"type":"chat.message","text":"hello"}')
# assert resp["text"] == "Echo: hello"


if __name__ == "__main__":
    print("Django Advanced Exercise")
    print("Implement each class/function and verify with the test cases.")
