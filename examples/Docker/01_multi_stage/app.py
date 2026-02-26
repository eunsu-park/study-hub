"""
Simple Flask application demonstrating containerization best practices.

This application provides:
- A basic hello world endpoint
- A health check endpoint for container orchestration
- Proper logging and error handling
"""

from flask import Flask, jsonify
import logging
import os
import signal
import sys

# Why: Containers write logs to stdout/stderr — structured logging with timestamps lets log aggregators (ELK, Loki) parse and correlate events across services
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/')
def hello():
    """Main endpoint returning a simple greeting."""
    logger.info("Hello endpoint accessed")
    return jsonify({
        'message': 'Hello from Docker!',
        'version': '1.0',
        'environment': os.getenv('ENVIRONMENT', 'development')
    })


@app.route('/health')
def health():
    """
    Health check endpoint.

    This is used by:
    - Docker HEALTHCHECK directive
    - Kubernetes liveness/readiness probes
    - Load balancers

    Returns:
        JSON response with health status
    """
    return jsonify({
        'status': 'healthy',
        'service': 'flask-app'
    }), 200


def graceful_shutdown(signum, frame):
    """
    Handle graceful shutdown on SIGTERM.

    This is important in container environments where the orchestrator
    sends SIGTERM before forcefully killing the container.
    """
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


# Why: Docker sends SIGTERM on `docker stop`; without a handler the process ignores it and gets SIGKILL after the timeout, losing in-flight requests
# Register signal handlers for graceful shutdown
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)


if __name__ == '__main__':
# Why: Environment variables are the 12-factor way to inject config — they work identically in Docker, K8s, and CI, avoiding baked-in config files
    # Get configuration from environment variables
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting Flask application on port {port}")

    # Why: Binding to 0.0.0.0 is required inside containers — 127.0.0.1 only accepts loopback traffic, making the app unreachable through Docker's port mapping
    app.run(host='0.0.0.0', port=port, debug=debug)
