# CLAUDE.md

This is the project instruction file for Claude Code.

## Project Overview

A web application built with Python/Flask.

## Tech Stack

- **Backend**: Python 3.12, Flask 3.x
- **Database**: PostgreSQL 16 with SQLAlchemy
- **Frontend**: Vanilla JS (ES6+), no framework
- **Testing**: pytest with coverage

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
flask run --port 5000 --debug

# Run tests
python -m pytest tests/ -v --cov=app

# Lint
ruff check . --fix
ruff format .

# Database migrations
flask db upgrade
flask db migrate -m "description"
```

## Code Style

- Python: PEP 8, enforced by ruff
- Max line length: 100
- Use type hints for all function signatures
- Docstrings: Google style

## Project Structure

```
app/
├── __init__.py       # App factory
├── models/           # SQLAlchemy models
├── routes/           # Blueprint routes
├── services/         # Business logic
├── templates/        # Jinja2 templates
└── static/           # CSS/JS assets
tests/
├── conftest.py       # Fixtures
├── test_models.py
├── test_routes.py
└── test_services.py
```

## Important Rules

- Always run tests before committing
- Never commit .env files or secrets
- Use database migrations, never modify schema directly
- All new endpoints must have corresponding tests
