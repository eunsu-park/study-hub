"""Configuration for the Study Viewer application."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
CONTENT_DIR = BASE_DIR.parent / "content"
EXAMPLES_DIR = BASE_DIR.parent / "examples"
EXERCISES_DIR = BASE_DIR.parent / "exercises"

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{BASE_DIR / 'data.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CONTENT_DIR = CONTENT_DIR
    EXAMPLES_DIR = EXAMPLES_DIR
    EXERCISES_DIR = EXERCISES_DIR

    # Language settings
    SUPPORTED_LANGUAGES = ["ko", "en"]
    DEFAULT_LANGUAGE = "ko"
    LANGUAGE_NAMES = {
        "ko": "한국어",
        "en": "English",
    }

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SECRET_KEY = os.environ.get("SECRET_KEY")

config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
