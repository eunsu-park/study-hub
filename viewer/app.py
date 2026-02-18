"""Study Materials Web Viewer - Flask Application with Multi-language Support."""
import os
from datetime import datetime, timezone
from pathlib import Path
from functools import wraps

from flask import Flask, render_template, request, jsonify, abort, redirect, url_for, make_response
from models import db, LessonRead, Bookmark
from config import Config
from utils.markdown_parser import parse_markdown, extract_excerpt
from utils.search import search, build_search_index, create_fts_table

app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
db.init_app(app)

CONTENT_DIR = Config.CONTENT_DIR
SUPPORTED_LANGS = set(Config.SUPPORTED_LANGUAGES)
DEFAULT_LANG = Config.DEFAULT_LANGUAGE
LANGUAGE_NAMES = Config.LANGUAGE_NAMES


@app.template_filter("timeago")
def timeago_filter(dt):
    """Format a datetime as a relative time string."""
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    diff = now - dt
    seconds = diff.total_seconds()
    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    elif seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    elif seconds < 604800:
        return f"{int(seconds // 86400)}d ago"
    else:
        return dt.strftime("%Y-%m-%d")


def get_content_dir(lang: str) -> Path:
    """Get content directory for a specific language."""
    return CONTENT_DIR / lang


def validate_lang(f):
    """Decorator to validate language parameter."""
    @wraps(f)
    def decorated_function(lang, *args, **kwargs):
        if lang not in SUPPORTED_LANGS:
            abort(404)
        return f(lang, *args, **kwargs)
    return decorated_function


def get_topics(lang: str) -> list[dict]:
    """Get list of all topics with lesson counts for a language."""
    content_dir = get_content_dir(lang)
    if not content_dir.exists():
        return []

    topics = []
    for topic_dir in sorted(content_dir.iterdir()):
        if not topic_dir.is_dir() or topic_dir.name.startswith("."):
            continue
        lessons = list(topic_dir.glob("*.md"))
        topics.append({
            "name": topic_dir.name,
            "lesson_count": len(lessons),
            "display_name": topic_dir.name.replace("_", " "),
        })
    return topics


def get_lessons(lang: str, topic: str) -> list[dict]:
    """Get list of lessons for a topic in a language."""
    topic_dir = get_content_dir(lang) / topic
    if not topic_dir.exists():
        return []

    lessons = []
    for md_file in sorted(topic_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        title = extract_title_from_content(content) or md_file.stem
        lessons.append({
            "filename": md_file.name,
            "title": title,
            "display_name": md_file.stem.replace("_", " "),
        })
    return lessons


def extract_title_from_content(content: str) -> str:
    """Extract first H1 from content."""
    import re
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else ""


def get_progress(lang: str, topic: str) -> dict:
    """Get reading progress for a topic in a language."""
    lessons = get_lessons(lang, topic)
    total = len(lessons)
    read_count = LessonRead.query.filter_by(language=lang, topic=topic).count()
    return {
        "total": total,
        "read": read_count,
        "percentage": round(read_count / total * 100) if total > 0 else 0,
    }


def is_lesson_read(lang: str, topic: str, filename: str) -> bool:
    """Check if a lesson has been read."""
    return LessonRead.query.filter_by(language=lang, topic=topic, filename=filename).first() is not None


def is_bookmarked(lang: str, topic: str, filename: str) -> bool:
    """Check if a lesson is bookmarked."""
    return Bookmark.query.filter_by(language=lang, topic=topic, filename=filename).first() is not None


def get_available_languages() -> list[dict]:
    """Get list of available languages with their names."""
    return [{"code": lang, "name": LANGUAGE_NAMES.get(lang, lang)} for lang in SUPPORTED_LANGS]


# Routes
@app.route("/")
def root():
    """Root - redirect to default language."""
    lang = request.cookies.get("lang", DEFAULT_LANG)
    if lang not in SUPPORTED_LANGS:
        lang = DEFAULT_LANG
    return redirect(url_for("index", lang=lang))


@app.route("/<lang>/")
@validate_lang
def index(lang: str):
    """Home page - learning hub with stats, recent activity, and topics grid."""
    topics = get_topics(lang)
    for topic in topics:
        topic["progress"] = get_progress(lang, topic["name"])

    # Overall progress
    total_lessons = sum(t["lesson_count"] for t in topics)
    total_read = LessonRead.query.filter_by(language=lang).count()
    overall = {
        "total": total_lessons,
        "read": total_read,
        "percentage": round(total_read / total_lessons * 100) if total_lessons > 0 else 0,
    }

    # Continue learning (1~99% progress topics, max 5)
    in_progress = [t for t in topics if 0 < t["progress"]["percentage"] < 100]
    in_progress.sort(key=lambda t: t["progress"]["read"], reverse=True)

    # Recently read lessons (latest 5)
    recent_reads = LessonRead.query.filter_by(language=lang) \
        .order_by(LessonRead.read_at.desc()).limit(5).all()
    recent_items = []
    for r in recent_reads:
        lessons = get_lessons(lang, r.topic)
        info = next((l for l in lessons if l["filename"] == r.filename), None)
        if info:
            recent_items.append({
                "topic": r.topic, "filename": r.filename,
                "title": info["title"], "read_at": r.read_at,
            })

    # Recent bookmarks (latest 5)
    recent_bookmarks = Bookmark.query.filter_by(language=lang) \
        .order_by(Bookmark.created_at.desc()).limit(5).all()
    bookmark_items = []
    for b in recent_bookmarks:
        lessons = get_lessons(lang, b.topic)
        info = next((l for l in lessons if l["filename"] == b.filename), None)
        if info:
            bookmark_items.append({
                "topic": b.topic, "filename": b.filename,
                "title": info["title"],
            })

    bookmark_count = Bookmark.query.filter_by(language=lang).count()

    response = make_response(render_template(
        "index.html",
        topics=topics,
        lang=lang,
        languages=get_available_languages(),
        overall=overall,
        in_progress=in_progress[:5],
        recent_reads=recent_items,
        bookmarks=bookmark_items,
        bookmark_count=bookmark_count,
    ))
    response.set_cookie("lang", lang, max_age=60*60*24*365)
    return response


@app.route("/<lang>/topic/<name>")
@validate_lang
def topic(lang: str, name: str):
    """Topic page - list lessons."""
    topic_dir = get_content_dir(lang) / name
    if not topic_dir.exists():
        abort(404)

    lessons = get_lessons(lang, name)
    for lesson in lessons:
        lesson["is_read"] = is_lesson_read(lang, name, lesson["filename"])
        lesson["is_bookmarked"] = is_bookmarked(lang, name, lesson["filename"])

    progress = get_progress(lang, name)
    return render_template(
        "topic.html",
        topic=name,
        display_name=name.replace("_", " "),
        lessons=lessons,
        progress=progress,
        lang=lang,
        languages=get_available_languages(),
    )


@app.route("/<lang>/topic/<name>/lesson/<filename>")
@validate_lang
def lesson(lang: str, name: str, filename: str):
    """Lesson page - render markdown content."""
    filepath = get_content_dir(lang) / name / filename
    if not filepath.exists():
        abort(404)

    content = filepath.read_text(encoding="utf-8")
    parsed = parse_markdown(content)

    # Get prev/next lessons for navigation
    lessons = get_lessons(lang, name)
    current_idx = next(
        (i for i, l in enumerate(lessons) if l["filename"] == filename), -1
    )
    prev_lesson = lessons[current_idx - 1] if current_idx > 0 else None
    next_lesson = lessons[current_idx + 1] if current_idx < len(lessons) - 1 else None

    return render_template(
        "lesson.html",
        topic=name,
        filename=filename,
        title=parsed["title"] or filename,
        content=parsed["html"],
        toc=parsed["toc"],
        is_read=is_lesson_read(lang, name, filename),
        is_bookmarked=is_bookmarked(lang, name, filename),
        prev_lesson=prev_lesson,
        next_lesson=next_lesson,
        lang=lang,
        languages=get_available_languages(),
    )


@app.route("/<lang>/search")
@validate_lang
def search_page(lang: str):
    """Search page."""
    query = request.args.get("q", "")
    results = []
    if query:
        db_path = Path(app.config["SQLALCHEMY_DATABASE_URI"].replace("sqlite:///", ""))
        results = search(db_path, query, lang=lang)
    return render_template(
        "search.html",
        query=query,
        results=results,
        lang=lang,
        languages=get_available_languages(),
    )


@app.route("/<lang>/dashboard")
@validate_lang
def dashboard(lang: str):
    """Dashboard page - overall progress."""
    topics = get_topics(lang)
    for topic in topics:
        topic["progress"] = get_progress(lang, topic["name"])

    # Calculate overall progress
    total_lessons = sum(t["lesson_count"] for t in topics)
    total_read = LessonRead.query.filter_by(language=lang).count()
    overall = {
        "total": total_lessons,
        "read": total_read,
        "percentage": round(total_read / total_lessons * 100) if total_lessons > 0 else 0,
    }

    return render_template(
        "dashboard.html",
        topics=topics,
        overall=overall,
        lang=lang,
        languages=get_available_languages(),
    )


@app.route("/<lang>/bookmarks")
@validate_lang
def bookmarks(lang: str):
    """Bookmarks page."""
    bookmarked = Bookmark.query.filter_by(language=lang).order_by(Bookmark.created_at.desc()).all()
    items = []
    for bm in bookmarked:
        lessons = get_lessons(lang, bm.topic)
        lesson_info = next((l for l in lessons if l["filename"] == bm.filename), None)
        if lesson_info:
            items.append({
                "topic": bm.topic,
                "filename": bm.filename,
                "title": lesson_info["title"],
                "created_at": bm.created_at,
            })
    return render_template(
        "bookmarks.html",
        bookmarks=items,
        lang=lang,
        languages=get_available_languages(),
    )


# API Routes
@app.route("/api/mark-read", methods=["POST"])
def api_mark_read():
    """Mark a lesson as read/unread."""
    data = request.get_json()
    lang = data.get("lang", DEFAULT_LANG)
    topic = data.get("topic")
    filename = data.get("filename")
    is_read = data.get("is_read", True)

    if not topic or not filename:
        return jsonify({"error": "Missing topic or filename"}), 400

    existing = LessonRead.query.filter_by(language=lang, topic=topic, filename=filename).first()

    if is_read and not existing:
        lesson_read = LessonRead(language=lang, topic=topic, filename=filename)
        db.session.add(lesson_read)
        db.session.commit()
    elif not is_read and existing:
        db.session.delete(existing)
        db.session.commit()

    return jsonify({"success": True, "is_read": is_read})


@app.route("/api/bookmark", methods=["POST"])
def api_bookmark():
    """Add or remove a bookmark."""
    data = request.get_json()
    lang = data.get("lang", DEFAULT_LANG)
    topic = data.get("topic")
    filename = data.get("filename")

    if not topic or not filename:
        return jsonify({"error": "Missing topic or filename"}), 400

    existing = Bookmark.query.filter_by(language=lang, topic=topic, filename=filename).first()

    if existing:
        db.session.delete(existing)
        db.session.commit()
        return jsonify({"success": True, "bookmarked": False})
    else:
        bookmark = Bookmark(language=lang, topic=topic, filename=filename)
        db.session.add(bookmark)
        db.session.commit()
        return jsonify({"success": True, "bookmarked": True})


@app.route("/api/clear-user-data", methods=["POST"])
def api_clear_user_data():
    """Delete all reading history and bookmarks."""
    deleted_reads = LessonRead.query.delete()
    deleted_bookmarks = Bookmark.query.delete()
    db.session.commit()
    return jsonify({
        "success": True,
        "deleted_reads": deleted_reads,
        "deleted_bookmarks": deleted_bookmarks,
    })


@app.route("/api/search")
def api_search():
    """Search API endpoint."""
    query = request.args.get("q", "")
    lang = request.args.get("lang", DEFAULT_LANG)
    if not query:
        return jsonify({"results": []})

    db_path = Path(app.config["SQLALCHEMY_DATABASE_URI"].replace("sqlite:///", ""))
    results = search(db_path, query, lang=lang)
    return jsonify({"results": results})


@app.route("/api/set-language", methods=["POST"])
def api_set_language():
    """Set preferred language."""
    data = request.get_json()
    lang = data.get("lang", DEFAULT_LANG)

    if lang not in SUPPORTED_LANGS:
        return jsonify({"error": "Unsupported language"}), 400

    response = jsonify({"success": True, "lang": lang})
    response.set_cookie("lang", lang, max_age=60*60*24*365)
    return response


# CLI Commands
@app.cli.command("init-db")
def init_db():
    """Initialize the database."""
    with app.app_context():
        db.create_all()
        db_path = Path(app.config["SQLALCHEMY_DATABASE_URI"].replace("sqlite:///", ""))
        create_fts_table(db_path)
        print("Database initialized.")


@app.cli.command("build-index")
def build_index():
    """Build the search index for all languages."""
    db_path = Path(app.config["SQLALCHEMY_DATABASE_URI"].replace("sqlite:///", ""))
    for lang in SUPPORTED_LANGS:
        lang_content_dir = get_content_dir(lang)
        if lang_content_dir.exists():
            print(f"Building index for {lang}...")
            build_search_index(lang_content_dir, db_path, lang=lang)
    print("Search index built for all languages.")


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
