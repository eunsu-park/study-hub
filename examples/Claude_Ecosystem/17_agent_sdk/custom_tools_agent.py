"""
Claude Agent SDK: Custom Tools Agent Example

Demonstrates building an agent with custom tool definitions
for a specific domain (file analysis and reporting).

Requirements:
    pip install claude-agent-sdk
"""

import claude_code_sdk as sdk
import asyncio
import json
from pathlib import Path
from datetime import datetime


# --- Custom Tool Definitions ---

def analyze_file_stats(directory: str = ".") -> str:
    """Analyze file statistics in a directory."""
    path = Path(directory)
    if not path.is_dir():
        return json.dumps({"error": f"Not a directory: {directory}"})

    stats = {}
    total_size = 0
    total_files = 0

    for f in path.rglob("*"):
        if f.is_file() and not any(
            part.startswith(".") for part in f.relative_to(path).parts
        ):
            ext = f.suffix.lower() or "(no extension)"
            if ext not in stats:
                stats[ext] = {"count": 0, "total_size": 0}
            stats[ext]["count"] += 1
            stats[ext]["total_size"] += f.stat().st_size
            total_files += 1
            total_size += f.stat().st_size

    return json.dumps({
        "directory": str(path.resolve()),
        "total_files": total_files,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "by_extension": dict(
            sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True)
        ),
    }, indent=2)


def find_large_files(directory: str = ".", min_size_kb: int = 100) -> str:
    """Find files larger than a given size."""
    path = Path(directory)
    if not path.is_dir():
        return json.dumps({"error": f"Not a directory: {directory}"})

    large_files = []
    min_size_bytes = min_size_kb * 1024

    for f in path.rglob("*"):
        if f.is_file() and f.stat().st_size >= min_size_bytes:
            large_files.append({
                "path": str(f.relative_to(path)),
                "size_kb": round(f.stat().st_size / 1024, 1),
                "modified": datetime.fromtimestamp(
                    f.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M"),
            })

    large_files.sort(key=lambda x: x["size_kb"], reverse=True)

    return json.dumps({
        "directory": str(path.resolve()),
        "min_size_kb": min_size_kb,
        "count": len(large_files),
        "files": large_files[:20],  # Top 20
    }, indent=2)


def find_duplicates(directory: str = ".") -> str:
    """Find potential duplicate files by size."""
    path = Path(directory)
    if not path.is_dir():
        return json.dumps({"error": f"Not a directory: {directory}"})

    size_map: dict[int, list[str]] = {}

    for f in path.rglob("*"):
        if f.is_file():
            size = f.stat().st_size
            if size > 0:  # Skip empty files
                rel_path = str(f.relative_to(path))
                if size not in size_map:
                    size_map[size] = []
                size_map[size].append(rel_path)

    # Only keep sizes with multiple files (potential duplicates)
    duplicates = {
        str(size): files
        for size, files in size_map.items()
        if len(files) > 1
    }

    return json.dumps({
        "directory": str(path.resolve()),
        "potential_duplicate_groups": len(duplicates),
        "groups": dict(
            sorted(duplicates.items(), key=lambda x: int(x[0]), reverse=True)[:10]
        ),
    }, indent=2)


def generate_report(title: str, sections: list[dict]) -> str:
    """Generate a markdown report."""
    lines = [f"# {title}", f"", f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*", ""]

    for section in sections:
        lines.append(f"## {section.get('heading', 'Section')}")
        lines.append("")
        lines.append(section.get("content", ""))
        lines.append("")

    report = "\n".join(lines)

    # Save to file
    report_path = Path(f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    report_path.write_text(report)

    return json.dumps({
        "saved_to": str(report_path),
        "length": len(report),
        "sections": len(sections),
    })


# --- Tool Registry ---

CUSTOM_TOOLS = {
    "analyze_file_stats": {
        "func": analyze_file_stats,
        "schema": {
            "name": "analyze_file_stats",
            "description": "Analyze file type distribution and sizes in a directory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to analyze (default: current dir)",
                        "default": ".",
                    },
                },
            },
        },
    },
    "find_large_files": {
        "func": find_large_files,
        "schema": {
            "name": "find_large_files",
            "description": "Find files exceeding a minimum size threshold.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "default": "."},
                    "min_size_kb": {
                        "type": "integer",
                        "description": "Minimum file size in KB",
                        "default": 100,
                    },
                },
            },
        },
    },
    "find_duplicates": {
        "func": find_duplicates,
        "schema": {
            "name": "find_duplicates",
            "description": "Find potential duplicate files (same size).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "default": "."},
                },
            },
        },
    },
    "generate_report": {
        "func": generate_report,
        "schema": {
            "name": "generate_report",
            "description": "Generate and save a markdown report.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Report title"},
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "heading": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                        "description": "Report sections with headings and content",
                    },
                },
                "required": ["title", "sections"],
            },
        },
    },
}


# --- Agent Runner ---

async def run_custom_agent():
    """Run the file analysis agent with custom tools."""

    print("File Analysis Agent")
    print("=" * 60)
    print()

    # Use the Agent SDK with custom tools
    result = await sdk.query(
        prompt=(
            "Analyze the current project directory. "
            "1) Show file statistics by type, "
            "2) Find any files larger than 50KB, "
            "3) Check for potential duplicates, "
            "4) Generate a summary report with your findings."
        ),
        options=sdk.QueryOptions(
            max_turns=10,
            system_prompt=(
                "You are a file system analysis agent. "
                "Use the provided tools to analyze directories and generate reports. "
                "Be thorough but concise in your analysis."
            ),
        ),
    )

    for message in result:
        if message.type == "text":
            print(message.content)


# --- Main ---

if __name__ == "__main__":
    asyncio.run(run_custom_agent())
