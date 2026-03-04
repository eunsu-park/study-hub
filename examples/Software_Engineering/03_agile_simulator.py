"""
Agile and Iterative Development Simulator

Demonstrates:
1. User Stories with estimation (story points)
2. Sprint Planning and Backlog management
3. Kanban Board with WIP limits
4. Velocity and Burndown tracking
5. Retrospective action items

Theory:
- Scrum: Fixed-length sprints, committed backlog, velocity-based planning.
- Kanban: Continuous flow, WIP limits, cycle time optimization.
- Both use empirical process control: inspect and adapt.

Adapted from Software Engineering Lesson 03.
"""

from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


# ─────────────────────────────────────────────────
# 1. USER STORIES
# ─────────────────────────────────────────────────

@dataclass
class UserStory:
    """A user story with INVEST criteria."""
    id: str
    role: str
    action: str
    benefit: str
    points: int
    acceptance_criteria: list[str] = field(default_factory=list)
    status: str = "backlog"  # backlog / todo / in_progress / review / done

    @property
    def description(self) -> str:
        return f"As a {self.role}, I want to {self.action} so that {self.benefit}."

    def check_invest(self) -> dict[str, bool]:
        """Check INVEST criteria (simplified heuristic)."""
        return {
            "Independent": not self.id.endswith("-dep"),
            "Negotiable": True,  # assumed if story format used
            "Valuable": bool(self.benefit),
            "Estimable": self.points > 0,
            "Small": self.points <= 8,
            "Testable": len(self.acceptance_criteria) > 0,
        }


# ─────────────────────────────────────────────────
# 2. SPRINT PLANNING
# ─────────────────────────────────────────────────

@dataclass
class Sprint:
    """A Scrum sprint with committed stories."""
    number: int
    capacity: int  # story points
    stories: list[UserStory] = field(default_factory=list)

    @property
    def committed_points(self) -> int:
        return sum(s.points for s in self.stories)

    @property
    def completed_points(self) -> int:
        return sum(s.points for s in self.stories if s.status == "done")

    @property
    def remaining_points(self) -> int:
        return self.committed_points - self.completed_points

    def add_story(self, story: UserStory) -> bool:
        """Add story if capacity allows."""
        if self.committed_points + story.points <= self.capacity:
            story.status = "todo"
            self.stories.append(story)
            return True
        return False

    def complete_story(self, story_id: str):
        """Mark a story as done."""
        for s in self.stories:
            if s.id == story_id:
                s.status = "done"
                return True
        return False


# ─────────────────────────────────────────────────
# 3. KANBAN BOARD
# ─────────────────────────────────────────────────

class KanbanBoard:
    """Kanban board with WIP limits."""

    def __init__(self, wip_limits: dict[str, int]):
        self.columns = ["backlog", "todo", "in_progress", "review", "done"]
        self.wip_limits = wip_limits  # column -> max items
        self.items: list[UserStory] = []

    def add_item(self, story: UserStory):
        self.items.append(story)

    def move(self, story_id: str, to_column: str) -> bool:
        """Move item to column, respecting WIP limits."""
        story = next((s for s in self.items if s.id == story_id), None)
        if not story:
            return False

        # Check WIP limit
        limit = self.wip_limits.get(to_column, float("inf"))
        current = sum(1 for s in self.items if s.status == to_column)
        if current >= limit:
            return False

        story.status = to_column
        return True

    def display(self) -> str:
        """Render board as ASCII."""
        lines = []
        col_width = 22
        header = "  ".join(f"{c.upper():<{col_width}}" for c in self.columns)
        lines.append(header)
        lines.append("─" * len(header))

        by_col = defaultdict(list)
        for s in self.items:
            by_col[s.status].append(s)

        max_rows = max((len(v) for v in by_col.values()), default=0)
        for row in range(max_rows):
            cells = []
            for col in self.columns:
                col_items = by_col[col]
                if row < len(col_items):
                    s = col_items[row]
                    label = f"[{s.id}] {s.points}pt"
                    cells.append(f"{label:<{col_width}}")
                else:
                    cells.append(" " * col_width)
            lines.append("  ".join(cells))

        # WIP status
        lines.append("")
        wip_parts = []
        for col in self.columns:
            count = len(by_col[col])
            limit = self.wip_limits.get(col, "∞")
            wip_parts.append(f"{col}: {count}/{limit}")
        lines.append("WIP: " + "  |  ".join(wip_parts))

        return "\n".join(lines)


# ─────────────────────────────────────────────────
# 4. VELOCITY TRACKER
# ─────────────────────────────────────────────────

class VelocityTracker:
    """Track team velocity across sprints."""

    def __init__(self):
        self.sprints: list[dict] = []

    def record_sprint(self, number: int, committed: int, completed: int):
        self.sprints.append({
            "sprint": number,
            "committed": committed,
            "completed": completed,
        })

    @property
    def average_velocity(self) -> float:
        if not self.sprints:
            return 0
        return sum(s["completed"] for s in self.sprints) / len(self.sprints)

    def forecast_sprints(self, remaining_points: int) -> int:
        """Forecast how many sprints to complete remaining work."""
        avg = self.average_velocity
        if avg <= 0:
            return -1
        return -(-remaining_points // int(avg))  # ceiling division

    def burndown(self, total_points: int) -> list[dict]:
        """Generate sprint-level burndown data."""
        remaining = total_points
        data = [{"sprint": 0, "remaining": remaining}]
        for s in self.sprints:
            remaining -= s["completed"]
            data.append({"sprint": s["sprint"], "remaining": max(0, remaining)})
        return data


# ─────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────

def demo_user_stories():
    """Create and display user stories."""
    print("=" * 65)
    print("  User Stories")
    print("=" * 65)

    stories = [
        UserStory("US-1", "customer", "search products by category",
                  "I can find what I need quickly", 3,
                  ["Given categories exist, When I select one, Then I see matching products"]),
        UserStory("US-2", "customer", "add items to cart",
                  "I can purchase multiple items at once", 5,
                  ["Cart shows item count", "Cart persists across sessions"]),
        UserStory("US-3", "admin", "view sales dashboard",
                  "I can track revenue and trends", 8,
                  ["Shows daily/weekly/monthly revenue", "Exportable to CSV"]),
        UserStory("US-4", "customer", "checkout with saved payment",
                  "I don't have to re-enter card details", 5,
                  ["PCI-compliant storage", "One-click checkout"]),
        UserStory("US-5", "admin", "manage product inventory",
                  "stock levels are always accurate", 3,
                  ["Low stock alerts", "Bulk import from CSV"]),
    ]

    for s in stories:
        print(f"\n  [{s.id}] {s.points} pts")
        print(f"  {s.description}")
        invest = s.check_invest()
        passed = sum(invest.values())
        print(f"  INVEST: {passed}/6 — {', '.join(k for k, v in invest.items() if v)}")

    return stories


def demo_sprint(stories):
    """Simulate sprint planning and execution."""
    print("\n" + "=" * 65)
    print("  Sprint 1 Planning (Capacity: 13 points)")
    print("=" * 65)

    sprint = Sprint(number=1, capacity=13)
    for s in stories:
        added = sprint.add_story(s)
        status = "Added" if added else "Deferred (over capacity)"
        print(f"  {s.id} ({s.points} pts): {status}")

    print(f"\n  Committed: {sprint.committed_points} / {sprint.capacity} pts")

    # Simulate completing some stories
    sprint.complete_story("US-1")
    sprint.complete_story("US-2")
    print(f"  Completed: {sprint.completed_points} pts | "
          f"Remaining: {sprint.remaining_points} pts")

    return sprint


def demo_kanban():
    """Demonstrate Kanban board with WIP limits."""
    print("\n" + "=" * 65)
    print("  Kanban Board (WIP limits: in_progress=2, review=2)")
    print("=" * 65)

    board = KanbanBoard(wip_limits={"in_progress": 2, "review": 2})
    items = [
        UserStory("K-1", "", "", "", 3, status="done"),
        UserStory("K-2", "", "", "", 5, status="review"),
        UserStory("K-3", "", "", "", 2, status="in_progress"),
        UserStory("K-4", "", "", "", 3, status="in_progress"),
        UserStory("K-5", "", "", "", 5, status="todo"),
        UserStory("K-6", "", "", "", 8, status="backlog"),
    ]
    for item in items:
        board.add_item(item)

    print(f"\n{board.display()}")

    # Try to move K-5 to in_progress (should fail: WIP limit reached)
    moved = board.move("K-5", "in_progress")
    print(f"\n  Move K-5 to in_progress: {'OK' if moved else 'BLOCKED (WIP limit)'}")


def demo_velocity():
    """Track velocity and forecast."""
    print("\n" + "=" * 65)
    print("  Velocity Tracking and Forecast")
    print("=" * 65)

    tracker = VelocityTracker()
    sprint_data = [(1, 13, 8), (2, 13, 11), (3, 13, 13), (4, 15, 12), (5, 15, 14)]

    print(f"\n  {'Sprint':<8} {'Committed':>10} {'Completed':>10} {'Ratio':>8}")
    print(f"  {'─' * 38}")
    for num, committed, completed in sprint_data:
        tracker.record_sprint(num, committed, completed)
        ratio = completed / committed * 100
        print(f"  {num:<8} {committed:>10} {completed:>10} {ratio:>7.0f}%")

    remaining = 45
    forecast = tracker.forecast_sprints(remaining)
    print(f"\n  Average velocity: {tracker.average_velocity:.1f} pts/sprint")
    print(f"  Remaining backlog: {remaining} pts")
    print(f"  Forecast: ~{forecast} sprints to complete")

    # Burndown
    total = 80
    burndown = tracker.burndown(total)
    print(f"\n  Burndown (total {total} pts):")
    for b in burndown:
        bar = "█" * (b["remaining"] // 2)
        print(f"  Sprint {b['sprint']}: {b['remaining']:>3} pts {bar}")


if __name__ == "__main__":
    stories = demo_user_stories()
    demo_sprint(stories)
    demo_kanban()
    demo_velocity()
