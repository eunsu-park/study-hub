# Exercises (연습문제 풀이)

Practice problem solutions for all study topics. Each file corresponds to a lesson and contains working solutions to the exercises in that lesson.

각 토픽의 레슨별 연습문제 풀이 코드입니다. 레슨의 Exercises / Practice Problems 섹션에 대한 풀이를 제공합니다.

## Directory Structure

```
exercises/
├── README.md
├── Programming/          # Python
├── Python/               # Python
├── Git/                  # Shell
├── Shell_Script/         # Bash
├── Linux/                # Bash
├── Web_Development/      # HTML/CSS/JS
├── Docker/               # Dockerfile/YAML/Bash
├── LaTeX/                # .tex
├── Cloud_Computing/      # Python
├── VIM/                  # Vim script/text
├── C_Programming/        # C
├── CPP/                  # C++
├── Calculus_and_Differential_Equations/  # Python
├── Data_Science/         # Python
├── Machine_Learning/     # Python
├── PostgreSQL/           # SQL
├── ... (all topics)
└── Space_Weather/        # Python
```

## File Naming Convention

- `{lesson_number}_{snake_case_name}.{ext}` — matches the lesson file it solves
- Example: `01_variables_and_types.py` corresponds to `01_Variables_and_Types.md`

## File Structure

Each exercise file follows this pattern:

```python
"""
Exercises for Lesson XX: Title
Topic: TopicName

Solutions to practice problems from the lesson.
"""

# === Exercise 1: Title ===
# Problem: [problem description]

def exercise_1():
    """Solution"""
    # solution code
    ...

# === Exercise 2: Title ===
# Problem: [problem description]

def exercise_2():
    """Solution"""
    ...

# Verification
if __name__ == "__main__":
    print("=== Exercise 1 ===")
    exercise_1()
    print("\n=== Exercise 2 ===")
    exercise_2()
    print("\nAll exercises completed!")
```

## Excluded Lessons

The following lesson types do NOT have corresponding exercise files:

- `00_Overview.md` — no exercises
- `Impl_*` lessons (Deep_Learning) — implementation IS the exercise
- Project lessons — the project itself serves as practice
- Spanish — language course, not code-based

## Relationship with examples/

| Folder | Purpose | Content |
|--------|---------|---------|
| `examples/` | Concept demonstrations | Working code that illustrates lesson concepts |
| `exercises/` | Practice problem solutions | Solutions to the exercises at the end of each lesson |

## How to Use

1. Study the lesson in `content/en/` or `content/ko/`
2. Attempt the exercises on your own
3. Check your solutions against the files here
4. Run the exercise file to verify: `python exercises/TopicName/01_file.py`

## Language by Topic

| Language | Topics |
|----------|--------|
| Python | Most topics (Programming, ML, DL, Data_Science, etc.) |
| C | C_Programming |
| C++ | CPP |
| SQL | PostgreSQL |
| Bash | Shell_Script, Linux, Git |
| JavaScript | Web_Development |
| LaTeX (.tex) | LaTeX |
