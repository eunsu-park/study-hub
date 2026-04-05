#!/bin/bash
# Exercises for Lesson 13: CI/CD Integration
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Basic GitHub Actions Workflow ===
# Problem: Create a GitHub Actions workflow that runs pytest on push
# to main and on pull requests. Include Python setup, dependency
# installation, and test execution.
exercise_1() {
    echo "=== Exercise 1: Basic GitHub Actions Workflow ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# .github/workflows/tests.yml
#
# name: Tests
#
# on:
#   push:
#     branches: [main]
#   pull_request:
#     branches: [main]
#
# concurrency:
#   group: tests-${{ github.ref }}
#   cancel-in-progress: true
#
# jobs:
#   test:
#     name: Run Tests
#     runs-on: ubuntu-latest
#
#     steps:
#       - name: Checkout code
#         uses: actions/checkout@v4
#
#       - name: Set up Python
#         uses: actions/setup-python@v5
#         with:
#           python-version: '3.12'
#           cache: 'pip'
#
#       - name: Install dependencies
#         run: |
#           python -m pip install --upgrade pip
#           pip install -r requirements.txt
#           pip install -r requirements-dev.txt
#
#       - name: Run tests
#         run: pytest tests/ -v --tb=short
#
#       - name: Debug on failure
#         if: failure()
#         run: |
#           echo "Python: $(python --version)"
#           pip list
#           pwd && ls -la tests/

# To validate this workflow locally, you can use a Python script
# that parses and checks the YAML structure:

import yaml
from pathlib import Path


def validate_workflow(workflow_path: str) -> list[str]:
    """Validate a GitHub Actions workflow YAML file."""
    errors = []
    path = Path(workflow_path)

    if not path.exists():
        return [f"Workflow file not found: {workflow_path}"]

    with open(path) as f:
        workflow = yaml.safe_load(f)

    # Check required top-level keys
    if "name" not in workflow:
        errors.append("Missing 'name' field")
    if "on" not in workflow:
        errors.append("Missing 'on' trigger configuration")
    if "jobs" not in workflow:
        errors.append("Missing 'jobs' definition")

    # Check that at least one job exists
    jobs = workflow.get("jobs", {})
    if not jobs:
        errors.append("No jobs defined")

    # Check each job has runs-on and steps
    for job_id, job_config in jobs.items():
        if "runs-on" not in job_config:
            errors.append(f"Job '{job_id}' missing 'runs-on'")
        if "steps" not in job_config:
            errors.append(f"Job '{job_id}' missing 'steps'")
        else:
            step_names = [s.get("name", s.get("uses", "unnamed"))
                          for s in job_config["steps"]]
            # Verify checkout step exists
            has_checkout = any("checkout" in str(s) for s in job_config["steps"])
            if not has_checkout:
                errors.append(f"Job '{job_id}' missing checkout step")

    return errors


# Usage:
# errors = validate_workflow(".github/workflows/tests.yml")
# assert not errors, f"Workflow validation failed: {errors}"
SOLUTION
}

# === Exercise 2: Matrix Testing ===
# Problem: Create a workflow with matrix strategy to test on
# Python 3.10, 3.11, and 3.12 across Ubuntu and macOS.
# Exclude Python 3.10 on macOS.
exercise_2() {
    echo "=== Exercise 2: Matrix Testing ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# .github/workflows/matrix-tests.yml
#
# name: Matrix Tests
#
# on:
#   push:
#     branches: [main]
#   pull_request:
#     branches: [main]
#
# jobs:
#   test:
#     name: Python ${{ matrix.python-version }} on ${{ matrix.os }}
#     runs-on: ${{ matrix.os }}
#
#     strategy:
#       fail-fast: false
#       matrix:
#         python-version: ['3.10', '3.11', '3.12']
#         os: [ubuntu-latest, macos-latest]
#         exclude:
#           - python-version: '3.10'
#             os: macos-latest
#
#     steps:
#       - uses: actions/checkout@v4
#
#       - name: Set up Python ${{ matrix.python-version }}
#         uses: actions/setup-python@v5
#         with:
#           python-version: ${{ matrix.python-version }}
#           cache: 'pip'
#
#       - name: Install dependencies
#         run: |
#           python -m pip install --upgrade pip
#           pip install -r requirements.txt
#           pip install -r requirements-dev.txt
#
#       - name: Run tests
#         run: pytest tests/ -v --tb=short
#
#       - name: Run tests with coverage (primary combo only)
#         if: matrix.python-version == '3.12' && matrix.os == 'ubuntu-latest'
#         run: |
#           pytest tests/ --cov=myapp --cov-report=xml:coverage.xml
#
#       - name: Upload coverage
#         if: matrix.python-version == '3.12' && matrix.os == 'ubuntu-latest'
#         uses: actions/upload-artifact@v4
#         with:
#           name: coverage-report
#           path: coverage.xml

# Script to verify expected matrix combinations
import itertools


def compute_matrix_combinations(
    python_versions: list[str],
    os_list: list[str],
    exclude: list[dict] = None,
) -> list[dict]:
    """Compute the actual job matrix after exclusions."""
    exclude = exclude or []
    combos = []
    for py, os_name in itertools.product(python_versions, os_list):
        excluded = any(
            e.get("python-version") == py and e.get("os") == os_name
            for e in exclude
        )
        if not excluded:
            combos.append({"python-version": py, "os": os_name})
    return combos


# Verify our matrix
combos = compute_matrix_combinations(
    python_versions=["3.10", "3.11", "3.12"],
    os_list=["ubuntu-latest", "macos-latest"],
    exclude=[{"python-version": "3.10", "os": "macos-latest"}],
)

assert len(combos) == 5  # 6 total - 1 excluded = 5
assert {"python-version": "3.10", "os": "macos-latest"} not in combos
assert {"python-version": "3.12", "os": "ubuntu-latest"} in combos
print(f"Matrix will create {len(combos)} jobs:")
for c in combos:
    print(f"  Python {c['python-version']} on {c['os']}")
SOLUTION
}

# === Exercise 3: Coverage Gate ===
# Problem: Add a step that fails the build if test coverage drops
# below 80%. Upload the coverage report as an artifact.
exercise_3() {
    echo "=== Exercise 3: Coverage Gate ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# .github/workflows/coverage.yml
#
# name: Coverage Gate
#
# on:
#   push:
#     branches: [main]
#   pull_request:
#     branches: [main]
#
# jobs:
#   coverage:
#     name: Test with Coverage Gate
#     runs-on: ubuntu-latest
#
#     steps:
#       - uses: actions/checkout@v4
#
#       - uses: actions/setup-python@v5
#         with:
#           python-version: '3.12'
#           cache: 'pip'
#
#       - name: Install dependencies
#         run: |
#           python -m pip install --upgrade pip
#           pip install -r requirements.txt
#           pip install -r requirements-dev.txt
#           pip install pytest-cov
#
#       - name: Run tests with coverage
#         run: |
#           pytest tests/ \
#             --cov=myapp \
#             --cov-report=html:coverage-html \
#             --cov-report=xml:coverage.xml \
#             --cov-report=term-missing \
#             --cov-fail-under=80
#
#       - name: Upload coverage HTML report
#         if: always()
#         uses: actions/upload-artifact@v4
#         with:
#           name: coverage-report
#           path: coverage-html/
#           retention-days: 14

# Python helper: parse coverage XML and enforce thresholds
import xml.etree.ElementTree as ET
from pathlib import Path


def check_coverage_threshold(
    coverage_xml: str, threshold: float = 80.0
) -> tuple[bool, float, list[dict]]:
    """Parse coverage.xml and check against a threshold.

    Returns:
        (passes, overall_rate, low_coverage_files)
    """
    tree = ET.parse(coverage_xml)
    root = tree.getroot()

    # Overall line rate is on the root <coverage> element
    overall_rate = float(root.attrib.get("line-rate", 0)) * 100

    # Find files below threshold
    low_files = []
    for package in root.findall(".//package"):
        for cls in package.findall(".//class"):
            file_rate = float(cls.attrib.get("line-rate", 0)) * 100
            if file_rate < threshold:
                low_files.append({
                    "filename": cls.attrib["filename"],
                    "coverage": round(file_rate, 1),
                })

    passes = overall_rate >= threshold
    return passes, round(overall_rate, 1), low_files


# Usage in a test:
def test_coverage_meets_threshold():
    """Verify coverage meets the 80% minimum."""
    xml_path = "coverage.xml"
    if not Path(xml_path).exists():
        return  # Skip if no coverage data

    passes, rate, low_files = check_coverage_threshold(xml_path, 80.0)

    if not passes:
        msg = f"Overall coverage {rate}% is below 80% threshold.\n"
        if low_files:
            msg += "Files below threshold:\n"
            for f in sorted(low_files, key=lambda x: x["coverage"]):
                msg += f"  {f['filename']}: {f['coverage']}%\n"
        raise AssertionError(msg)
SOLUTION
}

# === Exercise 4: Parallel Test Splitting ===
# Problem: Configure CI to split a large test suite across multiple
# parallel jobs using pytest-split, and combine the results.
exercise_4() {
    echo "=== Exercise 4: Parallel Test Splitting ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# .github/workflows/parallel-tests.yml
#
# name: Parallel Tests
#
# on:
#   push:
#     branches: [main]
#   pull_request:
#     branches: [main]
#
# jobs:
#   test:
#     name: Test Shard ${{ matrix.shard }}/${{ strategy.job-total }}
#     runs-on: ubuntu-latest
#
#     strategy:
#       fail-fast: false
#       matrix:
#         shard: [1, 2, 3, 4]
#
#     steps:
#       - uses: actions/checkout@v4
#
#       - uses: actions/setup-python@v5
#         with:
#           python-version: '3.12'
#           cache: 'pip'
#
#       - name: Install dependencies
#         run: |
#           pip install -r requirements.txt -r requirements-dev.txt
#           pip install pytest-split pytest-xdist
#
#       - name: Run test shard
#         run: |
#           pytest tests/ \
#             --splits 4 \
#             --group ${{ matrix.shard }} \
#             --splitting-algorithm least_duration \
#             --junitxml=results-shard-${{ matrix.shard }}.xml \
#             -v
#
#       - name: Upload shard results
#         if: always()
#         uses: actions/upload-artifact@v4
#         with:
#           name: test-results-shard-${{ matrix.shard }}
#           path: results-shard-${{ matrix.shard }}.xml
#
#   combine-results:
#     name: Combine Test Results
#     needs: test
#     if: always()
#     runs-on: ubuntu-latest
#
#     steps:
#       - name: Download all shard results
#         uses: actions/download-artifact@v4
#         with:
#           pattern: test-results-shard-*
#           merge-multiple: true
#
#       - name: Display combined results
#         run: |
#           echo "=== Test Results Summary ==="
#           for f in results-shard-*.xml; do
#             echo "--- $f ---"
#             python3 -c "
#           import xml.etree.ElementTree as ET
#           tree = ET.parse('$f')
#           suite = tree.getroot()
#           tests = suite.attrib.get('tests', 0)
#           fails = suite.attrib.get('failures', 0)
#           errors = suite.attrib.get('errors', 0)
#           print(f'  Tests: {tests}, Failures: {fails}, Errors: {errors}')
#           "
#           done

# Python script to generate .pytest_splits timing data
import json
from pathlib import Path


def generate_split_timings(junit_xml_dir: str, output: str = ".pytest_splits"):
    """Parse JUnit XML results and generate timing data for pytest-split."""
    import xml.etree.ElementTree as ET

    timings = {}
    xml_dir = Path(junit_xml_dir)

    for xml_file in xml_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        for testcase in tree.findall(".//testcase"):
            name = f"{testcase.attrib.get('classname', '')}::{testcase.attrib['name']}"
            time_taken = float(testcase.attrib.get("time", 0))
            timings[name] = time_taken

    with open(output, "w") as f:
        json.dump(timings, f, indent=2)

    print(f"Generated {output} with {len(timings)} test timings")
    return timings
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 13: CI/CD Integration"
echo "======================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
