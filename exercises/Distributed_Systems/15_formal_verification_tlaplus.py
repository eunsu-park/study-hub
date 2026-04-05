"""
Exercises for Lesson 15: Formal Verification with TLA+
Topic: Distributed_Systems

Solutions to practice problems from the lesson.
These exercises work with TLA+ specifications represented as Python
strings, validating their structure and demonstrating key concepts.
"""

import re
from typing import Dict, List, Optional, Set, Tuple


# === Exercise 1: TLA+ Spec for Mutual Exclusion ===
# Problem: Write a TLA+ specification for mutual exclusion (embedded
# as strings). Parse and validate the structure to ensure it contains
# the required elements: variables, init, next, and invariants.

MUTEX_SPEC = r"""
---- MODULE MutualExclusion ----
EXTENDS Integers, FiniteSets

CONSTANTS Procs

VARIABLES pc, turn

vars == <<pc, turn>>

Init ==
    /\ pc = [p \in Procs |-> "idle"]
    /\ turn = CHOOSE p \in Procs : TRUE

Request(p) ==
    /\ pc[p] = "idle"
    /\ pc' = [pc EXCEPT ![p] = "waiting"]
    /\ UNCHANGED turn

Enter(p) ==
    /\ pc[p] = "waiting"
    /\ turn = p
    /\ pc' = [pc EXCEPT ![p] = "critical"]
    /\ UNCHANGED turn

Exit(p) ==
    /\ pc[p] = "critical"
    /\ \E q \in Procs \ {p} :
        /\ pc' = [pc EXCEPT ![p] = "idle"]
        /\ turn' = q

Next ==
    \E p \in Procs :
        \/ Request(p)
        \/ Enter(p)
        \/ Exit(p)

MutualExclusion ==
    \A p, q \in Procs :
        (p /= q) => ~(pc[p] = "critical" /\ pc[q] = "critical")

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

THEOREM Spec => []MutualExclusion

====
"""


def validate_tlaplus_spec(spec: str) -> Dict[str, any]:
    """
    Validate the structure of a TLA+ specification.

    Checks for:
    - MODULE declaration
    - EXTENDS clause
    - VARIABLES declaration
    - Init predicate
    - Next action
    - At least one invariant/property
    - Closing ====

    Returns a dict with validation results.
    """
    results = {
        "has_module": False,
        "module_name": None,
        "has_extends": False,
        "extends_modules": [],
        "has_variables": False,
        "variables": [],
        "has_init": False,
        "has_next": False,
        "has_invariant": False,
        "invariant_names": [],
        "has_spec": False,
        "has_theorem": False,
        "has_closing": False,
        "is_valid": False,
        "errors": [],
    }

    # Check MODULE
    module_match = re.search(r'----\s*MODULE\s+(\w+)\s*----', spec)
    if module_match:
        results["has_module"] = True
        results["module_name"] = module_match.group(1)
    else:
        results["errors"].append("Missing MODULE declaration")

    # Check EXTENDS
    extends_match = re.search(r'EXTENDS\s+(.+)', spec)
    if extends_match:
        results["has_extends"] = True
        modules = [m.strip() for m in extends_match.group(1).split(",")]
        results["extends_modules"] = modules

    # Check VARIABLES
    vars_match = re.search(r'VARIABLES?\s+(.+)', spec)
    if vars_match:
        results["has_variables"] = True
        variables = [v.strip() for v in vars_match.group(1).split(",")]
        results["variables"] = variables
    else:
        results["errors"].append("Missing VARIABLES declaration")

    # Check Init
    if re.search(r'^Init\s*==', spec, re.MULTILINE):
        results["has_init"] = True
    else:
        results["errors"].append("Missing Init predicate")

    # Check Next
    if re.search(r'^Next\s*==', spec, re.MULTILINE):
        results["has_next"] = True
    else:
        results["errors"].append("Missing Next action")

    # Check for invariants (properties defined as predicates)
    # Look for predicates that look like invariants
    invariant_candidates = re.findall(
        r'^(\w+)\s*==\s*\n?\s*\\A', spec, re.MULTILINE
    )
    if invariant_candidates:
        results["has_invariant"] = True
        results["invariant_names"] = invariant_candidates

    # Check Spec
    if re.search(r'^Spec\s*==', spec, re.MULTILINE):
        results["has_spec"] = True

    # Check THEOREM
    if re.search(r'^THEOREM', spec, re.MULTILINE):
        results["has_theorem"] = True

    # Check closing
    if re.search(r'^====', spec, re.MULTILINE):
        results["has_closing"] = True
    else:
        results["errors"].append("Missing closing ====")

    results["is_valid"] = (
        results["has_module"]
        and results["has_variables"]
        and results["has_init"]
        and results["has_next"]
        and results["has_closing"]
    )

    return results


def exercise_1():
    """
    Validate the mutual exclusion TLA+ specification.
    """
    print("=== Exercise 1: TLA+ Mutual Exclusion Spec ===\n")

    results = validate_tlaplus_spec(MUTEX_SPEC)

    print(f"Module: {results['module_name']}")
    print(f"Extends: {results['extends_modules']}")
    print(f"Variables: {results['variables']}")
    print(f"Has Init: {results['has_init']}")
    print(f"Has Next: {results['has_next']}")
    print(f"Invariants: {results['invariant_names']}")
    print(f"Has Spec: {results['has_spec']}")
    print(f"Has Theorem: {results['has_theorem']}")
    print(f"Valid structure: {results['is_valid']}")

    if results["errors"]:
        print(f"Errors: {results['errors']}")

    assert results["is_valid"], "Spec should be structurally valid"
    assert results["module_name"] == "MutualExclusion"
    assert "MutualExclusion" in results["invariant_names"]
    print("\nSpec validation passed.")
    print()


# === Exercise 2: PlusCal to Pseudocode Converter ===
# Problem: Parse a PlusCal algorithm (embedded as a string) and
# convert it to readable pseudocode.

PLUSCAL_ALGORITHM = r"""
--algorithm Peterson {
    variables flag = [i \in {0, 1} |-> FALSE], turn = 0;

    fair process (proc \in {0, 1})
    variables other = 1 - self;
    {
        ncs: while (TRUE) {
            (* Non-critical section *)
            skip;

            set_flag:
            flag[self] := TRUE;

            set_turn:
            turn := other;

            wait:
            await ~flag[other] \/ turn = self;

            cs:
            (* Critical section *)
            skip;

            reset:
            flag[self] := FALSE;
        }
    }
}
"""


def pluscal_to_pseudocode(pluscal: str) -> str:
    """
    Convert a PlusCal algorithm to readable pseudocode.

    Handles:
    - Algorithm name extraction
    - Variable declarations
    - Process definitions
    - Labels and statements
    - PlusCal operators to pseudocode equivalents
    """
    lines = pluscal.strip().split("\n")
    output_lines = []
    indent = 0

    def add_line(text: str):
        output_lines.append("  " * indent + text)

    # Extract algorithm name
    algo_match = re.search(r'--algorithm\s+(\w+)', pluscal)
    algo_name = algo_match.group(1) if algo_match else "Unknown"
    add_line(f"Algorithm: {algo_name}")
    add_line("")

    # Extract variables
    vars_match = re.search(
        r'variables\s+(.+?);', pluscal, re.DOTALL
    )
    if vars_match:
        var_text = vars_match.group(1).strip()
        var_text = var_text.replace("|->", "=>")
        var_text = var_text.replace("\\in", "in")
        add_line(f"Variables: {var_text}")
        add_line("")

    # Extract process
    proc_match = re.search(
        r'process\s+\((\w+)\s+\\in\s+(.+?)\)', pluscal
    )
    if proc_match:
        proc_name = proc_match.group(1)
        proc_set = proc_match.group(2).replace("\\in", "in")
        add_line(f"Process {proc_name} in {proc_set}:")
        indent += 1

    # Process local variables
    local_vars = re.findall(
        r'variables\s+(\w+)\s*=\s*(.+?);', pluscal
    )
    for var_name, var_val in local_vars:
        if var_name not in ("flag", "turn"):
            var_val = var_val.replace("\\in", "in")
            add_line(f"local {var_name} = {var_val.strip()}")

    # Extract labels and statements
    label_pattern = re.compile(r'(\w+):\s*(.*)')
    for line in lines:
        line = line.strip()

        # Skip PlusCal delimiters
        if line.startswith("--algorithm") or line in ("{", "}", ""):
            continue
        if line.startswith("variables") or line.startswith("fair process"):
            continue

        # Handle labels
        label_match = label_pattern.match(line)
        if label_match:
            label = label_match.group(1)
            stmt = label_match.group(2).strip()

            add_line("")
            add_line(f"[{label}]")
            if stmt:
                stmt = convert_statement(stmt)
                if stmt:
                    indent += 1
                    add_line(stmt)
                    indent -= 1
            continue

        # Handle statements
        stmt = convert_statement(line)
        if stmt:
            add_line(stmt)

    return "\n".join(output_lines)


def convert_statement(stmt: str) -> Optional[str]:
    """Convert a PlusCal statement to pseudocode."""
    stmt = stmt.strip().rstrip(";").strip()
    if not stmt or stmt in ("{", "}"):
        return None

    # Remove comments
    stmt = re.sub(r'\(\*.*?\*\)', '', stmt).strip()
    if not stmt:
        return None

    # Conversions
    stmt = stmt.replace(":=", " = ")
    stmt = stmt.replace("\\/", "or")
    stmt = stmt.replace("/\\", "and")
    stmt = stmt.replace("~", "not ")
    stmt = stmt.replace("\\in", "in")
    stmt = stmt.replace("await", "wait until")
    stmt = stmt.replace("while (TRUE)", "loop forever:")
    stmt = stmt.replace("skip", "// do nothing")

    return stmt


def exercise_2():
    """
    Convert PlusCal to pseudocode.
    """
    print("=== Exercise 2: PlusCal to Pseudocode ===\n")

    pseudocode = pluscal_to_pseudocode(PLUSCAL_ALGORITHM)
    print(pseudocode)
    print()

    # Verify key elements are present
    assert "Peterson" in pseudocode, "Should contain algorithm name"
    assert "wait until" in pseudocode, "Should convert await to wait until"
    print("Conversion completed successfully.")
    print()


# === Exercise 3: Raft Leader Election Invariants ===
# Problem: Write invariants for a simplified Raft leader election spec.
# Define and check invariants against simulated states.

RAFT_ELECTION_SPEC = r"""
---- MODULE RaftLeaderElection ----
EXTENDS Integers, FiniteSets

CONSTANTS Servers

VARIABLES currentTerm, state, votedFor, votesGranted

vars == <<currentTerm, state, votedFor, votesGranted>>

Init ==
    /\ currentTerm = [s \in Servers |-> 0]
    /\ state = [s \in Servers |-> "follower"]
    /\ votedFor = [s \in Servers |-> "none"]
    /\ votesGranted = [s \in Servers |-> {}]

\* Invariant 1: At most one leader per term
ElectionSafety ==
    \A t \in 0..100 :
        Cardinality({s \in Servers : state[s] = "leader" /\ currentTerm[s] = t}) <= 1

\* Invariant 2: A leader must have received votes from a majority
LeaderHasMajority ==
    \A s \in Servers :
        state[s] = "leader" =>
            Cardinality(votesGranted[s]) * 2 > Cardinality(Servers)

\* Invariant 3: Each server votes for at most one candidate per term
VoteUniqueness ==
    \A s \in Servers :
        votedFor[s] /= "none" =>
            \A t \in Servers :
                (votedFor[t] = s /\ currentTerm[t] = currentTerm[s])
                    => t = s \/ votedFor[s] = t

====
"""


@dataclass
class RaftState:
    """Simulated Raft state for invariant checking."""
    servers: List[str]
    current_term: Dict[str, int]
    state: Dict[str, str]  # "follower", "candidate", "leader"
    voted_for: Dict[str, Optional[str]]
    votes_granted: Dict[str, Set[str]]


def check_election_safety(raft: RaftState) -> bool:
    """At most one leader per term."""
    leaders_per_term: Dict[int, List[str]] = {}
    for server in raft.servers:
        if raft.state[server] == "leader":
            term = raft.current_term[server]
            if term not in leaders_per_term:
                leaders_per_term[term] = []
            leaders_per_term[term].append(server)

    for term, leaders in leaders_per_term.items():
        if len(leaders) > 1:
            return False
    return True


def check_leader_has_majority(raft: RaftState) -> bool:
    """Every leader must have received votes from a majority."""
    n = len(raft.servers)
    majority = n // 2 + 1
    for server in raft.servers:
        if raft.state[server] == "leader":
            if len(raft.votes_granted[server]) < majority:
                return False
    return True


def check_vote_uniqueness(raft: RaftState) -> bool:
    """Each server votes for at most one candidate per term."""
    # Group votes by term
    votes_per_term: Dict[int, Dict[str, str]] = {}
    for server in raft.servers:
        term = raft.current_term[server]
        if raft.voted_for[server] is not None:
            if term not in votes_per_term:
                votes_per_term[term] = {}
            if server in votes_per_term[term]:
                return False  # voted twice in same term
            votes_per_term[term][server] = raft.voted_for[server]
    return True


def exercise_3():
    """
    Check Raft leader election invariants against simulated states.
    """
    print("=== Exercise 3: Raft Leader Election Invariants ===\n")

    # Validate the TLA+ spec structure
    results = validate_tlaplus_spec(RAFT_ELECTION_SPEC)
    print(f"Spec valid: {results['is_valid']}")
    print(f"Module: {results['module_name']}")
    print()

    # Valid state: one leader in term 3 with majority votes
    servers = ["S1", "S2", "S3", "S4", "S5"]
    valid_state = RaftState(
        servers=servers,
        current_term={"S1": 3, "S2": 3, "S3": 3, "S4": 2, "S5": 3},
        state={"S1": "leader", "S2": "follower", "S3": "follower",
               "S4": "follower", "S5": "follower"},
        voted_for={"S1": "S1", "S2": "S1", "S3": "S1",
                    "S4": None, "S5": "S1"},
        votes_granted={"S1": {"S1", "S2", "S3", "S5"},
                        "S2": set(), "S3": set(), "S4": set(), "S5": set()},
    )

    print("Valid state (1 leader, majority votes):")
    print(f"  ElectionSafety:    {check_election_safety(valid_state)}")
    print(f"  LeaderHasMajority: {check_leader_has_majority(valid_state)}")
    print(f"  VoteUniqueness:    {check_vote_uniqueness(valid_state)}")

    # Invalid state 1: two leaders in same term
    invalid_state1 = RaftState(
        servers=servers,
        current_term={"S1": 3, "S2": 3, "S3": 3, "S4": 3, "S5": 3},
        state={"S1": "leader", "S2": "leader", "S3": "follower",
               "S4": "follower", "S5": "follower"},
        voted_for={"S1": "S1", "S2": "S2", "S3": "S1",
                    "S4": "S2", "S5": "S1"},
        votes_granted={"S1": {"S1", "S3", "S5"},
                        "S2": {"S2", "S4"}, "S3": set(),
                        "S4": set(), "S5": set()},
    )

    print("\nInvalid state 1 (two leaders, same term):")
    print(f"  ElectionSafety:    {check_election_safety(invalid_state1)}")

    # Invalid state 2: leader without majority
    invalid_state2 = RaftState(
        servers=servers,
        current_term={"S1": 5, "S2": 5, "S3": 4, "S4": 4, "S5": 4},
        state={"S1": "leader", "S2": "follower", "S3": "follower",
               "S4": "follower", "S5": "follower"},
        voted_for={"S1": "S1", "S2": "S1", "S3": None,
                    "S4": None, "S5": None},
        votes_granted={"S1": {"S1", "S2"}, "S2": set(), "S3": set(),
                        "S4": set(), "S5": set()},
    )

    print("\nInvalid state 2 (leader without majority):")
    print(f"  LeaderHasMajority: {check_leader_has_majority(invalid_state2)}")

    assert check_election_safety(valid_state) is True
    assert check_election_safety(invalid_state1) is False
    assert check_leader_has_majority(invalid_state2) is False
    print("\nAll invariant checks passed.")
    print()


# === Main ===

if __name__ == "__main__":
    exercise_1()
    exercise_2()
    exercise_3()
