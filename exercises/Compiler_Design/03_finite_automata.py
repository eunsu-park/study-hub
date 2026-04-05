"""
Exercises for Lesson 03: Finite Automata
Topic: Compiler_Design

Solutions to practice problems from the lesson.
"""

from collections import defaultdict
from itertools import product as itertools_product


# === Exercise 1: NFA Construction and Simulation ===
# Problem: Construct an NFA for (ab|ba)* and simulate on "abba", "abab", "aabb", ""

def exercise_1():
    """NFA construction and simulation for (ab|ba)*."""

    # NFA representation: transitions[state][symbol] = set of states
    # Epsilon transitions: transitions[state][''] = set of states
    # States: 0=start, 1-10 internal, 11=accept
    # (ab|ba)* using Thompson's construction:
    #
    # ab: 1 --a--> 2 --b--> 3
    # ba: 4 --b--> 5 --a--> 6
    # ab|ba: 7 --eps--> 1, 7 --eps--> 4, 3 --eps--> 8, 6 --eps--> 8
    # (ab|ba)*: 0 --eps--> 7, 0 --eps--> 9(accept), 8 --eps--> 7, 8 --eps--> 9

    transitions = defaultdict(lambda: defaultdict(set))
    # ab path
    transitions[1]['a'].add(2)
    transitions[2]['b'].add(3)
    # ba path
    transitions[4]['b'].add(5)
    transitions[5]['a'].add(6)
    # union
    transitions[7][''].add(1)
    transitions[7][''].add(4)
    transitions[3][''].add(8)
    transitions[6][''].add(8)
    # star
    transitions[0][''].add(7)
    transitions[0][''].add(9)
    transitions[8][''].add(7)
    transitions[8][''].add(9)

    start = 0
    accept = {9}

    print("NFA for (ab|ba)*")
    print("  States: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}")
    print("  Start: 0, Accept: {9}")
    print("  Transitions:")
    for state in sorted(transitions.keys()):
        for sym in sorted(transitions[state].keys()):
            targets = transitions[state][sym]
            sym_label = 'eps' if sym == '' else sym
            print(f"    q{state} --{sym_label}--> {{{', '.join(f'q{t}' for t in sorted(targets))}}}")
    print()

    def epsilon_closure(states):
        closure = set(states)
        stack = list(states)
        while stack:
            s = stack.pop()
            for ns in transitions[s].get('', set()):
                if ns not in closure:
                    closure.add(ns)
                    stack.append(ns)
        return frozenset(closure)

    def simulate(input_str):
        current = epsilon_closure({start})
        for ch in input_str:
            next_states = set()
            for s in current:
                next_states.update(transitions[s].get(ch, set()))
            current = epsilon_closure(next_states)
        return bool(current & accept)

    test_cases = [
        ("abba", True),   # ab then ba
        ("abab", True),   # ab then ab
        ("aabb", False),  # "aa" is not a valid prefix pair
        ("", True),       # zero repetitions of (ab|ba)
        ("ab", True),     # one ab
        ("ba", True),     # one ba
        ("a", False),     # incomplete
        ("baba", True),   # ba then ba
    ]

    print("  Simulation results:")
    for s, expected in test_cases:
        result = simulate(s)
        status = "PASS" if result == expected else "FAIL"
        print(f"    [{status}] '{s}' -> {'Accept' if result else 'Reject'} "
              f"(expected {'Accept' if expected else 'Reject'})")


# === Exercise 2: Subset Construction ===
# Problem: Given NFA with epsilon transitions, apply subset construction.
# NFA table:
#   q0: a->{q1}, b->{}, eps->{q2}
#   q1: a->{}, b->{q1,q3}, eps->{}
#   q2: a->{q3}, b->{}, eps->{}
#   *q3: a->{}, b->{}, eps->{}

def exercise_2():
    """Subset construction on a given NFA."""
    # NFA definition
    nfa_transitions = {
        0: {'a': {1}, 'b': set(), '': {2}},
        1: {'a': set(), 'b': {1, 3}, '': set()},
        2: {'a': {3}, 'b': set(), '': set()},
        3: {'a': set(), 'b': set(), '': set()},
    }
    nfa_start = 0
    nfa_accept = {3}
    alphabet = {'a', 'b'}

    def epsilon_closure(states):
        closure = set(states)
        stack = list(states)
        while stack:
            s = stack.pop()
            for ns in nfa_transitions.get(s, {}).get('', set()):
                if ns not in closure:
                    closure.add(ns)
                    stack.append(ns)
        return frozenset(closure)

    # Step 1: Epsilon-closures
    print("Step 1: Epsilon-closures")
    for state in sorted(nfa_transitions.keys()):
        ec = epsilon_closure({state})
        print(f"  eps-closure(q{state}) = {{{', '.join(f'q{s}' for s in sorted(ec))}}}")
    print()

    # Step 2: Subset construction
    start_closure = epsilon_closure({nfa_start})
    dfa_states = {}
    dfa_transitions = {}
    dfa_accept_states = set()
    worklist = [start_closure]
    dfa_states[start_closure] = 0
    state_counter = 1

    print("Step 2: Subset construction")
    while worklist:
        current = worklist.pop(0)
        current_id = dfa_states[current]
        is_accept = bool(current & nfa_accept)
        if is_accept:
            dfa_accept_states.add(current_id)

        for symbol in sorted(alphabet):
            next_nfa = set()
            for s in current:
                next_nfa.update(nfa_transitions.get(s, {}).get(symbol, set()))
            next_closure = epsilon_closure(next_nfa)
            if not next_closure:
                continue
            if next_closure not in dfa_states:
                dfa_states[next_closure] = state_counter
                state_counter += 1
                worklist.append(next_closure)
            dfa_transitions[(current_id, symbol)] = dfa_states[next_closure]
            nfa_set_str = '{' + ', '.join(f'q{s}' for s in sorted(current)) + '}'
            target_str = '{' + ', '.join(f'q{s}' for s in sorted(next_closure)) + '}'
            print(f"  D{current_id}{nfa_set_str} --{symbol}--> D{dfa_states[next_closure]}{target_str}")

    print()
    print(f"  DFA has {len(dfa_states)} states")
    print(f"  Accept states: {dfa_accept_states}")
    print()

    # Step 3: DFA transition table
    print("Step 3: DFA transition table")
    print(f"  {'State':>6} | NFA states          | {'a':>4} | {'b':>4}")
    print(f"  {'-'*6}-+{'-'*21}-+{'-'*4}-+{'-'*4}")
    for nfa_set, dfa_id in sorted(dfa_states.items(), key=lambda x: x[1]):
        nfa_str = '{' + ','.join(f'q{s}' for s in sorted(nfa_set)) + '}'
        accept_mark = '*' if dfa_id in dfa_accept_states else ' '
        a_target = dfa_transitions.get((dfa_id, 'a'), '-')
        b_target = dfa_transitions.get((dfa_id, 'b'), '-')
        print(f"  {accept_mark}D{dfa_id:>4} | {nfa_str:<19} | {a_target:>4} | {b_target:>4}")
    print()

    # Step 4: Language description
    print("Step 4: Language recognized")
    print("  The automaton accepts strings that can reach q3.")
    print("  From q0, via eps to q2, then 'a' reaches q3 (accept).")
    print("  From q0, 'a' reaches q1, then 'b' reaches {q1,q3} (accept).")
    print("  Language: strings of the form a, ab+, (ab*)*a, etc.")
    print("  More precisely: L = { w in {a,b}* | w contains 'a' and either")
    print("  ends with 'a' (via q2->q3) or 'a' is followed by any 'b's (via q1) }")


# === Exercise 3: DFA Minimization ===
# Problem: Minimize DFA using table-filling algorithm.
# States: A(start), B, C(accept), D, E(accept), F(accept)
# Transitions on {0, 1}:
#   A: 0->B, 1->C
#   B: 0->D, 1->E
#   C: 0->F, 1->C
#   D: 0->B, 1->E
#   E: 0->F, 1->C
#   F: 0->F, 1->C

def exercise_3():
    """DFA minimization using the table-filling algorithm."""
    states = ['A', 'B', 'C', 'D', 'E', 'F']
    accept = {'C', 'E', 'F'}
    non_accept = {'A', 'B', 'D'}
    transitions = {
        'A': {'0': 'B', '1': 'C'},
        'B': {'0': 'D', '1': 'E'},
        'C': {'0': 'F', '1': 'C'},
        'D': {'0': 'B', '1': 'E'},
        'E': {'0': 'F', '1': 'C'},
        'F': {'0': 'F', '1': 'C'},
    }

    print("Original DFA:")
    print("  State | 0 | 1")
    print("  ------+---+---")
    for s in states:
        mark = '*' if s in accept else ' '
        print(f"  {mark}{s:>4}  | {transitions[s]['0']} | {transitions[s]['1']}")
    print()

    # Table-filling algorithm
    # Initialize: mark all (accept, non-accept) pairs as distinguishable
    n = len(states)
    distinguishable = [[False] * n for _ in range(n)]
    idx = {s: i for i, s in enumerate(states)}

    # Step 1: Mark accept/non-accept pairs
    print("Step 1: Mark accept/non-accept pairs")
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = states[i], states[j]
            if (si in accept) != (sj in accept):
                distinguishable[i][j] = True
                distinguishable[j][i] = True
                print(f"  Mark ({si}, {sj}) - one accepting, one not")
    print()

    # Step 2: Iterate until no new marks
    print("Step 2: Iterative marking")
    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        for i in range(n):
            for j in range(i + 1, n):
                if distinguishable[i][j]:
                    continue
                si, sj = states[i], states[j]
                for symbol in ['0', '1']:
                    ti = idx[transitions[si][symbol]]
                    tj = idx[transitions[sj][symbol]]
                    if distinguishable[ti][tj]:
                        distinguishable[i][j] = True
                        distinguishable[j][i] = True
                        changed = True
                        print(f"  Iteration {iteration}: Mark ({si}, {sj}) "
                              f"because on '{symbol}': "
                              f"({transitions[si][symbol]}, {transitions[sj][symbol]}) "
                              f"are distinguishable")
                        break
    print()

    # Step 3: Identify equivalent pairs
    print("Step 3: Equivalent (unmarked) pairs")
    equiv_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if not distinguishable[i][j]:
                equiv_pairs.append((states[i], states[j]))
                print(f"  {states[i]} ~ {states[j]}")
    print()

    # Step 4: Build equivalence classes
    # Use union-find
    parent = {s: s for s in states}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for s1, s2 in equiv_pairs:
        union(s1, s2)

    classes = defaultdict(set)
    for s in states:
        classes[find(s)].add(s)

    print("Step 4: Equivalence classes (minimized DFA states)")
    class_names = {}
    for i, (rep, members) in enumerate(sorted(classes.items(), key=lambda x: states.index(min(x[1])))):
        class_name = '{' + ','.join(sorted(members)) + '}'
        class_names[rep] = class_name
        is_accept = any(m in accept for m in members)
        is_start = any(m == 'A' for m in members)
        markers = []
        if is_start:
            markers.append("start")
        if is_accept:
            markers.append("accept")
        print(f"  S{i}: {class_name}" + (f"  ({', '.join(markers)})" if markers else ""))
    print()

    # Build minimized DFA
    print("Minimized DFA transitions:")
    print("  State     | 0         | 1")
    print("  ----------+-----------+----------")
    for rep, members in sorted(classes.items(), key=lambda x: states.index(min(x[1]))):
        sample = min(members)
        t0 = find(transitions[sample]['0'])
        t1 = find(transitions[sample]['1'])
        src = class_names[rep]
        dst0 = class_names[t0]
        dst1 = class_names[t1]
        is_accept = any(m in accept for m in members)
        mark = '*' if is_accept else ' '
        print(f"  {mark}{src:<9} | {dst0:<9} | {dst1}")


# === Exercise 4: Pumping Lemma Proofs ===
# Problem: Prove L1={a^n b^(2n)}, L2={equal a's and b's}, L3={a^(n!)| n>=1} are not regular.

def exercise_4():
    """Pumping lemma proofs that certain languages are not regular."""
    print("Pumping Lemma Proofs")
    print("=" * 50)
    print()

    print("Recall: If L is regular, then there exists p (pumping length) such that")
    print("for any s in L with |s| >= p, s = xyz where:")
    print("  1. |y| > 0")
    print("  2. |xy| <= p")
    print("  3. For all i >= 0, xy^iz is in L")
    print()

    # Proof 1: L1 = {a^n b^(2n) | n >= 0}
    print("Proof 1: L1 = {a^n b^(2n) | n >= 0} is NOT regular")
    print("-" * 50)
    print("  Assume L1 is regular with pumping length p.")
    print("  Choose s = a^p b^(2p). Then s is in L1 and |s| = 3p >= p.")
    print("  Since |xy| <= p, y consists entirely of a's: y = a^k for some k >= 1.")
    print("  Pumping down (i=0): xz = a^(p-k) b^(2p).")
    print("  For this to be in L1, we need 2(p-k) = 2p, i.e., k=0.")
    print("  But k >= 1, contradiction.")
    print("  Therefore L1 is not regular.")
    print()

    # Verification by testing a pumping attempt
    print("  Verification (computational):")
    p = 5
    s = 'a' * p + 'b' * (2 * p)
    print(f"    s = a^{p} b^{2*p} = '{s}'")
    # Try all valid decompositions
    found_valid = False
    for xy_len in range(1, p + 1):
        for y_len in range(1, xy_len + 1):
            x = s[:xy_len - y_len]
            y = s[xy_len - y_len:xy_len]
            z = s[xy_len:]
            # Check pumping for i=0 and i=2
            for i in [0, 2]:
                pumped = x + y * i + z
                a_count = pumped.count('a')
                b_count = pumped.count('b')
                if b_count == 2 * a_count:
                    found_valid = True
    print(f"    Any valid decomposition that pumps? {found_valid}")
    print()

    # Proof 2: L2 = {w in {a,b}* | equal number of a's and b's}
    print("Proof 2: L2 = {w in {a,b}* | #a = #b} is NOT regular")
    print("-" * 50)
    print("  Assume L2 is regular with pumping length p.")
    print("  Choose s = a^p b^p. Then s is in L2 and |s| = 2p >= p.")
    print("  Since |xy| <= p, y consists entirely of a's: y = a^k for some k >= 1.")
    print("  Pumping up (i=2): xy^2z = a^(p+k) b^p.")
    print("  Since k >= 1, p+k > p, so #a != #b.")
    print("  This string is not in L2, contradiction.")
    print("  Therefore L2 is not regular.")
    print()

    # Proof 3: L3 = {a^(n!) | n >= 1}
    print("Proof 3: L3 = {a^(n!) | n >= 1} is NOT regular")
    print("-" * 50)
    print("  Assume L3 is regular with pumping length p.")
    print("  Choose n such that n! >= p (e.g., n = p).")
    print("  Let s = a^(n!). Then s is in L3 and |s| = n! >= p.")
    print("  Write s = xyz with |y| = k >= 1 and |xy| <= p.")
    print("  Pumping up (i=2): |xy^2z| = n! + k.")
    print("  For this to be in L3, we need n! + k = m! for some m > n.")
    print("  But (n+1)! = (n+1) * n!, so (n+1)! - n! = n * n!.")
    print("  Since k <= p <= n! and k < n * n! = (n+1)! - n!,")
    print("  we have n! < n! + k < (n+1)!, so n! + k is NOT a factorial.")
    print("  Contradiction. Therefore L3 is not regular.")


# === Exercise 5: Closure Properties ===
# Problem: Product construction for intersection; complement construction.

def exercise_5():
    """DFA closure properties: product construction and complement."""

    # D1: accepts strings containing "ab" over {a,b}
    # States: {s0, s1, s2}, s0=start, s2=accept
    # s0 --a--> s1, s0 --b--> s0
    # s1 --a--> s1, s1 --b--> s2
    # s2 --a--> s2, s2 --b--> s2
    d1_trans = {
        's0': {'a': 's1', 'b': 's0'},
        's1': {'a': 's1', 'b': 's2'},
        's2': {'a': 's2', 'b': 's2'},
    }
    d1_start = 's0'
    d1_accept = {'s2'}

    # D2: accepts strings with even length over {a,b}
    # States: {e0, e1}, e0=start=accept
    # e0 --a--> e1, e0 --b--> e1
    # e1 --a--> e0, e1 --b--> e0
    d2_trans = {
        'e0': {'a': 'e1', 'b': 'e1'},
        'e1': {'a': 'e0', 'b': 'e0'},
    }
    d2_start = 'e0'
    d2_accept = {'e0'}

    # Part 1: Product construction for L(D1) ∩ L(D2)
    print("Part 1: Product construction for L(D1) ∩ L(D2)")
    print("  D1: accepts strings containing 'ab'")
    print("  D2: accepts strings with even length")
    print()

    product_states = {}
    product_trans = {}
    product_accept = set()
    worklist = [(d1_start, d2_start)]
    product_states[(d1_start, d2_start)] = 0
    counter = 1

    while worklist:
        s1, s2 = worklist.pop(0)
        pid = product_states[(s1, s2)]
        if s1 in d1_accept and s2 in d2_accept:
            product_accept.add(pid)

        for symbol in ['a', 'b']:
            ns1 = d1_trans[s1][symbol]
            ns2 = d2_trans[s2][symbol]
            if (ns1, ns2) not in product_states:
                product_states[(ns1, ns2)] = counter
                counter += 1
                worklist.append((ns1, ns2))
            product_trans[(pid, symbol)] = product_states[(ns1, ns2)]

    print("  Product DFA transition table:")
    print(f"  {'State':>6} | {'Components':>12} | {'a':>4} | {'b':>4}")
    print(f"  {'-'*6}-+{'-'*14}+{'-'*6}+{'-'*5}")
    for (s1, s2), pid in sorted(product_states.items(), key=lambda x: x[1]):
        mark = '*' if pid in product_accept else ' '
        a_next = product_trans.get((pid, 'a'), '-')
        b_next = product_trans.get((pid, 'b'), '-')
        print(f"  {mark}P{pid:>4} | ({s1},{s2}){' '*(7-len(s1)-len(s2))} | P{a_next:>3} | P{b_next:>3}")
    print()

    # Verify with test strings
    def simulate_product(s):
        state = 0
        for ch in s:
            state = product_trans[(state, ch)]
        return state in product_accept

    test = ["ab", "aabb", "abab", "a", "bb", "abba", "bab", "baab"]
    print("  Verification (must contain 'ab' AND have even length):")
    for w in test:
        contains_ab = 'ab' in w
        even_len = len(w) % 2 == 0
        expected = contains_ab and even_len
        result = simulate_product(w)
        status = "PASS" if result == expected else "FAIL"
        print(f"    [{status}] '{w}': contains_ab={contains_ab}, even_len={even_len} -> {result}")
    print()

    # Part 2: Complement of D1
    print("Part 2: Complement of L(D1)")
    print("  Complement: swap accept and non-accept states")
    print("  Original D1 accept: {s2}")
    print("  Complement accept: {s0, s1}")
    print("  The complement accepts all strings over {a,b} that do NOT contain 'ab'.")
    print()

    d1_comp_accept = set(d1_trans.keys()) - d1_accept
    print("  Complement DFA:")
    print(f"  {'State':>6} | {'a':>4} | {'b':>4}")
    for s in sorted(d1_trans.keys()):
        mark = '*' if s in d1_comp_accept else ' '
        print(f"  {mark}{s:>5} | {d1_trans[s]['a']:>4} | {d1_trans[s]['b']:>4}")

    print()
    print("  Verification (must NOT contain 'ab'):")
    def simulate_complement(w):
        state = d1_start
        for ch in w:
            state = d1_trans[state][ch]
        return state in d1_comp_accept

    for w in ["", "a", "b", "ba", "bb", "bbb", "ab", "aab", "bab"]:
        expected = 'ab' not in w
        result = simulate_complement(w)
        status = "PASS" if result == expected else "FAIL"
        print(f"    [{status}] '{w}' -> {result} (expected {expected})")


# === Exercise 6: Implementation Challenge ===
# Problem: Implement regex -> NFA (Thompson) -> DFA (subset) -> minimize (Hopcroft) -> simulate.

class State:
    """NFA/DFA state."""
    _counter = 0

    def __init__(self):
        self.id = State._counter
        State._counter += 1
        self.char_transitions = {}  # char -> set of State
        self.epsilon = set()        # set of State

    def __repr__(self):
        return f"S{self.id}"

    def __hash__(self):
        return self.id


class Fragment:
    """NFA fragment with start and accept states."""
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept


def parse_regex(pattern):
    """
    Simple regex parser supporting: concat, |, *, +, ?, parentheses.
    Returns an NFA Fragment.
    """
    State._counter = 0
    pos = [0]

    def peek():
        return pattern[pos[0]] if pos[0] < len(pattern) else None

    def advance():
        ch = pattern[pos[0]]
        pos[0] += 1
        return ch

    def parse_expr():
        """expr = term ('|' term)*"""
        left = parse_term()
        while peek() == '|':
            advance()  # consume '|'
            right = parse_term()
            # Union
            start = State()
            accept = State()
            start.epsilon.add(left.start)
            start.epsilon.add(right.start)
            left.accept.epsilon.add(accept)
            right.accept.epsilon.add(accept)
            left = Fragment(start, accept)
        return left

    def parse_term():
        """term = factor+"""
        result = parse_factor()
        if result is None:
            # Empty term
            start = State()
            accept = State()
            start.epsilon.add(accept)
            return Fragment(start, accept)
        while peek() and peek() not in '|)':
            right = parse_factor()
            if right is None:
                break
            # Concatenation
            result.accept.epsilon.add(right.start)
            result = Fragment(result.start, right.accept)
        return result

    def parse_factor():
        """factor = atom ('*' | '+' | '?')?"""
        atom = parse_atom()
        if atom is None:
            return None
        while peek() in ('*', '+', '?'):
            op = advance()
            if op == '*':
                start = State()
                accept = State()
                start.epsilon.add(atom.start)
                start.epsilon.add(accept)
                atom.accept.epsilon.add(atom.start)
                atom.accept.epsilon.add(accept)
                atom = Fragment(start, accept)
            elif op == '+':
                # a+ = aa*
                start = State()
                accept = State()
                start.epsilon.add(atom.start)
                atom.accept.epsilon.add(atom.start)
                atom.accept.epsilon.add(accept)
                atom = Fragment(start, accept)
            elif op == '?':
                start = State()
                accept = State()
                start.epsilon.add(atom.start)
                start.epsilon.add(accept)
                atom.accept.epsilon.add(accept)
                atom = Fragment(start, accept)
        return atom

    def parse_atom():
        """atom = '(' expr ')' | '.' | literal_char"""
        ch = peek()
        if ch is None or ch in '|)*+?':
            return None
        if ch == '(':
            advance()
            expr = parse_expr()
            if peek() == ')':
                advance()
            return expr
        elif ch == '.':
            advance()
            # Match any printable ASCII character
            start = State()
            accept = State()
            for c in (chr(i) for i in range(32, 127)):
                start.char_transitions.setdefault(c, set()).add(accept)
            return Fragment(start, accept)
        elif ch == '\\':
            advance()
            escaped = advance()
            start = State()
            accept = State()
            start.char_transitions.setdefault(escaped, set()).add(accept)
            return Fragment(start, accept)
        else:
            advance()
            start = State()
            accept = State()
            start.char_transitions.setdefault(ch, set()).add(accept)
            return Fragment(start, accept)

    return parse_expr()


def nfa_to_dfa(nfa_fragment, alphabet):
    """Convert NFA to DFA using subset construction."""
    def eps_closure(states):
        closure = set(states)
        stack = list(states)
        while stack:
            s = stack.pop()
            for ns in s.epsilon:
                if ns not in closure:
                    closure.add(ns)
                    stack.append(ns)
        return frozenset(closure)

    start_set = eps_closure({nfa_fragment.start})
    dfa_states = {start_set: 0}
    dfa_trans = {}
    dfa_accept = set()
    worklist = [start_set]
    counter = 1

    if nfa_fragment.accept in start_set:
        dfa_accept.add(0)

    while worklist:
        current = worklist.pop(0)
        cid = dfa_states[current]

        for sym in alphabet:
            next_nfa = set()
            for s in current:
                next_nfa.update(s.char_transitions.get(sym, set()))
            if not next_nfa:
                continue
            next_set = eps_closure(next_nfa)
            if next_set not in dfa_states:
                dfa_states[next_set] = counter
                if nfa_fragment.accept in next_set:
                    dfa_accept.add(counter)
                counter += 1
                worklist.append(next_set)
            dfa_trans[(cid, sym)] = dfa_states[next_set]

    return dfa_states, dfa_trans, dfa_accept, 0


def minimize_dfa(num_states, transitions, accept_states, start, alphabet):
    """Minimize DFA using Hopcroft's algorithm (partition refinement)."""
    # Initial partition: accept vs non-accept
    accept = frozenset(accept_states)
    non_accept = frozenset(set(range(num_states)) - accept_states)

    partitions = set()
    if accept:
        partitions.add(accept)
    if non_accept:
        partitions.add(non_accept)

    worklist = list(partitions)

    while worklist:
        splitter = worklist.pop(0)
        for sym in alphabet:
            # Find states that transition into splitter on sym
            predecessors = set()
            for s in range(num_states):
                if (s, sym) in transitions and transitions[(s, sym)] in splitter:
                    predecessors.add(s)

            new_partitions = set()
            for partition in partitions:
                intersection = partition & frozenset(predecessors)
                difference = partition - frozenset(predecessors)
                if intersection and difference:
                    new_partitions.add(intersection)
                    new_partitions.add(difference)
                    if partition in worklist:
                        worklist.remove(partition)
                        worklist.append(intersection)
                        worklist.append(difference)
                    else:
                        if len(intersection) <= len(difference):
                            worklist.append(intersection)
                        else:
                            worklist.append(difference)
                else:
                    new_partitions.add(partition)
            partitions = new_partitions

    # Build minimized DFA
    state_to_partition = {}
    partition_list = sorted(partitions, key=lambda p: min(p))
    for i, partition in enumerate(partition_list):
        for s in partition:
            state_to_partition[s] = i

    min_trans = {}
    min_accept = set()
    min_start = state_to_partition[start]

    for i, partition in enumerate(partition_list):
        rep = min(partition)
        if rep in accept_states:
            min_accept.add(i)
        for sym in alphabet:
            if (rep, sym) in transitions:
                target = transitions[(rep, sym)]
                min_trans[(i, sym)] = state_to_partition[target]

    return len(partition_list), min_trans, min_accept, min_start


def exercise_6():
    """Full pipeline: regex -> NFA -> DFA -> minimize -> simulate."""
    test_cases = [
        (
            "(a|b)+",
            "Identifier-like (simplified)",
            ['a', 'b'],
            [("a", True), ("ab", True), ("bba", True), ("", False)],
        ),
        (
            "(0|1(01*0)*1)*",
            "Binary multiples of 3",
            ['0', '1'],
            [("", True), ("0", True), ("00", True), ("11", True),
             ("110", True), ("1001", True), ("1", False), ("10", False)],
        ),
        (
            "ab*c",
            "a followed by b's then c",
            ['a', 'b', 'c'],
            [("ac", True), ("abc", True), ("abbc", True), ("a", False),
             ("bc", False), ("ab", False)],
        ),
    ]

    for pattern, description, alphabet, tests in test_cases:
        print(f"  Pattern: {pattern} ({description})")

        # Step 1: Parse regex to NFA
        nfa = parse_regex(pattern)
        print(f"    NFA built (start=S{nfa.start.id}, accept=S{nfa.accept.id})")

        # Step 2: NFA to DFA
        dfa_states, dfa_trans, dfa_accept, dfa_start = nfa_to_dfa(nfa, alphabet)
        print(f"    DFA: {len(dfa_states)} states, {len(dfa_accept)} accepting")

        # Step 3: Minimize DFA
        min_count, min_trans, min_accept, min_start = minimize_dfa(
            len(dfa_states), dfa_trans, dfa_accept, dfa_start, alphabet
        )
        print(f"    Minimized DFA: {min_count} states, {len(min_accept)} accepting")

        # Step 4: Simulate
        def simulate(input_str):
            state = min_start
            for ch in input_str:
                key = (state, ch)
                if key not in min_trans:
                    return False
                state = min_trans[key]
            return state in min_accept

        print(f"    Test results:")
        for s, expected in tests:
            result = simulate(s)
            status = "PASS" if result == expected else "FAIL"
            print(f"      [{status}] '{s}' -> {'Accept' if result else 'Reject'}")
        print()


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: NFA Construction and Simulation ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Subset Construction ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: DFA Minimization ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Pumping Lemma Proofs ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Closure Properties ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Implementation Challenge ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
