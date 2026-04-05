"""
Exercises for Lesson 06: Bottom-Up Parsing
Topic: Compiler_Design

Solutions to practice problems from the lesson.
"""

from collections import defaultdict, OrderedDict


# === Exercise 1: LR(0) Automaton Construction ===
# Problem: Construct LR(0) automaton for S'->S, S->AB, A->a, B->b

def exercise_1():
    """Construct LR(0) automaton for a simple grammar."""
    print("Grammar:")
    print("  S' -> S")
    print("  S  -> A B")
    print("  A  -> a")
    print("  B  -> b")
    print()

    # Represent items as (lhs, rhs_tuple, dot_position)
    grammar = {
        "S'": [('S',)],
        'S':  [('A', 'B')],
        'A':  [('a',)],
        'B':  [('b',)],
    }
    terminals = {'a', 'b'}
    nonterminals = {"S'", 'S', 'A', 'B'}

    def closure(items):
        """Compute closure of a set of LR(0) items."""
        result = set(items)
        changed = True
        while changed:
            changed = False
            new_items = set()
            for (lhs, rhs, dot) in result:
                if dot < len(rhs) and rhs[dot] in nonterminals:
                    nt = rhs[dot]
                    for prod in grammar.get(nt, []):
                        item = (nt, prod, 0)
                        if item not in result:
                            new_items.add(item)
                            changed = True
            result.update(new_items)
        return frozenset(result)

    def goto(items, symbol):
        """Compute GOTO(items, symbol)."""
        moved = set()
        for (lhs, rhs, dot) in items:
            if dot < len(rhs) and rhs[dot] == symbol:
                moved.add((lhs, rhs, dot + 1))
        if moved:
            return closure(moved)
        return frozenset()

    def format_item(item):
        lhs, rhs, dot = item
        parts = list(rhs)
        parts.insert(dot, '.')
        return f"{lhs} -> {' '.join(parts)}"

    # Build the automaton
    start_item = ("S'", ('S',), 0)
    state0 = closure({start_item})

    states = [state0]
    state_map = {state0: 0}
    transitions = {}
    worklist = [state0]

    all_symbols = terminals | (nonterminals - {"S'"})

    while worklist:
        current = worklist.pop(0)
        current_id = state_map[current]

        for symbol in sorted(all_symbols):
            next_state = goto(current, symbol)
            if not next_state:
                continue
            if next_state not in state_map:
                state_map[next_state] = len(states)
                states.append(next_state)
                worklist.append(next_state)
            transitions[(current_id, symbol)] = state_map[next_state]

    # Print states
    for state_set, state_id in sorted(state_map.items(), key=lambda x: x[1]):
        print(f"State I{state_id}:")
        for item in sorted(state_set, key=lambda x: (x[0], x[1], x[2])):
            print(f"  {format_item(item)}")
        # Print transitions from this state
        for sym in sorted(all_symbols):
            key = (state_id, sym)
            if key in transitions:
                print(f"  --{sym}--> I{transitions[key]}")
        print()

    print(f"Total states: {len(states)}")


# === Exercise 2: SLR(1) Table Construction ===
# Problem: For S'->S, S->CC, C->cC|d, build SLR table and trace "c d c d".

def exercise_2():
    """SLR(1) table construction and parse trace."""
    grammar = {
        "S'": [('S',)],
        'S':  [('C', 'C')],
        'C':  [('c', 'C'), ('d',)],
    }
    terminals = {'c', 'd'}
    nonterminals = {"S'", 'S', 'C'}

    # 1. FIRST and FOLLOW
    print("1. FIRST and FOLLOW sets:")
    print("  FIRST(S)  = {c, d}")
    print("  FIRST(C)  = {c, d}")
    print("  FOLLOW(S) = {$}")
    print("  FOLLOW(C) = {c, d, $}")
    print()
    follow_C = {'c', 'd', '$'}
    follow_S = {'$'}

    # 2. Construct LR(0) automaton
    def closure(items):
        result = set(items)
        changed = True
        while changed:
            changed = False
            for (lhs, rhs, dot) in list(result):
                if dot < len(rhs) and rhs[dot] in nonterminals:
                    for prod in grammar[rhs[dot]]:
                        item = (rhs[dot], prod, 0)
                        if item not in result:
                            result.add(item)
                            changed = True
        return frozenset(result)

    def goto(items, symbol):
        moved = set()
        for (lhs, rhs, dot) in items:
            if dot < len(rhs) and rhs[dot] == symbol:
                moved.add((lhs, rhs, dot + 1))
        return closure(moved) if moved else frozenset()

    start = closure({("S'", ('S',), 0)})
    states = [start]
    state_map = {start: 0}
    trans = {}
    all_syms = {'S', 'C', 'c', 'd'}

    worklist = [start]
    while worklist:
        current = worklist.pop(0)
        cid = state_map[current]
        for sym in all_syms:
            ns = goto(current, sym)
            if not ns:
                continue
            if ns not in state_map:
                state_map[ns] = len(states)
                states.append(ns)
                worklist.append(ns)
            trans[(cid, sym)] = state_map[ns]

    def fmt(item):
        lhs, rhs, dot = item
        parts = list(rhs)
        parts.insert(dot, '.')
        return f"{lhs} -> {' '.join(parts)}"

    print("2. LR(0) Automaton:")
    for s, sid in sorted(state_map.items(), key=lambda x: x[1]):
        print(f"  I{sid}: {{{', '.join(fmt(i) for i in sorted(s, key=lambda x: (x[0],x[1],x[2])))}}}")
    print()

    # 3. Build SLR(1) table
    # Productions numbered for reduce actions
    productions = [
        ("S'", ('S',)),    # 0
        ('S', ('C', 'C')), # 1
        ('C', ('c', 'C')), # 2
        ('C', ('d',)),     # 3
    ]

    action = {}
    goto_table = {}

    for state_set, sid in state_map.items():
        for item in state_set:
            lhs, rhs, dot = item
            if dot < len(rhs):
                sym = rhs[dot]
                if sym in terminals:
                    target = trans.get((sid, sym))
                    if target is not None:
                        action[(sid, sym)] = f"s{target}"
            else:
                # Reduce item
                if lhs == "S'" and rhs == ('S',):
                    action[(sid, '$')] = "acc"
                else:
                    prod_num = productions.index((lhs, rhs))
                    follow_set = follow_C if lhs == 'C' else follow_S
                    for t in follow_set:
                        action[(sid, t)] = f"r{prod_num}"

        # GOTO entries
        for nt in ('S', 'C'):
            target = trans.get((sid, nt))
            if target is not None:
                goto_table[(sid, nt)] = target

    print("3. SLR(1) Parsing Table:")
    print(f"  {'State':>5} | {'c':>5} | {'d':>5} | {'$':>5} | {'S':>5} | {'C':>5}")
    print(f"  {'-'*5}-+{'-'*7}+{'-'*7}+{'-'*7}+{'-'*7}+{'-'*6}")
    for sid in range(len(states)):
        row = f"  {sid:>5} |"
        for t in ['c', 'd', '$']:
            entry = action.get((sid, t), '')
            row += f" {entry:>5} |"
        for nt in ['S', 'C']:
            entry = goto_table.get((sid, nt), '')
            row += f" {str(entry):>5} |"
        print(row)
    print()

    # 4. Trace parse of "c d c d"
    print("4. Parse trace for 'c d c d':")
    input_tokens = ['c', 'd', 'c', 'd', '$']
    stack = [0]
    pos = 0

    print(f"  {'Stack':<25} {'Input':<15} {'Action'}")
    print(f"  {'-'*25} {'-'*15} {'-'*25}")

    steps = 0
    while steps < 50:
        steps += 1
        state = stack[-1]
        token = input_tokens[pos]
        stack_str = ' '.join(str(s) for s in stack)
        input_str = ' '.join(input_tokens[pos:])

        act = action.get((state, token), '')
        print(f"  {stack_str:<25} {input_str:<15} {act}")

        if act == 'acc':
            print("  ACCEPTED!")
            break
        elif act.startswith('s'):
            next_state = int(act[1:])
            stack.append(token)
            stack.append(next_state)
            pos += 1
        elif act.startswith('r'):
            prod_num = int(act[1:])
            lhs, rhs = productions[prod_num]
            # Pop 2*|rhs| items from stack
            for _ in range(2 * len(rhs)):
                stack.pop()
            top_state = stack[-1]
            stack.append(lhs)
            stack.append(goto_table[(top_state, lhs)])
        else:
            print(f"  ERROR: no action for state {state}, token {token}")
            break


# === Exercise 3: SLR vs LR(1) ===
# Problem: Show grammar S->L=R|R, L->*R|id, R->L is not SLR(1) but is LR(1).

def exercise_3():
    """Demonstrate SLR vs LR(1) difference."""
    print("Grammar:")
    print("  S -> L = R | R")
    print("  L -> * R | id")
    print("  R -> L")
    print()

    print("1. Why this grammar is NOT SLR(1):")
    print()
    print("  Augmented: S' -> S")
    print()
    print("  Consider the LR(0) item set containing:")
    print("    S -> L . = R")
    print("    R -> L .")
    print()
    print("  (State reached after shifting L from start state)")
    print()
    print("  For the SLR table:")
    print("  - S -> L . = R  => shift on '=' (from the dot before '=')")
    print("  - R -> L .      => reduce by R -> L on tokens in FOLLOW(R)")
    print()
    print("  FOLLOW(R) = FOLLOW(S) U {'='}")
    print("  (Because S -> L = R, so what follows R is what follows S, which includes $)")
    print("  (Also R -> L, and L appears as S -> L = R, so '=' follows L... but we need FOLLOW(R))")
    print("  FOLLOW(R) includes '$' (from S' -> S, S -> R)")
    print("  But actually: FOLLOW(R) = {$, =}")
    print("  Wait, let's compute properly:")
    print("    FOLLOW(S) = {$}")
    print("    S -> L = R  =>  '=' in FOLLOW(L) and FOLLOW(S) subset FOLLOW(R)")
    print("    S -> R      =>  FOLLOW(S) subset FOLLOW(R), so '$' in FOLLOW(R)")
    print("    R -> L      =>  FOLLOW(R) subset FOLLOW(L)")
    print("    L -> * R    =>  FOLLOW(L) subset FOLLOW(R)")
    print("  So FOLLOW(R) = FOLLOW(L) = {$, =}")
    print()
    print("  In the state with {S -> L . = R, R -> L .}:")
    print("  - On '=': shift (from S -> L . = R)")
    print("  - On '=': reduce R -> L (because '=' is in FOLLOW(R))")
    print("  CONFLICT! Shift-reduce conflict on '='.")
    print()

    print("2. Why this grammar IS LR(1):")
    print()
    print("  In LR(1), each item includes a lookahead token.")
    print("  The critical items in the LR(1) automaton would be:")
    print("    [S -> L . = R, $]     => shift on '='")
    print("    [R -> L ., $]         => reduce on '$' only (not on '=')")
    print()
    print("  The lookahead for R -> L. is '$' (because R appears in S -> R,")
    print("  and what follows S is $).")
    print("  The lookahead does NOT include '=' for R -> L. in this state,")
    print("  because R -> L was reached from the context S -> .R (not S -> L = .R).")
    print()
    print("  So there is no conflict:")
    print("    On '=': only shift applies (from S -> L . = R)")
    print("    On '$': only reduce R -> L applies")
    print()
    print("  This is why LR(1) is strictly more powerful than SLR(1).")


# === Exercise 4: PLY-style Parser ===
# Problem: Implement LALR-style calculator with precedence and associativity.

class CalcLexer:
    """Tokenizer for calculator language."""
    def __init__(self, text):
        self.text = text
        self.pos = 0

    def tokens(self):
        result = []
        while self.pos < len(self.text):
            if self.text[self.pos].isspace():
                self.pos += 1
                continue
            if self.text[self.pos:self.pos+5] == 'print':
                result.append(('PRINT', 'print'))
                self.pos += 5
            elif self.text[self.pos:self.pos+2] == '**':
                result.append(('POWER', '**'))
                self.pos += 2
            elif self.text[self.pos].isdigit() or (self.text[self.pos] == '.' and
                    self.pos + 1 < len(self.text) and self.text[self.pos + 1].isdigit()):
                start = self.pos
                while self.pos < len(self.text) and (self.text[self.pos].isdigit() or self.text[self.pos] == '.'):
                    self.pos += 1
                val = self.text[start:self.pos]
                result.append(('NUM', float(val) if '.' in val else int(val)))
            elif self.text[self.pos].isalpha() or self.text[self.pos] == '_':
                start = self.pos
                while self.pos < len(self.text) and (self.text[self.pos].isalnum() or self.text[self.pos] == '_'):
                    self.pos += 1
                result.append(('ID', self.text[start:self.pos]))
            elif self.text[self.pos] == '+':
                result.append(('PLUS', '+'))
                self.pos += 1
            elif self.text[self.pos] == '-':
                result.append(('MINUS', '-'))
                self.pos += 1
            elif self.text[self.pos] == '*':
                result.append(('TIMES', '*'))
                self.pos += 1
            elif self.text[self.pos] == '/':
                result.append(('DIVIDE', '/'))
                self.pos += 1
            elif self.text[self.pos] == '=':
                result.append(('EQUALS', '='))
                self.pos += 1
            elif self.text[self.pos] == '(':
                result.append(('LPAREN', '('))
                self.pos += 1
            elif self.text[self.pos] == ')':
                result.append(('RPAREN', ')'))
                self.pos += 1
            else:
                raise SyntaxError(f"Unexpected char: {self.text[self.pos]}")
        result.append(('EOF', None))
        return result


class CalcParser:
    """
    Operator-precedence parser for calculator language.

    Precedence (low to high):
      1. + - (left-associative)
      2. * / (left-associative)
      3. unary - (right-associative, prefix)
      4. ** (right-associative)
    """
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.variables = {}

    def peek(self):
        return self.tokens[self.pos]

    def consume(self, expected=None):
        tok = self.tokens[self.pos]
        if expected and tok[0] != expected:
            raise SyntaxError(f"Expected {expected}, got {tok[0]}")
        self.pos += 1
        return tok

    def parse_line(self):
        """Parse a single statement."""
        if self.peek()[0] == 'PRINT':
            self.consume('PRINT')
            val = self.parse_expr()
            return ('print', val)
        elif self.peek()[0] == 'ID' and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1][0] == 'EQUALS':
            name = self.consume('ID')[1]
            self.consume('EQUALS')
            val = self.parse_expr()
            return ('assign', name, val)
        else:
            return ('expr', self.parse_expr())

    def parse_expr(self):
        """expr = term (('+' | '-') term)*"""
        left = self.parse_term()
        while self.peek()[0] in ('PLUS', 'MINUS'):
            op = self.consume()[0]
            right = self.parse_term()
            left = (op, left, right)
        return left

    def parse_term(self):
        """term = unary (('*' | '/') unary)*"""
        left = self.parse_unary()
        while self.peek()[0] in ('TIMES', 'DIVIDE'):
            op = self.consume()[0]
            right = self.parse_unary()
            left = (op, left, right)
        return left

    def parse_unary(self):
        """unary = '-' unary | power"""
        if self.peek()[0] == 'MINUS':
            self.consume()
            operand = self.parse_unary()
            return ('NEG', operand)
        return self.parse_power()

    def parse_power(self):
        """power = primary ('**' unary)?  (right-associative)"""
        base = self.parse_primary()
        if self.peek()[0] == 'POWER':
            self.consume()
            # Right-associative: parse with parse_unary (higher or same level)
            exp = self.parse_unary()
            return ('POWER', base, exp)
        return base

    def parse_primary(self):
        """primary = '(' expr ')' | NUM | ID"""
        tok = self.peek()
        if tok[0] == 'LPAREN':
            self.consume()
            val = self.parse_expr()
            self.consume('RPAREN')
            return val
        elif tok[0] == 'NUM':
            return ('NUM', self.consume()[1])
        elif tok[0] == 'ID':
            return ('ID', self.consume()[1])
        else:
            raise SyntaxError(f"Unexpected: {tok}")

    def evaluate(self, node):
        """Evaluate an AST node."""
        if isinstance(node, tuple):
            if node[0] == 'NUM':
                return node[1]
            elif node[0] == 'ID':
                if node[1] not in self.variables:
                    raise NameError(f"Undefined: {node[1]}")
                return self.variables[node[1]]
            elif node[0] == 'NEG':
                return -self.evaluate(node[1])
            elif node[0] == 'PLUS':
                return self.evaluate(node[1]) + self.evaluate(node[2])
            elif node[0] == 'MINUS':
                return self.evaluate(node[1]) - self.evaluate(node[2])
            elif node[0] == 'TIMES':
                return self.evaluate(node[1]) * self.evaluate(node[2])
            elif node[0] == 'DIVIDE':
                return self.evaluate(node[1]) / self.evaluate(node[2])
            elif node[0] == 'POWER':
                return self.evaluate(node[1]) ** self.evaluate(node[2])
            elif node[0] == 'assign':
                val = self.evaluate(node[2])
                self.variables[node[1]] = val
                return val
            elif node[0] == 'print':
                val = self.evaluate(node[1])
                return val
            elif node[0] == 'expr':
                return self.evaluate(node[1])
        return node


def exercise_4():
    """Calculator parser with precedence and associativity."""
    print("Precedence (low to high):")
    print("  1. +, -   (left-associative)")
    print("  2. *, /   (left-associative)")
    print("  3. unary- (right, prefix)")
    print("  4. **     (right-associative)")
    print()

    test_inputs = [
        "x = 2 + 3 * 4",       # x = 14
        "print x ** 2",         # 196
        "y = -3 + 7",           # y = 4
        "print x * y",          # 56
        "print 2 ** 3 ** 2",    # 2^(3^2) = 2^9 = 512 (right-assoc)
        "print (2 + 3) * 4",    # 20
        "print -x",             # -14
    ]

    parser = None
    variables = {}

    for line in test_inputs:
        lexer = CalcLexer(line)
        tokens = lexer.tokens()
        p = CalcParser(tokens)
        p.variables = variables

        ast = p.parse_line()
        result = p.evaluate(ast)
        variables = p.variables

        if ast[0] == 'print':
            print(f"  {line:<30} => {result}")
        elif ast[0] == 'assign':
            print(f"  {line:<30} => {ast[1]} = {result}")
        else:
            print(f"  {line:<30} => {result}")


# === Exercise 5: Conflict Analysis (Dangling Else) ===
# Problem: Analyze shift-reduce conflict in IF/THEN/ELSE grammar.

def exercise_5():
    """Dangling else conflict analysis."""
    print("Grammar:")
    print("  stmt : IF expr THEN stmt")
    print("       | IF expr THEN stmt ELSE stmt")
    print("       | OTHER")
    print()

    print("1. LALR(1) conflict identification:")
    print()
    print("  Consider parsing: IF e1 THEN IF e2 THEN OTHER . ELSE OTHER")
    print("  At the '.', the parser has on its stack:")
    print("    ... IF expr THEN IF expr THEN stmt .")
    print()
    print("  There are two valid continuations:")
    print()
    print("  Option A (SHIFT 'ELSE'):")
    print("    Associate ELSE with inner IF:")
    print("    IF e1 THEN (IF e2 THEN OTHER ELSE OTHER)")
    print("    The parser shifts ELSE, continues to build: IF expr THEN stmt ELSE stmt")
    print()
    print("  Option B (REDUCE to stmt):")
    print("    Reduce 'IF e2 THEN OTHER' to stmt, then the ELSE associates with outer IF:")
    print("    (IF e1 THEN (IF e2 THEN OTHER) ELSE OTHER)")
    print("    Reduce inner, then shift ELSE for outer IF")
    print()

    print("2. Effect of %left or %nonassoc on ELSE:")
    print()
    print("  %left ELSE:")
    print("    When there's a shift-reduce conflict on ELSE, the parser would REDUCE")
    print("    (because left-associativity prefers reduce over shift).")
    print("    This gives: (IF e1 THEN (IF e2 THEN s1)) ELSE s2")
    print("    The ELSE matches the OUTER if. This is WRONG by convention.")
    print()
    print("  %nonassoc ELSE:")
    print("    The conflict becomes a syntax error -- ambiguous IF statements")
    print("    would be rejected, requiring explicit braces.")
    print()

    print("3. Why 'shift on conflict' is the right default:")
    print()
    print("  Shifting ELSE means the ELSE matches the NEAREST unmatched IF.")
    print("  This is the conventional interpretation in C, Java, Python, etc.")
    print("  Example:")
    print("    if (a) if (b) x=1; else x=2;")
    print("  Shift interpretation: if(a) { if(b) x=1; else x=2; }  (CORRECT)")
    print("  Reduce interpretation: if(a) { if(b) x=1; } else x=2; (WRONG)")
    print()
    print("  Most parser generators (Yacc, Bison, PLY) default to shift on")
    print("  shift-reduce conflicts, which handles the dangling else correctly.")


# === Exercise 6: Error Recovery with Error Productions ===
# Problem: Add error productions to expression grammar for error recovery.

def exercise_6():
    """Error recovery using error productions in shift-reduce parsing."""
    print("Error productions added to expression grammar:")
    print("  E -> E + T | T | error + T | E + error")
    print("  T -> T * F | F")
    print("  F -> ( E ) | id | ( error )")
    print()

    # Implement a simple shift-reduce parser with error recovery
    # We'll use a hand-coded approach that demonstrates the concept

    def tokenize(text):
        tokens = []
        for part in text.split():
            if part == 'id':
                tokens.append(('id', 'id'))
            elif part in '()+*':
                tokens.append((part, part))
            else:
                tokens.append(('ERROR', part))
        tokens.append(('$', '$'))
        return tokens

    def parse_with_errors(text):
        """Simplified error-recovery parser for demonstration."""
        print(f"  Input: {text}")
        tokens = tokenize(text)
        pos = [0]
        errors = []

        def current():
            return tokens[pos[0]] if pos[0] < len(tokens) else ('$', '$')

        def advance():
            tok = tokens[pos[0]]
            pos[0] += 1
            return tok

        def parse_E():
            # E -> T (('+') T)*
            try:
                left = parse_T()
            except SyntaxError as e:
                # error + T recovery
                errors.append(f"Error in left operand: {e}")
                # Skip until we find '+'
                while current()[0] not in ('+', '$', ')'):
                    advance()
                left = 0  # error value
                if current()[0] != '+':
                    return left

            while current()[0] == '+':
                advance()  # consume '+'
                try:
                    right = parse_T()
                except SyntaxError as e:
                    # E + error recovery
                    errors.append(f"Error in right operand: {e}")
                    while current()[0] not in ('+', '$', ')'):
                        advance()
                    right = 0  # error value
                left = left + right
            return left

        def parse_T():
            left = parse_F()
            while current()[0] == '*':
                advance()
                right = parse_F()
                left = left * right
            return left

        def parse_F():
            tok = current()
            if tok[0] == '(':
                advance()  # consume (
                try:
                    val = parse_E()
                except SyntaxError as e:
                    # ( error ) recovery
                    errors.append(f"Error in parenthesized expr: {e}")
                    while current()[0] not in (')', '$'):
                        advance()
                    val = 0
                if current()[0] == ')':
                    advance()
                else:
                    errors.append("Missing closing parenthesis")
                return val
            elif tok[0] == 'id':
                advance()
                return 1  # symbolic value
            else:
                raise SyntaxError(f"Unexpected token '{tok[1]}' at position {pos[0]}")

        try:
            result = parse_E()
            print(f"  Result: {result} (symbolic)")
        except SyntaxError as e:
            errors.append(str(e))
            print(f"  Parse failed with error: {e}")

        if errors:
            print(f"  Recovered errors ({len(errors)}):")
            for err in errors:
                print(f"    - {err}")
        else:
            print(f"  No errors!")
        print()

    # Test cases from the exercise
    parse_with_errors("+ id * id")       # bad left operand
    parse_with_errors("id + * id")       # bad right operand (missing operand)
    parse_with_errors("( + ) * id")      # bad parenthesized expression


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: LR(0) Automaton Construction ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: SLR(1) Table Construction ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: SLR vs LR(1) ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: PLY-style Parser ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Conflict Analysis (Dangling Else) ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Error Recovery ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
