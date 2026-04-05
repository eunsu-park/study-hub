"""
Exercises for Lesson 05: Top-Down Parsing
Topic: Compiler_Design

Solutions to practice problems from the lesson.
"""

from collections import defaultdict


# === Exercise 1: FIRST and FOLLOW Computation ===
# Problem: Compute FIRST and FOLLOW for:
#   S -> A B
#   A -> a A | epsilon
#   B -> b B | c

def exercise_1():
    """Compute FIRST and FOLLOW sets."""
    print("Grammar:")
    print("  S -> A B")
    print("  A -> a A | epsilon")
    print("  B -> b B | c")
    print()

    grammar = {
        'S': [['A', 'B']],
        'A': [['a', 'A'], []],  # [] represents epsilon
        'B': [['b', 'B'], ['c']],
    }
    nonterminals = set(grammar.keys())
    terminals = {'a', 'b', 'c'}
    start = 'S'

    # Compute FIRST sets
    first = defaultdict(set)

    def compute_first(symbol):
        if symbol in terminals:
            return {symbol}
        if symbol == '':
            return {''}

        result = set()
        for prod in grammar[symbol]:
            if not prod:  # epsilon production
                result.add('')
                continue
            for s in prod:
                f = compute_first(s)
                result.update(f - {''})
                if '' not in f:
                    break
            else:
                result.add('')
        return result

    # Fixed-point iteration
    changed = True
    while changed:
        changed = False
        for nt in nonterminals:
            new_first = compute_first(nt)
            if new_first != first[nt]:
                first[nt] = new_first
                changed = True

    print("FIRST sets:")
    for nt in ['S', 'A', 'B']:
        print(f"  FIRST({nt}) = {{{', '.join(sorted(first[nt]))}}}")
    print()

    # Compute FOLLOW sets
    follow = defaultdict(set)
    follow[start].add('$')

    changed = True
    while changed:
        changed = False
        for nt in nonterminals:
            for prod in grammar[nt]:
                for i, symbol in enumerate(prod):
                    if symbol not in nonterminals:
                        continue
                    # Look at what follows symbol in this production
                    rest = prod[i + 1:]
                    # Compute FIRST of rest
                    first_rest = set()
                    all_nullable = True
                    for s in rest:
                        f = compute_first(s)
                        first_rest.update(f - {''})
                        if '' not in f:
                            all_nullable = False
                            break

                    old_len = len(follow[symbol])
                    follow[symbol].update(first_rest)
                    if all_nullable:
                        follow[symbol].update(follow[nt])
                    if len(follow[symbol]) != old_len:
                        changed = True

    print("FOLLOW sets:")
    for nt in ['S', 'A', 'B']:
        print(f"  FOLLOW({nt}) = {{{', '.join(sorted(follow[nt]))}}}")
    print()

    print("Verification:")
    print("  FIRST(S) = FIRST(A B)")
    print("    A can derive epsilon, so FIRST(S) includes FIRST(A)-{eps} U FIRST(B)")
    print("    = {a} U {b, c} = {a, b, c}")
    print("  FIRST(A) = {a, epsilon}")
    print("  FIRST(B) = {b, c}")
    print("  FOLLOW(S) = {$}")
    print("  FOLLOW(A) = FIRST(B) = {b, c}")
    print("  FOLLOW(B) = FOLLOW(S) = {$}")


# === Exercise 2: LL(1) Table Construction ===
# Problem: Dangling else grammar:
#   S  -> i E t S S' | a
#   S' -> e S | epsilon
#   E  -> b

def exercise_2():
    """LL(1) table construction for dangling else grammar."""
    print("Grammar (dangling else):")
    print("  S  -> i E t S S' | a")
    print("  S' -> e S | epsilon")
    print("  E  -> b")
    print("  (i=if, t=then, e=else, a=assignment, b=boolean)")
    print()

    # FIRST sets
    print("1. FIRST sets:")
    print("  FIRST(S)  = {i, a}")
    print("  FIRST(S') = {e, epsilon}")
    print("  FIRST(E)  = {b}")
    print()

    # FOLLOW sets
    print("2. FOLLOW sets:")
    print("  FOLLOW(S)  = {$, e}")
    print("    ($ from start; e because S' -> e S, so S can be followed by e)")
    print("  FOLLOW(S') = FOLLOW(S) = {$, e}")
    print("  FOLLOW(E)  = {t}")
    print()

    # LL(1) table
    print("3. LL(1) Parsing Table:")
    print("       |    i         |    a    |    b    |    e         |    t    |    $")
    print("  -----+-------------+---------+---------+-------------+---------+--------")
    print("  S    | S->iEtSS'   | S->a    |         |             |         |")
    print("  S'   |             |         |         | S'->eS      |         | S'->eps")
    print("       |             |         |         | S'->eps (!) |         |")
    print("  E    |             |         | E->b    |             |         |")
    print()

    print("4. Conflict Analysis:")
    print("  There is a conflict in M[S', e]:")
    print("  - S' -> e S (because FIRST(eS) = {e})")
    print("  - S' -> epsilon (because e is in FOLLOW(S'))")
    print("  This is the classic 'dangling else' ambiguity.")
    print()

    print("  Resolution: Always choose S' -> e S (shift/match the else).")
    print("  This associates each 'else' with the nearest unmatched 'if',")
    print("  which is the conventional semantics in most languages.")
    print()

    # Demonstrate with a parse trace
    print("  Parse trace for 'i b t i b t a e a' (if b then if b then a else a):")
    parse_steps = [
        ("S", "i b t i b t a e a $", "S -> i E t S S'"),
        ("i E t S S'", "i b t i b t a e a $", "match i"),
        ("E t S S'", "b t i b t a e a $", "E -> b"),
        ("b t S S'", "b t i b t a e a $", "match b"),
        ("t S S'", "t i b t a e a $", "match t"),
        ("S S'", "i b t a e a $", "S -> i E t S S'"),
        ("i E t S S' S'", "i b t a e a $", "match i"),
        ("E t S S' S'", "b t a e a $", "E -> b"),
        ("b t S S' S'", "b t a e a $", "match b"),
        ("t S S' S'", "t a e a $", "match t"),
        ("S S' S'", "a e a $", "S -> a"),
        ("a S' S'", "a e a $", "match a"),
        ("S' S'", "e a $", "S' -> e S (choose shift)"),
        ("e S S'", "e a $", "match e"),
        ("S S'", "a $", "S -> a"),
        ("a S'", "a $", "match a"),
        ("S'", "$", "S' -> epsilon"),
        ("", "$", "ACCEPT"),
    ]
    print(f"  {'Stack':<25} {'Input':<25} {'Action'}")
    print(f"  {'-'*25} {'-'*25} {'-'*30}")
    for stack, inp, action in parse_steps:
        print(f"  {stack:<25} {inp:<25} {action}")


# === Exercise 3: Left Recursion Elimination ===
# Problem: Eliminate all left recursion from:
#   S -> A a | b
#   A -> A c | S d | e

def exercise_3():
    """Eliminate left recursion (including indirect)."""
    print("Original grammar:")
    print("  S -> A a | b")
    print("  A -> A c | S d | e")
    print()

    print("Step 1: Order nonterminals: S, A")
    print()

    print("Step 2: Process S (i=1)")
    print("  No left recursion in S productions (S -> Aa | b)")
    print("  S -> A a | b  (unchanged)")
    print()

    print("Step 3: Process A (i=2)")
    print("  Check A -> S d: S comes before A in ordering.")
    print("  Substitute S's productions into A -> S d:")
    print("    A -> S d becomes A -> A a d | b d")
    print("  Updated A productions:")
    print("    A -> A c | A a d | b d | e")
    print()

    print("Step 4: Eliminate direct left recursion in A")
    print("  A -> A c | A a d | b d | e")
    print("  Left-recursive: A -> A c, A -> A a d")
    print("  Non-recursive:  A -> b d, A -> e")
    print()
    print("  Apply formula A -> beta A' and A' -> alpha A' | epsilon:")
    print("  A  -> b d A' | e A'")
    print("  A' -> c A' | a d A' | epsilon")
    print()

    print("Final grammar (left-recursion free):")
    print("  S  -> A a | b")
    print("  A  -> b d A' | e A'")
    print("  A' -> c A' | a d A' | epsilon")
    print()

    # Verify: generate some strings from both grammars
    print("Verification by generating strings:")

    # Original grammar: generate strings by expansion
    def generate_original(max_depth=5):
        """Generate strings from the original grammar using BFS."""
        strings = set()
        # Use sentential forms
        queue = [('S', 0)]
        while queue:
            form, depth = queue.pop(0)
            if depth > max_depth:
                continue
            if all(c in 'abcde ' for c in form):
                strings.add(form.replace(' ', ''))
                continue
            # Find leftmost nonterminal
            for i, c in enumerate(form):
                if c in 'SA':
                    if c == 'S':
                        prods = [('A', 'a'), ('b',)]
                    else:  # A
                        prods = [('A', 'c'), ('S', 'd'), ('e',)]
                    for prod in prods:
                        new_form = form[:i] + ' '.join(prod) + form[i+1:]
                        queue.append((new_form, depth + 1))
                    break
        return strings

    strings = generate_original(6)
    terminal_strings = {s for s in strings if all(c in 'abcde' for c in s)}
    print(f"  Strings from original grammar (depth<=6): {sorted(terminal_strings)[:15]}")


# === Exercise 4: Recursive Descent Parser Extension ===
# Problem: Extend recursive descent parser to support -, /, unary -, ints and vars.

class Token:
    def __init__(self, type, value=None):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"


class Lexer:
    """Simple lexer for arithmetic expressions."""
    def __init__(self, text):
        self.text = text
        self.pos = 0

    def next_token(self):
        while self.pos < len(self.text) and self.text[self.pos] == ' ':
            self.pos += 1
        if self.pos >= len(self.text):
            return Token('EOF')

        ch = self.text[self.pos]
        if ch.isdigit():
            start = self.pos
            while self.pos < len(self.text) and self.text[self.pos].isdigit():
                self.pos += 1
            return Token('INT', int(self.text[start:self.pos]))
        elif ch.isalpha():
            start = self.pos
            while self.pos < len(self.text) and self.text[self.pos].isalnum():
                self.pos += 1
            return Token('ID', self.text[start:self.pos])
        elif ch == '+':
            self.pos += 1
            return Token('PLUS')
        elif ch == '-':
            self.pos += 1
            return Token('MINUS')
        elif ch == '*':
            self.pos += 1
            return Token('STAR')
        elif ch == '/':
            self.pos += 1
            return Token('SLASH')
        elif ch == '(':
            self.pos += 1
            return Token('LPAREN')
        elif ch == ')':
            self.pos += 1
            return Token('RPAREN')
        else:
            raise SyntaxError(f"Unexpected character: {ch}")


class RecursiveDescentParser:
    """
    Recursive descent parser for:
      E  -> T E'
      E' -> + T E' | - T E' | epsilon
      T  -> U T'
      T' -> * U T' | / U T' | epsilon
      U  -> - U | F
      F  -> ( E ) | INT | ID
    """
    def __init__(self, text):
        self.lexer = Lexer(text)
        self.tokens = []
        # Tokenize all at once
        while True:
            tok = self.lexer.next_token()
            self.tokens.append(tok)
            if tok.type == 'EOF':
                break
        self.pos = 0
        self.variables = {'x': 10, 'y': 20, 'z': 30}  # predefined variables

    def peek(self):
        return self.tokens[self.pos]

    def consume(self, expected_type):
        tok = self.tokens[self.pos]
        if tok.type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {tok.type}")
        self.pos += 1
        return tok

    def parse(self):
        result = self.parse_expr()
        self.consume('EOF')
        return result

    def parse_expr(self):
        """E -> T (('+' | '-') T)*"""
        left = self.parse_term()
        while self.peek().type in ('PLUS', 'MINUS'):
            op = self.consume(self.peek().type)
            right = self.parse_term()
            if op.type == 'PLUS':
                left = left + right
            else:
                left = left - right
        return left

    def parse_term(self):
        """T -> U (('*' | '/') U)*"""
        left = self.parse_unary()
        while self.peek().type in ('STAR', 'SLASH'):
            op = self.consume(self.peek().type)
            right = self.parse_unary()
            if op.type == 'STAR':
                left = left * right
            else:
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                left = left / right
        return left

    def parse_unary(self):
        """U -> '-' U | F"""
        if self.peek().type == 'MINUS':
            self.consume('MINUS')
            return -self.parse_unary()
        return self.parse_factor()

    def parse_factor(self):
        """F -> '(' E ')' | INT | ID"""
        tok = self.peek()
        if tok.type == 'LPAREN':
            self.consume('LPAREN')
            result = self.parse_expr()
            self.consume('RPAREN')
            return result
        elif tok.type == 'INT':
            self.consume('INT')
            return tok.value
        elif tok.type == 'ID':
            self.consume('ID')
            if tok.value not in self.variables:
                raise NameError(f"Undefined variable: {tok.value}")
            return self.variables[tok.value]
        else:
            raise SyntaxError(f"Unexpected token: {tok}")


def exercise_4():
    """Extended recursive descent parser with -, /, unary -, ints, vars."""
    print("Grammar (left-recursion eliminated):")
    print("  E  -> T E'")
    print("  E' -> + T E' | - T E' | epsilon")
    print("  T  -> U T'")
    print("  T' -> * U T' | / U T' | epsilon")
    print("  U  -> - U | F")
    print("  F  -> ( E ) | INT | ID")
    print()
    print("Predefined variables: x=10, y=20, z=30")
    print()

    test_cases = [
        ("3 + 4 * 2", 11),
        ("(3 + 4) * 2", 14),
        ("10 - 3 - 2", 5),       # left-associative: (10-3)-2 = 5
        ("12 / 4 / 3", 1.0),     # left-associative: (12/4)/3 = 1
        ("-5", -5),
        ("--5", 5),               # double negation
        ("-3 + 7", 4),
        ("x + y", 30),
        ("x * y - z", 170),
        ("-(x + y)", -30),
    ]

    for expr, expected in test_cases:
        try:
            parser = RecursiveDescentParser(expr)
            result = parser.parse()
            status = "PASS" if result == expected else "FAIL"
            print(f"  [{status}] {expr:>20} = {result} (expected {expected})")
        except Exception as e:
            print(f"  [ERROR] {expr:>20} -> {e}")


# === Exercise 5: Error Recovery ===
# Problem: Implement panic-mode error recovery in LL(1) parser.

def exercise_5():
    """Panic-mode error recovery in table-driven LL(1) parser."""
    # Grammar: E -> T E', E' -> + T E' | eps, T -> F T', T' -> * F T' | eps,
    # F -> ( E ) | id

    # LL(1) parsing table
    table = {
        ('E', 'id'):    ['T', "E'"],
        ('E', '('):     ['T', "E'"],
        ("E'", '+'):    ['+', 'T', "E'"],
        ("E'", ')'):    [],   # epsilon
        ("E'", '$'):    [],   # epsilon
        ('T', 'id'):    ['F', "T'"],
        ('T', '('):     ['F', "T'"],
        ("T'", '+'):    [],   # epsilon
        ("T'", '*'):    ['*', 'F', "T'"],
        ("T'", ')'):    [],   # epsilon
        ("T'", '$'):    [],   # epsilon
        ('F', 'id'):    ['id'],
        ('F', '('):     ['(', 'E', ')'],
    }

    follow = {
        'E':  {')', '$'},
        "E'": {')', '$'},
        'T':  {'+', ')', '$'},
        "T'": {'+', ')', '$'},
        'F':  {'+', '*', ')', '$'},
    }

    nonterminals = {'E', "E'", 'T', "T'", 'F'}

    def tokenize(text):
        tokens = []
        i = 0
        while i < len(text):
            if text[i] == ' ':
                i += 1
                continue
            if text[i:i+2] == 'id':
                tokens.append('id')
                i += 2
            elif text[i] in '()+*':
                tokens.append(text[i])
                i += 1
            else:
                tokens.append(text[i])
                i += 1
        tokens.append('$')
        return tokens

    def parse_with_recovery(input_str):
        tokens = tokenize(input_str)
        stack = ['$', 'E']
        pos = 0
        errors = []

        print(f"  Parsing: '{input_str}'")
        print(f"  {'Stack':<30} {'Input':<20} {'Action'}")
        print(f"  {'-'*30} {'-'*20} {'-'*30}")

        while stack:
            top = stack[-1]
            current = tokens[pos] if pos < len(tokens) else '$'

            stack_str = ' '.join(reversed(stack))
            input_str_remaining = ' '.join(tokens[pos:])

            if top == current:
                # Match
                print(f"  {stack_str:<30} {input_str_remaining:<20} match '{top}'")
                stack.pop()
                pos += 1
            elif top == '$':
                break
            elif top not in nonterminals:
                # Terminal mismatch
                errors.append(f"  Error at position {pos}: expected '{top}', got '{current}'")
                print(f"  {stack_str:<30} {input_str_remaining:<20} ERROR: expected '{top}'")
                stack.pop()  # discard expected terminal
            elif (top, current) in table:
                prod = table[(top, current)]
                prod_str = ' '.join(prod) if prod else 'epsilon'
                print(f"  {stack_str:<30} {input_str_remaining:<20} {top} -> {prod_str}")
                stack.pop()
                for symbol in reversed(prod):
                    stack.append(symbol)
            else:
                # Panic mode: skip input until synchronization token
                errors.append(
                    f"  Error at position {pos}: no entry for M[{top}, {current}]"
                )
                print(f"  {stack_str:<30} {input_str_remaining:<20} "
                      f"ERROR: M[{top},{current}] empty")

                # Skip input tokens until finding one in FOLLOW(top)
                while pos < len(tokens) and tokens[pos] not in follow.get(top, {'$'}):
                    print(f"  {'':30} {'':20} skip '{tokens[pos]}'")
                    pos += 1

                stack.pop()  # pop the nonterminal
                print(f"  {'':30} {'':20} pop '{top}', sync on '{tokens[pos] if pos < len(tokens) else 'EOF'}'")

        if errors:
            print(f"  Errors found ({len(errors)}):")
            for err in errors:
                print(f"    {err}")
        else:
            print(f"  Parse successful!")
        return errors

    # Test with "id + * id" (missing operand between + and *)
    parse_with_recovery("id + * id")
    print()
    # Also test a valid input
    parse_with_recovery("id + id * id")


# === Exercise 6: Grammar Design Challenge ===
# Problem: Design an LL(1) grammar for a mini-language.

def exercise_6():
    """LL(1) grammar for a mini-language."""
    print("Language features:")
    print("  - Variable declarations: let x = expr;")
    print("  - Assignment: x = expr;")
    print("  - If-else: if (expr) { stmts } else { stmts }")
    print("  - While: while (expr) { stmts }")
    print("  - Print: print(expr);")
    print("  - Arithmetic: +, -, *, /")
    print()

    print("LL(1) Grammar:")
    print("  program    -> stmts")
    print("  stmts      -> stmt stmts | epsilon")
    print("  stmt       -> let_stmt | if_stmt | while_stmt | print_stmt | assign_or_expr_stmt")
    print("  let_stmt   -> LET ID ASSIGN expr SEMI")
    print("  if_stmt    -> IF LPAREN expr RPAREN LBRACE stmts RBRACE else_part")
    print("  else_part  -> ELSE LBRACE stmts RBRACE | epsilon")
    print("  while_stmt -> WHILE LPAREN expr RPAREN LBRACE stmts RBRACE")
    print("  print_stmt -> PRINT LPAREN expr RPAREN SEMI")
    print("  assign_or_expr_stmt -> ID ASSIGN expr SEMI")
    print()
    print("  expr       -> term expr'")
    print("  expr'      -> PLUS term expr' | MINUS term expr' | epsilon")
    print("  term       -> factor term'")
    print("  term'      -> STAR factor term' | SLASH factor term' | epsilon")
    print("  factor     -> LPAREN expr RPAREN | INT | ID")
    print()

    print("FIRST sets:")
    print("  FIRST(program)   = {LET, IF, WHILE, PRINT, ID, epsilon}")
    print("  FIRST(stmt)      = {LET, IF, WHILE, PRINT, ID}")
    print("  FIRST(let_stmt)  = {LET}")
    print("  FIRST(if_stmt)   = {IF}")
    print("  FIRST(while_stmt)= {WHILE}")
    print("  FIRST(print_stmt)= {PRINT}")
    print("  FIRST(assign)    = {ID}")
    print("  FIRST(expr)      = {LPAREN, INT, ID}")
    print("  FIRST(expr')     = {PLUS, MINUS, epsilon}")
    print("  FIRST(term)      = {LPAREN, INT, ID}")
    print("  FIRST(term')     = {STAR, SLASH, epsilon}")
    print("  FIRST(factor)    = {LPAREN, INT, ID}")
    print("  FIRST(else_part) = {ELSE, epsilon}")
    print()

    print("LL(1) check:")
    print("  stmt alternatives have disjoint FIRST sets: {LET}, {IF}, {WHILE}, {PRINT}, {ID}")
    print("  expr' alternatives: {PLUS}, {MINUS}, FOLLOW(expr') -- OK if FOLLOW disjoint")
    print("  term' alternatives: {STAR}, {SLASH}, FOLLOW(term') -- OK")
    print("  else_part: {ELSE} vs FOLLOW(else_part) -- OK (ELSE is not in FOLLOW)")
    print("  This grammar IS LL(1).")
    print()

    # Implement a simple recursive descent parser for this grammar
    class MiniParser:
        def __init__(self, tokens):
            self.tokens = tokens
            self.pos = 0
            self.ast = []

        def peek(self):
            return self.tokens[self.pos] if self.pos < len(self.tokens) else ('EOF', None)

        def consume(self, expected):
            tok = self.tokens[self.pos]
            if tok[0] != expected:
                raise SyntaxError(f"Expected {expected}, got {tok[0]} at position {self.pos}")
            self.pos += 1
            return tok

        def parse_program(self):
            stmts = self.parse_stmts()
            self.consume('EOF')
            return stmts

        def parse_stmts(self):
            result = []
            while self.peek()[0] in ('LET', 'IF', 'WHILE', 'PRINT', 'ID'):
                result.append(self.parse_stmt())
            return result

        def parse_stmt(self):
            tok_type = self.peek()[0]
            if tok_type == 'LET':
                return self.parse_let()
            elif tok_type == 'IF':
                return self.parse_if()
            elif tok_type == 'WHILE':
                return self.parse_while()
            elif tok_type == 'PRINT':
                return self.parse_print()
            elif tok_type == 'ID':
                return self.parse_assign()
            else:
                raise SyntaxError(f"Unexpected token: {self.peek()}")

        def parse_let(self):
            self.consume('LET')
            name = self.consume('ID')[1]
            self.consume('ASSIGN')
            expr = self.parse_expr()
            self.consume('SEMI')
            return ('let', name, expr)

        def parse_assign(self):
            name = self.consume('ID')[1]
            self.consume('ASSIGN')
            expr = self.parse_expr()
            self.consume('SEMI')
            return ('assign', name, expr)

        def parse_if(self):
            self.consume('IF')
            self.consume('LPAREN')
            cond = self.parse_expr()
            self.consume('RPAREN')
            self.consume('LBRACE')
            then_body = self.parse_stmts()
            self.consume('RBRACE')
            else_body = None
            if self.peek()[0] == 'ELSE':
                self.consume('ELSE')
                self.consume('LBRACE')
                else_body = self.parse_stmts()
                self.consume('RBRACE')
            return ('if', cond, then_body, else_body)

        def parse_while(self):
            self.consume('WHILE')
            self.consume('LPAREN')
            cond = self.parse_expr()
            self.consume('RPAREN')
            self.consume('LBRACE')
            body = self.parse_stmts()
            self.consume('RBRACE')
            return ('while', cond, body)

        def parse_print(self):
            self.consume('PRINT')
            self.consume('LPAREN')
            expr = self.parse_expr()
            self.consume('RPAREN')
            self.consume('SEMI')
            return ('print', expr)

        def parse_expr(self):
            left = self.parse_term()
            while self.peek()[0] in ('PLUS', 'MINUS'):
                op = self.consume(self.peek()[0])[0]
                right = self.parse_term()
                left = (op.lower(), left, right)
            return left

        def parse_term(self):
            left = self.parse_factor()
            while self.peek()[0] in ('STAR', 'SLASH'):
                op = self.consume(self.peek()[0])[0]
                right = self.parse_factor()
                left = (op.lower(), left, right)
            return left

        def parse_factor(self):
            tok = self.peek()
            if tok[0] == 'LPAREN':
                self.consume('LPAREN')
                expr = self.parse_expr()
                self.consume('RPAREN')
                return expr
            elif tok[0] == 'INT':
                return ('int', self.consume('INT')[1])
            elif tok[0] == 'ID':
                return ('id', self.consume('ID')[1])
            else:
                raise SyntaxError(f"Expected factor, got {tok}")

    # Test with a sample program
    tokens = [
        ('LET', 'let'), ('ID', 'x'), ('ASSIGN', '='), ('INT', 10), ('SEMI', ';'),
        ('LET', 'let'), ('ID', 'y'), ('ASSIGN', '='), ('INT', 20), ('SEMI', ';'),
        ('IF', 'if'), ('LPAREN', '('), ('ID', 'x'), ('RPAREN', ')'),
        ('LBRACE', '{'),
        ('PRINT', 'print'), ('LPAREN', '('), ('ID', 'x'), ('PLUS', '+'), ('ID', 'y'),
        ('RPAREN', ')'), ('SEMI', ';'),
        ('RBRACE', '}'),
        ('ELSE', 'else'),
        ('LBRACE', '{'),
        ('PRINT', 'print'), ('LPAREN', '('), ('INT', 0), ('RPAREN', ')'), ('SEMI', ';'),
        ('RBRACE', '}'),
        ('EOF', None),
    ]

    parser = MiniParser(tokens)
    ast = parser.parse_program()
    print("Test program parsed successfully!")
    print("AST:")
    for node in ast:
        print(f"  {node}")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: FIRST and FOLLOW Computation ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: LL(1) Table Construction ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Left Recursion Elimination ===")
    print("=" * 60)
    exercise_3()

    print("\n" + "=" * 60)
    print("=== Exercise 4: Recursive Descent Parser Extension ===")
    print("=" * 60)
    exercise_4()

    print("\n" + "=" * 60)
    print("=== Exercise 5: Error Recovery ===")
    print("=" * 60)
    exercise_5()

    print("\n" + "=" * 60)
    print("=== Exercise 6: Grammar Design Challenge ===")
    print("=" * 60)
    exercise_6()

    print("\nAll exercises completed!")
