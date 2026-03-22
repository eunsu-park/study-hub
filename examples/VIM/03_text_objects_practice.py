"""
VIM TEXT OBJECTS PRACTICE
=========================

Open this file in Vim and practice text objects.
Text objects use 'i' (inner) and 'a' (around) with operator commands.

Syntax: {operator}{i/a}{object}
  di"  = delete inner double-quotes
  ca(  = change around parentheses
  yiw  = yank inner word

Practice each exercise by placing your cursor INSIDE the target
and executing the command. Use 'u' to undo and try again.
"""

# ============================================================================
# EXERCISE 1: Word Objects (iw, aw)
# ============================================================================

# Place cursor anywhere on "variable" and try:
#   diw  → deletes "variable" (keeps surrounding spaces)
#   daw  → deletes "variable" + one adjacent space
#   ciw  → deletes "variable" and enters Insert mode
#   yiw  → yanks "variable" (try pasting with p elsewhere)

my_variable = "hello"
another_variable = 42
some_other_variable = True


# ============================================================================
# EXERCISE 2: Quote Objects (i", a", i', a')
# ============================================================================

# Place cursor anywhere INSIDE the quotes and try:
#   di"  → deletes content inside quotes (keeps quotes)
#   da"  → deletes content + the quotes themselves
#   ci"  → change inside quotes (type new content, then Esc)
#   yi"  → yank inside quotes

greeting = "Hello, World!"
name = "Alice"
path = '/usr/local/bin/python3'
message = "The quick brown fox jumps over the lazy dog"

# Try changing "Alice" to "Bob": cursor inside quotes, ci", type Bob, Esc
# Try deleting the entire string including quotes: da"


# ============================================================================
# EXERCISE 3: Parenthesis Objects (i(, a(, i), a))
# ============================================================================

# Place cursor anywhere INSIDE the parentheses and try:
#   di(  → deletes arguments (keeps parentheses)
#   da(  → deletes arguments + parentheses
#   ci(  → change arguments
#   yi(  → yank arguments

def calculate_total(price, quantity, tax_rate):
    """Calculate the total price with tax."""
    subtotal = price * quantity
    return round(subtotal * (1 + tax_rate), 2)


result = calculate_total(19.99, 3, 0.08)
print(f"Total: ${result}")

# Try:
# 1. Cursor inside (19.99, 3, 0.08), type ci( to change arguments
# 2. Cursor inside (price, quantity, tax_rate), type yi( to yank params


# ============================================================================
# EXERCISE 4: Bracket Objects (i[, a[, i{, a{)
# ============================================================================

# Square brackets
colors = ["red", "green", "blue", "yellow"]
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Curly braces (dictionaries)
config = {
    "host": "localhost",
    "port": 8080,
    "debug": True,
    "database": {
        "name": "myapp",
        "user": "admin",
    },
}

# Try:
# 1. Cursor inside ["red"...], type di[ → deletes list contents
# 2. Cursor inside {"host"...}, type ci{ → change dict contents
# 3. Cursor inside inner {"name"...}, type da{ → delete inner dict + braces


# ============================================================================
# EXERCISE 5: Sentence and Paragraph Objects (is, as, ip, ap)
# ============================================================================

# Sentences (separated by . or ! or ? followed by space):

# This is the first sentence. This is the second sentence. And the third!
# Try: cursor in "second sentence", type dis to delete it.
# Try: cursor in "second sentence", type das to delete it + trailing space.


def function_one():
    """First function."""
    return 1

def function_two():
    """Second function."""
    return 2

def function_three():
    """Third function."""
    return 3

# Paragraph objects treat blank-line-separated blocks as paragraphs.
# Try: cursor inside function_two, type dip to delete the paragraph.
# Try: cursor inside function_two, type dap to delete paragraph + blank line.


# ============================================================================
# EXERCISE 6: Combining Text Objects with Operators
# ============================================================================

class ShoppingCart:
    """A simple shopping cart implementation."""

    def __init__(self):
        self.items = []
        self.discount = 0.0

    def add_item(self, name, price, quantity=1):
        """Add an item to the cart."""
        item = {
            "name": name,
            "price": price,
            "quantity": quantity,
        }
        self.items.append(item)

    def get_total(self):
        """Calculate the total price."""
        total = sum(
            item["price"] * item["quantity"]
            for item in self.items
        )
        return total * (1 - self.discount)

    def apply_discount(self, percentage):
        """Apply a discount (0.0 to 1.0)."""
        self.discount = min(max(percentage, 0.0), 1.0)

    def __str__(self):
        return f"Cart({len(self.items)} items, total=${self.get_total():.2f})"


# CHALLENGE: Perform these edits using text objects:
#
# 1. Change the class docstring: ci" on the docstring
# 2. Delete add_item's body: go inside the method, use dip
# 3. Change the discount default: cursor on 0.0, ciw
# 4. Yank the get_total method: cursor inside it, yip
# 5. Delete the format string in __str__: ci" on the f-string


# ============================================================================
# EXERCISE 7: Speed Challenge
# ============================================================================

# Transform each line using the FEWEST keystrokes possible.
# Record your keystroke count and try to beat these targets:

# Task 1: Change "old" to "new" → ci"  (target: 6 keys: ci"new<Esc>)
data = {"key": "old"}

# Task 2: Delete function params → di(  (target: 3 keys: di()
def unnecessary_function(param1, param2, param3):
    pass

# Task 3: Empty the list → di[  (target: 3 keys: di[)
items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Task 4: Change dict values (multi-step challenge)
user = {
    "name": "CHANGEME",
    "email": "CHANGEME",
    "role": "CHANGEME",
}
# For each "CHANGEME": cursor inside quotes, ci", type new value, Esc
# Use n (next search match) + . (dot repeat) if you search for CHANGEME first
