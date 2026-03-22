"""
Exercises for Lesson 08: Clean Code & Code Smells
Topic: Programming

Solutions to practice problems from the lesson.
"""


# === Exercise 1: Refactor This Code ===
# Problem: Refactor cryptic order processing code to follow clean code principles.

def exercise_1():
    """Solution: Refactor poorly named order price calculator."""

    # Original cryptic code:
    # def p(o):
    #     t = 0
    #     for i in o['items']:
    #         t += i['p'] * i['q']
    #     if o.get('c'):
    #         d = 0.1 if o['c'] == 'SAVE10' else 0.2 if o['c'] == 'SAVE20' else 0
    #         t = t - (t * d)
    #     return t

    # Refactored: meaningful names, extracted functions, documented intent
    DISCOUNT_CODES = {
        "SAVE10": 0.10,
        "SAVE20": 0.20,
    }

    def calculate_subtotal(items):
        """Sum up price * quantity for all items in the order."""
        return sum(item["price"] * item["quantity"] for item in items)

    def get_discount_rate(coupon_code):
        """Look up the discount percentage for a coupon code. Returns 0 if invalid."""
        return DISCOUNT_CODES.get(coupon_code, 0)

    def apply_discount(subtotal, discount_rate):
        """Reduce subtotal by the discount percentage."""
        return subtotal * (1 - discount_rate)

    def calculate_order_total(order):
        """
        Calculate the total price of an order, applying any coupon discount.

        Args:
            order: Dict with 'items' (list of {price, quantity}) and
                   optional 'coupon_code'.

        Returns:
            Final order total after discount.
        """
        subtotal = calculate_subtotal(order["items"])

        coupon_code = order.get("coupon_code")
        if coupon_code:
            discount_rate = get_discount_rate(coupon_code)
            return apply_discount(subtotal, discount_rate)

        return subtotal

    # Test
    order_no_coupon = {
        "items": [
            {"price": 10.0, "quantity": 2},
            {"price": 25.0, "quantity": 1},
        ]
    }

    order_with_coupon = {
        "items": [
            {"price": 10.0, "quantity": 2},
            {"price": 25.0, "quantity": 1},
        ],
        "coupon_code": "SAVE10",
    }

    print(f"  No coupon: ${calculate_order_total(order_no_coupon):.2f}")
    print(f"  With SAVE10: ${calculate_order_total(order_with_coupon):.2f}")
    print(f"  Expected: $45.00 and $40.50")


# === Exercise 2: Identify Code Smells ===
# Problem: Find at least 5 code smells in UserManager.

def exercise_2():
    """Solution: Identify code smells in the UserManager class."""

    smells = {
        "1. God Method (Long Method)": (
            "process() handles create, update, and delete in one method "
            "with 150+ lines. Each branch should be a separate method."
        ),
        "2. Feature Envy / Type Code": (
            "Using string 'type' parameter to dispatch behavior. "
            "This should use polymorphism (Command pattern) or separate methods."
        ),
        "3. Duplicate Code": (
            "getUserFirstName, getUserLastName, getUserEmail all run the "
            "same query. The query should be extracted into a single "
            "getUser() method that returns all fields."
        ),
        "4. SQL Injection Vulnerability": (
            "String concatenation for SQL: 'WHERE id = ' + id. "
            "Must use parameterized queries to prevent injection attacks."
        ),
        "5. Primitive Obsession": (
            "Returns raw String arrays (data[0], data[1]) instead of a "
            "User object. Magic indices are fragile and unclear."
        ),
        "6. Shotgun Surgery": (
            "If the database schema changes (column order), ALL three "
            "get methods must be updated. A single User model fixes this."
        ),
        "7. Missing Abstraction": (
            "No User model class. Data is raw strings/arrays without "
            "type safety or validation."
        ),
    }

    for smell, explanation in smells.items():
        print(f"  {smell}")
        print(f"    {explanation}")
        print()

    # Show the refactored version
    print("  Refactored approach:")
    print("    1. Split process() into create_user(), update_user(), delete_user()")
    print("    2. Create a User dataclass with named fields")
    print("    3. Single _get_user(id) method that returns a User object")
    print("    4. Use parameterized queries: cursor.execute('... WHERE id = ?', (id,))")


# === Exercise 3: Write Clean Functions ===
# Problem: Rewrite validateAndProcessForm with clean code principles.

def exercise_3():
    """Solution: Clean refactoring of form validation and processing."""

    # Original: one function doing validation, user creation, email, and redirect.
    # Problems: multiple responsibilities, no separation of concerns,
    # validation mixed with side effects.

    # Clean version: separate validation from processing

    class ValidationError(Exception):
        """Raised when form data fails validation."""
        pass

    def validate_email(email):
        """Validate email format. Raises ValidationError if invalid."""
        if not email or "@" not in email:
            raise ValidationError("Invalid email address")

    def validate_password(password, confirm_password):
        """Validate password strength and confirmation match."""
        if not password or len(password) < 8:
            raise ValidationError("Password must be at least 8 characters")
        if password != confirm_password:
            raise ValidationError("Passwords do not match")

    def validate_form(form_data):
        """
        Validate all form fields. Returns None on success, raises on failure.

        Separating validation into its own function makes it testable
        and reusable without triggering side effects.
        """
        validate_email(form_data.get("email", ""))
        validate_password(
            form_data.get("password", ""),
            form_data.get("confirm_password", ""),
        )

    def create_user_from_form(form_data):
        """Create a user dict from validated form data."""
        return {
            "email": form_data["email"],
            "password": f"hashed({form_data['password']})",  # Simulated hash
            "created_at": "2026-02-27",
        }

    def process_registration(form_data):
        """
        Main registration flow: validate, create, notify.

        Each step is a clean, focused function. Errors are handled
        at this orchestration level.
        """
        try:
            validate_form(form_data)
            user = create_user_from_form(form_data)
            # In real code: db.save(user), send_welcome_email(user), redirect()
            return True, f"User {user['email']} registered successfully"
        except ValidationError as e:
            return False, str(e)

    # Test cases
    test_forms = [
        {"email": "alice@example.com", "password": "secure123", "confirm_password": "secure123"},
        {"email": "bad-email", "password": "secure123", "confirm_password": "secure123"},
        {"email": "bob@test.com", "password": "short", "confirm_password": "short"},
        {"email": "eve@test.com", "password": "password1", "confirm_password": "password2"},
    ]

    for form in test_forms:
        success, message = process_registration(form)
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {form.get('email', 'N/A')}: {message}")


# === Exercise 4: Apply Boy Scout Rule ===
# Problem: Demonstrate making code slightly better.

def exercise_4():
    """Solution: Apply the Boy Scout Rule - leave code better than you found it."""

    # Before: A function with several small issues
    print("  BEFORE:")
    before_code = '''
    def calc(d, t):
        # calculate stuff
        r = d / t if t != 0 else 0
        unused_var = 42
        mph = r * 3600
        return mph
    '''
    print(before_code)

    # After: Four small improvements
    print("  AFTER (4 improvements):")
    after_code = '''
    def calculate_speed_mph(distance_miles, time_seconds):   # 1. Renamed function + params
        """Convert distance/time to miles per hour."""
        if time_seconds == 0:                                 # 2. Extracted guard clause
            return 0.0
        SECONDS_PER_HOUR = 3600                               # 3. Added explanatory constant
        speed_mph = (distance_miles / time_seconds) * SECONDS_PER_HOUR
        return speed_mph                                      # 4. Removed unused_var (dead code)
    '''
    print(after_code)

    # Actual working version
    def calculate_speed_mph(distance_miles, time_seconds):
        """Convert distance/time to miles per hour."""
        if time_seconds == 0:
            return 0.0
        SECONDS_PER_HOUR = 3600
        speed_mph = (distance_miles / time_seconds) * SECONDS_PER_HOUR
        return speed_mph

    print(f"  Test: 60 miles in 3600 seconds = {calculate_speed_mph(60, 3600)} mph")
    print(f"  Test: 0 miles in 0 seconds = {calculate_speed_mph(0, 0)} mph")

    print("\n  Changes documented:")
    print("    1. Renamed poorly-named variable (d -> distance_miles)")
    print("    2. Extracted long method (guard clause for division by zero)")
    print("    3. Removed dead code (unused_var)")
    print("    4. Added explanatory constant (3600 -> SECONDS_PER_HOUR)")


# === Exercise 5: Code Review Checklist ===
# Problem: Create a comprehensive code review checklist.

def exercise_5():
    """Solution: Code review checklist based on clean code principles."""

    checklist = {
        "Naming Checks": [
            "1. Are variable names descriptive and intention-revealing?",
            "2. Are function names verbs (do_something) and class names nouns?",
            "3. Are boolean variables named as predicates (is_active, has_permission)?",
            "4. Are abbreviations avoided (use 'customer' not 'cust')?",
            "5. Are names consistent (don't mix get/fetch/retrieve for same concept)?",
        ],
        "Function Checks": [
            "1. Does each function do ONE thing (Single Responsibility)?",
            "2. Are functions short (under 20 lines as a guideline)?",
            "3. Do functions have 3 or fewer parameters?",
            "4. Are there no side effects hidden in the function name?",
            "5. Do functions return early for error cases (guard clauses)?",
        ],
        "Code Smell Checks": [
            "1. No duplicate code (DRY principle)?",
            "2. No magic numbers (use named constants)?",
            "3. No deep nesting (max 2-3 levels)?",
            "4. No God classes (classes with too many responsibilities)?",
            "5. No feature envy (method uses another class's data excessively)?",
        ],
        "Comment Checks": [
            "1. Do comments explain WHY, not WHAT?",
            "2. Is there no commented-out code (use version control instead)?",
            "3. Are TODO comments tracked and have owners?",
        ],
    }

    for category, items in checklist.items():
        print(f"  {category}:")
        for item in items:
            print(f"    [ ] {item}")
        print()


if __name__ == "__main__":
    print("=== Exercise 1: Refactor This Code ===")
    exercise_1()
    print("\n=== Exercise 2: Identify Code Smells ===")
    exercise_2()
    print("\n=== Exercise 3: Write Clean Functions ===")
    exercise_3()
    print("\n=== Exercise 4: Apply Boy Scout Rule ===")
    exercise_4()
    print("\n=== Exercise 5: Code Review Checklist ===")
    exercise_5()
    print("\nAll exercises completed!")
