# HTML Forms and Tables

**Previous**: [HTML Basics](./01_HTML_Basics.md) | **Next**: [CSS Basics](./03_CSS_Basics.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the purpose of HTML forms and explain the difference between GET and POST methods
2. Implement text, date, selection, file upload, and button input elements in a form
3. Apply HTML5 built-in validation attributes such as required, pattern, minlength, and type-based checks
4. Write custom form validation logic using JavaScript
5. Construct tables with proper semantic structure using thead, tbody, tfoot, and caption
6. Apply colspan and rowspan to merge table cells
7. Implement accessible forms using labels, fieldset, legend, and ARIA attributes

---

Forms and tables are two of the most interaction-heavy HTML structures. Forms turn a static page into a two-way conversation -- they let users send data back to the server, whether it is a login credential, a search query, or a multi-step registration. Tables, meanwhile, remain the correct tool for displaying genuinely tabular data such as schedules, financial reports, and comparison matrices.

## Table of Contents

Before the reference, read [**Theory & Principles**](#theory--principles) — forms are the browser's built-in transport for user input (a serialization protocol with HTTP semantics baked in), and validation, accessibility, and tables each have their own data model that the markup encodes.

1. [Forms](#forms)
2. [Input Elements](#input-elements)
3. [Form Validation](#form-validation)
4. [Tables](#tables)
5. [Accessible Forms](#accessible-forms)
6. [Exercises](#exercises)

---

## Theory & Principles

A `<form>` is not a container for inputs — it is a small protocol. The browser provides a built-in pipeline that **collects** user input from descendant form-associated elements, **validates** it according to declarative rules, **serializes** it into a body, and **submits** it as an HTTP request. Every modern framework — React forms, HTMX, server actions in Next.js — sits on top of (or deliberately replaces) this pipeline. Understanding it is the difference between writing JavaScript that fights the browser and writing JavaScript that delegates to it.

### A. The Form Submission Pipeline

When a `submit` event fires (Enter in a single-line input, click on a `type="submit"` button, or `form.requestSubmit()`), the browser runs a fixed sequence:

1. **Constraint validation** runs across all *form-associated* descendants — `<input>`, `<select>`, `<textarea>`, `<button>`, and custom elements that opt in. Each element's `validity` state is computed from its attributes (`required`, `min`, `max`, `pattern`, `type`, `minlength`, `maxlength`, `step`).
2. **If anything is invalid,** the browser fires a non-bubbling `invalid` event on each invalid element and aborts. The default UI surfaces an error bubble on the first invalid element. Calling `event.preventDefault()` on `invalid` suppresses the bubble; calling `form.checkValidity()` runs the same algorithm without firing `submit`.
3. **If everything is valid,** the browser builds an **entry list** by walking the form tree in order and asking each successful control for its name/value pair. Disabled controls, unchecked checkboxes, and unselected radios contribute nothing.
4. **The entry list is encoded** according to the form's `enctype`:
   - `application/x-www-form-urlencoded` (default) → `name=value&name=value` in the URL or body.
   - `multipart/form-data` → MIME multipart, required for file uploads.
   - `text/plain` → debugging only.
5. **The request is sent** with the form's `method` (GET puts the body in the query string; POST puts it in the request body) to the form's `action` URL, replacing the current document with the response.

That last step is the one new developers most often forget exists: a plain `<form>` submission causes a full-page navigation. To stay on the page, you must call `event.preventDefault()` in a `submit` handler and send the request yourself with `fetch`.

### B. The Constraint Validation API

HTML5 specifies a typed validation model that runs entirely in the browser, before any JavaScript. Each form-associated element exposes:

- `element.validity` — a `ValidityState` object with boolean flags: `valueMissing`, `typeMismatch`, `patternMismatch`, `tooShort`, `tooLong`, `rangeUnderflow`, `rangeOverflow`, `stepMismatch`, `badInput`, `customError`, `valid`.
- `element.validationMessage` — the localized message the browser would show.
- `element.checkValidity()` / `form.checkValidity()` — runs the algorithm and fires `invalid` on failures.
- `element.setCustomValidity(message)` — sets the `customError` flag with your own message, marking the element invalid until you call `setCustomValidity('')`.

Two consequences:

1. **`required` and `pattern` are the same kind of thing as JavaScript validation.** They are not "fallback" validation — they participate in the same pipeline that your `customError` plugs into. There is no second, hidden validator.
2. **`type="email"`, `type="number"`, `type="url"` are validators, not just keyboards.** They participate in `typeMismatch`. A `type="number"` input refuses to give you a `value` until the user types something parseable; the value attribute is the *string they typed*, the `valueAsNumber` property is the *parsed number*.

The `:invalid`, `:valid`, `:required`, `:optional`, and `:user-invalid` pseudo-classes hook the same state into CSS, so styling can react to validation without any JavaScript at all.

### C. The Accessibility Contract: Labels, Names, and Groups

Every form control has an **accessible name** that assistive technology announces. The browser computes it by walking a fixed list of sources:

1. `aria-labelledby` (an ID reference list).
2. `aria-label` (a literal string).
3. The `<label>` element associated by `for`/`id` or by ancestry.
4. The control's `placeholder` (last-resort, and announced as such).
5. The `title` attribute.

`<label>` does two distinct things at once: it provides the accessible name *and* it widens the click/tap target — clicking the label focuses (and for checkboxes/radios, toggles) the associated control. This is why a missing `<label>` is both an accessibility bug and a usability bug; placeholder-only "labels" disappear the moment the user starts typing and cannot be recovered without clearing the field.

Grouping has its own contract. `<fieldset>` with `<legend>` produces a single accessible name for a *set* of related controls (a group of radios, a multi-part address). Without it, a screen reader announces seven inputs in a row with no idea they belong together.

### D. Tables Are a Data Model, Not a Layout Tool

A `<table>` declares two-dimensional tabular data. Its accessibility relies on the *semantic relationship* between header cells and data cells — built up from `<thead>`, `<tbody>`, `<tfoot>`, `<th>`, `<td>`, and `scope`/`headers` attributes. A screen reader navigating cell-by-cell announces the relevant header(s) for each cell so the user knows what the number means.

`<th scope="col">` says "I am a column header." `<th scope="row">` says "I am a row header." `<th headers="...">` lets a data cell point at the header IDs that apply to it for irregular tables. None of this is decoration; it is the entire reason `<table>` exists as a separate element instead of being a `<div>` grid. The corollary is the rule from the early 2000s: *do not use tables for layout*. Layout grids use CSS Grid; tables are reserved for data with row/column meaning.

### From Theory to the Reference Below

- **Forms** (section 1) covers `method`, `action`, and the `<form>` element — the entry point of the pipeline in §A.
- **Input Elements** (section 2) is a tour of the controls whose `name` and `value` populate the entry list.
- **Form Validation** (section 3) is §B: the declarative attributes and the JavaScript hooks into the same algorithm.
- **Tables** (section 4) implements §D — semantic structure for tabular data, never for layout.
- **Accessible Forms** (section 5) makes §C concrete: `<label>`, `<fieldset>`/`<legend>`, and ARIA only where native elements are not enough.

Read this lesson with the pipeline in mind: every attribute you add is configuring a stage of it.

---

## Forms

Forms are HTML elements that collect user input and send it to a server.

### Basic Form Structure

```html
<form action="/submit" method="POST">
    <!-- Form fields go here -->
    <button type="submit">Submit</button>
</form>
```

### Form Attributes

| Attribute | Description | Example Values |
|-----------|-------------|----------------|
| `action` | URL to send form data to | `/submit`, `https://api.example.com/form` |
| `method` | HTTP method | `GET`, `POST` |
| `enctype` | Encoding type | `application/x-www-form-urlencoded`, `multipart/form-data` |
| `target` | Where to display response | `_self`, `_blank` |
| `novalidate` | Disable browser validation | `novalidate` |

### GET vs POST

```html
<!-- GET: Data visible in URL (for search, etc.) -->
<form action="/search" method="GET">
    <input type="text" name="q">
    <button type="submit">Search</button>
</form>
<!-- Result: /search?q=keyword -->

<!-- POST: Data sent in request body (for login, registration, etc.) -->
<form action="/login" method="POST">
    <input type="text" name="username">
    <input type="password" name="password">
    <button type="submit">Login</button>
</form>
```

---

## Input Elements

### 1. Text Input

```html
<!-- Basic text input -->
<input type="text" name="username" placeholder="Enter username">

<!-- Password input -->
<input type="password" name="password" placeholder="Enter password">

<!-- Email input (with validation) -->
<input type="email" name="email" placeholder="email@example.com" required>

<!-- Number input -->
<input type="number" name="age" min="0" max="120" step="1">

<!-- Phone number input -->
<input type="tel" name="phone" pattern="[0-9]{3}-[0-9]{4}-[0-9]{4}">

<!-- URL input -->
<input type="url" name="website" placeholder="https://example.com">

<!-- Search input -->
<input type="search" name="search" placeholder="Search...">
```

### 2. Date and Time Input

```html
<!-- Date -->
<input type="date" name="birthday">

<!-- Time -->
<input type="time" name="appointment">

<!-- Date and time -->
<input type="datetime-local" name="meeting">

<!-- Month -->
<input type="month" name="month">

<!-- Week -->
<input type="week" name="week">
```

### 3. Selection Elements

```html
<!-- Radio button (single choice) -->
<fieldset>
    <legend>Select gender:</legend>
    <label>
        <input type="radio" name="gender" value="male" checked>
        Male
    </label>
    <label>
        <input type="radio" name="gender" value="female">
        Female
    </label>
    <label>
        <input type="radio" name="gender" value="other">
        Other
    </label>
</fieldset>

<!-- Checkbox (multiple choice) -->
<fieldset>
    <legend>Select hobbies:</legend>
    <label>
        <input type="checkbox" name="hobbies" value="reading">
        Reading
    </label>
    <label>
        <input type="checkbox" name="hobbies" value="sports">
        Sports
    </label>
    <label>
        <input type="checkbox" name="hobbies" value="music">
        Music
    </label>
</fieldset>

<!-- Dropdown -->
<label for="country">Country:</label>
<select id="country" name="country">
    <option value="">-- Select country --</option>
    <option value="us">United States</option>
    <option value="uk">United Kingdom</option>
    <option value="kr">South Korea</option>
</select>

<!-- Multiple selection dropdown -->
<select name="languages" multiple size="4">
    <option value="html">HTML</option>
    <option value="css">CSS</option>
    <option value="js">JavaScript</option>
    <option value="python">Python</option>
</select>
```

### 4. Textarea and File Upload

```html
<!-- Multi-line text input -->
<textarea name="message" rows="5" cols="30" placeholder="Enter message"></textarea>

<!-- File upload -->
<input type="file" name="profile-pic" accept="image/*">

<!-- Multiple file upload -->
<input type="file" name="documents" multiple accept=".pdf,.doc,.docx">
```

### 5. Other Input Types

```html
<!-- Color picker -->
<input type="color" name="color" value="#ff0000">

<!-- Range slider -->
<input type="range" name="volume" min="0" max="100" value="50">

<!-- Hidden field -->
<input type="hidden" name="user-id" value="12345">
```

### 6. Buttons

```html
<!-- Submit button -->
<button type="submit">Submit</button>
<input type="submit" value="Submit">

<!-- Reset button -->
<button type="reset">Reset</button>

<!-- Regular button -->
<button type="button" onclick="doSomething()">Click</button>
```

---

## Form Validation

### 1. HTML5 Validation Attributes

```html
<form>
    <!-- Required field -->
    <input type="text" name="username" required>

    <!-- Minimum/maximum length -->
    <input type="text" name="username" minlength="3" maxlength="20">

    <!-- Number range -->
    <input type="number" name="age" min="18" max="100">

    <!-- Pattern matching (regex) -->
    <input type="text" name="zipcode" pattern="[0-9]{5}">

    <!-- Email format validation -->
    <input type="email" name="email" required>

    <button type="submit">Submit</button>
</form>
```

### 2. Custom Error Messages (JavaScript)

```html
<form id="myForm">
    <input type="email" id="email" name="email" required>
    <button type="submit">Submit</button>
</form>

<script>
const emailInput = document.getElementById('email');

emailInput.addEventListener('invalid', (e) => {
    if (emailInput.validity.valueMissing) {
        emailInput.setCustomValidity('Please enter your email.');
    } else if (emailInput.validity.typeMismatch) {
        emailInput.setCustomValidity('Please enter a valid email format.');
    }
});

emailInput.addEventListener('input', () => {
    emailInput.setCustomValidity('');
});
</script>
```

### 3. Complete Form Validation Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Form Validation Example</title>
    <style>
        .error {
            color: red;
            font-size: 0.875rem;
        }
        input:invalid {
            border-color: red;
        }
        input:valid {
            border-color: green;
        }
    </style>
</head>
<body>
    <form id="registrationForm">
        <div>
            <label for="username">Username:</label>
            <input
                type="text"
                id="username"
                name="username"
                minlength="3"
                maxlength="20"
                required
            >
            <span class="error" id="usernameError"></span>
        </div>

        <div>
            <label for="email">Email:</label>
            <input
                type="email"
                id="email"
                name="email"
                required
            >
            <span class="error" id="emailError"></span>
        </div>

        <div>
            <label for="password">Password:</label>
            <input
                type="password"
                id="password"
                name="password"
                minlength="8"
                required
            >
            <span class="error" id="passwordError"></span>
        </div>

        <button type="submit">Register</button>
    </form>

    <script>
        const form = document.getElementById('registrationForm');

        form.addEventListener('submit', (e) => {
            e.preventDefault();

            // Clear previous error messages
            document.querySelectorAll('.error').forEach(el => el.textContent = '');

            // Validation
            let isValid = true;

            const username = document.getElementById('username');
            if (username.value.length < 3) {
                document.getElementById('usernameError').textContent =
                    'Username must be at least 3 characters.';
                isValid = false;
            }

            const email = document.getElementById('email');
            if (!email.value.includes('@')) {
                document.getElementById('emailError').textContent =
                    'Please enter a valid email.';
                isValid = false;
            }

            const password = document.getElementById('password');
            if (password.value.length < 8) {
                document.getElementById('passwordError').textContent =
                    'Password must be at least 8 characters.';
                isValid = false;
            }

            if (isValid) {
                alert('Registration successful!');
                form.reset();
            }
        });
    </script>
</body>
</html>
```

---

## Tables

Tables are used to display data in rows and columns.

### 1. Basic Table Structure

```html
<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Age</th>
            <th>City</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>John</td>
            <td>25</td>
            <td>New York</td>
        </tr>
        <tr>
            <td>Jane</td>
            <td>30</td>
            <td>Los Angeles</td>
        </tr>
    </tbody>
</table>
```

### 2. Table Tags

| Tag | Description |
|-----|-------------|
| `<table>` | Table root element |
| `<thead>` | Table header section |
| `<tbody>` | Table body section |
| `<tfoot>` | Table footer section |
| `<tr>` | Table row |
| `<th>` | Header cell |
| `<td>` | Data cell |
| `<caption>` | Table caption |

### 3. Complete Table Example

```html
<table border="1">
    <caption>2024 Sales Data</caption>
    <thead>
        <tr>
            <th>Quarter</th>
            <th>Revenue</th>
            <th>Expenses</th>
            <th>Profit</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Q1</td>
            <td>$100,000</td>
            <td>$70,000</td>
            <td>$30,000</td>
        </tr>
        <tr>
            <td>Q2</td>
            <td>$120,000</td>
            <td>$80,000</td>
            <td>$40,000</td>
        </tr>
    </tbody>
    <tfoot>
        <tr>
            <th>Total</th>
            <td>$220,000</td>
            <td>$150,000</td>
            <td>$70,000</td>
        </tr>
    </tfoot>
</table>
```

### 4. Cell Merging

```html
<!-- Column span (colspan) -->
<table border="1">
    <tr>
        <th colspan="3">Header spanning 3 columns</th>
    </tr>
    <tr>
        <td>Cell 1</td>
        <td>Cell 2</td>
        <td>Cell 3</td>
    </tr>
</table>

<!-- Row span (rowspan) -->
<table border="1">
    <tr>
        <th rowspan="2">Header spanning 2 rows</th>
        <td>Data 1</td>
    </tr>
    <tr>
        <td>Data 2</td>
    </tr>
</table>

<!-- Complex table -->
<table border="1">
    <tr>
        <th rowspan="2">Name</th>
        <th colspan="2">Scores</th>
    </tr>
    <tr>
        <th>Math</th>
        <th>English</th>
    </tr>
    <tr>
        <td>John</td>
        <td>90</td>
        <td>85</td>
    </tr>
    <tr>
        <td>Jane</td>
        <td>95</td>
        <td>92</td>
    </tr>
</table>
```

### 5. Styled Table (with CSS)

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Styled Table</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
    </style>
</head>
<body>
    <table>
        <thead>
            <tr>
                <th>Name</th>
                <th>Email</th>
                <th>Department</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>John Doe</td>
                <td>john@example.com</td>
                <td>IT</td>
            </tr>
            <tr>
                <td>Jane Smith</td>
                <td>jane@example.com</td>
                <td>HR</td>
            </tr>
            <tr>
                <td>Bob Johnson</td>
                <td>bob@example.com</td>
                <td>Sales</td>
            </tr>
        </tbody>
    </table>
</body>
</html>
```

---

## Accessible Forms

Accessible forms ensure all users can input information easily.

### 1. Label Usage

```html
<!-- Method 1: Wrap input with label -->
<label>
    Username:
    <input type="text" name="username">
</label>

<!-- Method 2: Connect with for attribute -->
<label for="email">Email:</label>
<input type="email" id="email" name="email">
```

### 2. Fieldset and Legend

```html
<form>
    <fieldset>
        <legend>Personal Information</legend>

        <label for="name">Name:</label>
        <input type="text" id="name" name="name"><br>

        <label for="email">Email:</label>
        <input type="email" id="email" name="email">
    </fieldset>

    <fieldset>
        <legend>Preferences</legend>

        <label>
            <input type="checkbox" name="newsletter">
            Subscribe to newsletter
        </label>
    </fieldset>
</form>
```

### 3. ARIA Attributes

```html
<form>
    <label for="username">Username:</label>
    <input
        type="text"
        id="username"
        name="username"
        aria-required="true"
        aria-describedby="username-help"
    >
    <span id="username-help">Username must be 3-20 characters.</span>

    <label for="email">Email:</label>
    <input
        type="email"
        id="email"
        name="email"
        aria-invalid="false"
    >
    <span role="alert" id="email-error"></span>
</form>
```

---

## Exercises

### Exercise 1: Create a Contact Form
Create a contact form with the following fields:
- Name (required)
- Email (required, email validation)
- Subject (dropdown)
- Message (textarea, required)
- Submit button

### Exercise 2: Create a Registration Form
Create a registration form with:
- Username (3-20 characters)
- Email (email validation)
- Password (at least 8 characters)
- Password confirmation (must match password)
- Gender (radio button)
- Agree to terms (checkbox, required)
- Submit button

**Sample Code:**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Registration Form</title>
</head>
<body>
    <form id="registrationForm">
        <fieldset>
            <legend>User Registration</legend>

            <label for="username">Username:</label>
            <input type="text" id="username" name="username"
                   minlength="3" maxlength="20" required><br><br>

            <label for="email">Email:</label>
            <input type="email" id="email" name="email" required><br><br>

            <label for="password">Password:</label>
            <input type="password" id="password" name="password"
                   minlength="8" required><br><br>

            <label for="confirmPassword">Confirm Password:</label>
            <input type="password" id="confirmPassword"
                   name="confirmPassword" required><br><br>

            <fieldset>
                <legend>Gender:</legend>
                <label>
                    <input type="radio" name="gender" value="male"> Male
                </label>
                <label>
                    <input type="radio" name="gender" value="female"> Female
                </label>
                <label>
                    <input type="radio" name="gender" value="other"> Other
                </label>
            </fieldset><br>

            <label>
                <input type="checkbox" name="terms" required>
                I agree to the terms and conditions
            </label><br><br>

            <button type="submit">Register</button>
        </fieldset>
    </form>
</body>
</html>
```

### Exercise 3: Create a Product Table
Create a product table with:
- Product name, price, quantity, total columns
- Table header and footer
- At least 5 products
- Footer row showing grand total
- Styling with CSS

---

## Summary

This lesson covered:
1. Form structure and attributes
2. Various input types and usage
3. Form validation (HTML5, JavaScript)
4. Table creation and cell merging
5. Accessible forms (labels, fieldset, ARIA)

---

**Previous**: [HTML Basics](./01_HTML_Basics.md) | **Next**: [CSS Basics](./03_CSS_Basics.md)
