#!/bin/bash
# Exercises for Lesson 22: Ansible Basics
# Topic: Linux
# Solutions to practice problems from the lesson.

# === Exercise 1: Write Inventory ===
# Problem: Write YAML inventory with webservers (app1, app2), dbservers (db1),
#          shared user=deploy, webservers http_port=8080.
exercise_1() {
    echo "=== Exercise 1: Ansible Inventory in YAML ==="
    echo ""
    echo "Scenario: Define server groups with shared and group-specific variables."
    echo ""

    echo "Solution: inventory/hosts.yml"
    echo ""
    cat << 'INVENTORY'
# inventory/hosts.yml
# YAML inventory format (recommended over INI for complex setups)
all:
  vars:
    # Variables defined here apply to ALL hosts
    ansible_user: deploy       # SSH user for all connections

  children:
    # 'children' defines sub-groups of 'all'

    webservers:
      hosts:
        app1:                  # Hostname or IP (can add per-host vars below)
        app2:
      vars:
        http_port: 8080        # Group-specific variable, only for webservers
        # Can reference in templates: {{ http_port }}

    dbservers:
      hosts:
        db1:
INVENTORY
    echo ""

    echo "Variable precedence (lowest to highest):"
    echo "  1. all:vars          (applies to everything)"
    echo "  2. group:vars        (applies to group members)"
    echo "  3. host:vars         (applies to specific host)"
    echo "  4. Extra vars (-e)   (command-line override, highest priority)"
    echo ""
    echo "Why YAML over INI format?"
    echo "  - Better support for nested data structures"
    echo "  - Consistent with playbook format (both YAML)"
    echo "  - Clearer variable scoping with children/vars hierarchy"
    echo ""

    echo "Verify inventory:"
    echo "  ansible-inventory -i inventory/hosts.yml --list    # JSON output"
    echo "  ansible-inventory -i inventory/hosts.yml --graph   # Tree visualization"
    echo "  ansible all -i inventory/hosts.yml -m ping         # Test connectivity"
}

# === Exercise 2: Write Playbook ===
# Problem: Install and start nginx, support RedHat+Debian, use handlers and tags.
exercise_2() {
    echo "=== Exercise 2: Ansible Playbook for nginx ==="
    echo ""
    echo "Scenario: Write a cross-platform playbook with best practices (handlers, tags)."
    echo ""

    echo "Solution: playbooks/nginx.yml"
    echo ""
    cat << 'PLAYBOOK'
---
# playbooks/nginx.yml
- name: Install and configure nginx
  hosts: webservers
  become: yes               # Run tasks with sudo (privilege escalation)

  tasks:
    # --- Installation (tagged for selective execution) ---

    - name: Install nginx (RedHat family)
      yum:
        name: nginx
        state: present       # Ensure installed (idempotent)
      when: ansible_os_family == "RedHat"
      # 'when' uses Jinja2 expressions; ansible_os_family is auto-detected
      # Matches: RHEL, CentOS, Fedora, Rocky, AlmaLinux
      tags: install

    - name: Install nginx (Debian family)
      apt:
        name: nginx
        state: present
        update_cache: yes    # Run apt-get update before install
      when: ansible_os_family == "Debian"
      # Matches: Debian, Ubuntu, Mint
      tags: install

    # --- Configuration (tagged separately) ---

    - name: Copy nginx config
      template:
        src: nginx.conf.j2   # Jinja2 template (in templates/ dir)
        dest: /etc/nginx/nginx.conf
        owner: root
        group: root
        mode: '0644'
        validate: nginx -t -c %s   # Validate config before applying
      notify: Restart nginx
      # 'notify' triggers the handler ONLY if this task changes something
      # If config is identical, handler is NOT triggered (idempotent)
      tags: configure

    # --- Service Management ---

    - name: Start and enable nginx
      service:
        name: nginx
        state: started       # Ensure running
        enabled: yes         # Start on boot
      tags: install

  # --- Handlers (run once at end, only if notified) ---
  handlers:
    - name: Restart nginx
      service:
        name: nginx
        state: restarted
      # Handlers run AFTER all tasks complete, even if notified multiple times
      # This prevents unnecessary restarts during a play
PLAYBOOK
    echo ""

    echo "Key concepts demonstrated:"
    echo ""
    echo "  Handlers vs Tasks:"
    echo "    Tasks run every time (but are idempotent)"
    echo "    Handlers run only when notified by a changed task"
    echo "    Handlers run once at end, no matter how many notifications"
    echo ""
    echo "  Tags allow selective execution:"
    echo "    ansible-playbook nginx.yml --tags install     # Only install tasks"
    echo "    ansible-playbook nginx.yml --tags configure   # Only config tasks"
    echo "    ansible-playbook nginx.yml --skip-tags install"
    echo ""
    echo "  'when' conditions for cross-platform support:"
    echo "    ansible_os_family: RedHat, Debian, Suse, etc."
    echo "    ansible_distribution: Ubuntu, CentOS, Fedora, etc."
    echo "    ansible_distribution_major_version: 22, 9, etc."
    echo ""
    echo "Run the playbook:"
    echo "  ansible-playbook -i inventory/hosts.yml playbooks/nginx.yml"
    echo "  ansible-playbook -i inventory/hosts.yml playbooks/nginx.yml --check  # Dry run"
    echo "  ansible-playbook -i inventory/hosts.yml playbooks/nginx.yml -v       # Verbose"
}

# === Exercise 3: Write Role ===
# Problem: Write tasks/main.yml for a PostgreSQL installation role.
exercise_3() {
    echo "=== Exercise 3: Ansible Role for PostgreSQL ==="
    echo ""
    echo "Scenario: Create a reusable role that installs PostgreSQL, starts the service,"
    echo "and creates an initial database with a user."
    echo ""

    echo "Role directory structure:"
    echo "  roles/postgresql/"
    echo "    tasks/main.yml       # Main task list (this exercise)"
    echo "    defaults/main.yml    # Default variables (overridable)"
    echo "    handlers/main.yml    # Handlers (service restart, etc.)"
    echo "    templates/           # Jinja2 config templates"
    echo "    vars/main.yml        # Role-specific variables"
    echo ""

    echo "Solution: roles/postgresql/tasks/main.yml"
    echo ""
    cat << 'ROLE'
# roles/postgresql/tasks/main.yml
---
- name: Install PostgreSQL packages
  apt:
    name:
      - postgresql             # PostgreSQL server
      - postgresql-contrib     # Extra modules (pg_stat_statements, etc.)
      - python3-psycopg2       # Python adapter (required for Ansible postgresql_* modules)
    state: present
    update_cache: yes
  # Why python3-psycopg2? Ansible's postgresql_db and postgresql_user modules
  # use Python's psycopg2 library to connect to PostgreSQL.

- name: Start and enable PostgreSQL service
  service:
    name: postgresql
    state: started
    enabled: yes
  # PostgreSQL initializes its data directory automatically on first start

- name: Create application database
  become_user: postgres        # Switch to postgres system user
  # PostgreSQL uses "peer" auth by default: local connections authenticate
  # via the OS user. The 'postgres' user has superuser access.
  postgresql_db:
    name: "{{ pg_database | default('appdb') }}"
    state: present
  # '| default()' provides a fallback if variable is not defined
  # This makes the role flexible: callers can override via role vars

- name: Create database user
  become_user: postgres
  postgresql_user:
    name: "{{ pg_user | default('appuser') }}"
    password: "{{ pg_password }}"          # Should come from Ansible Vault!
    db: "{{ pg_database | default('appdb') }}"
    priv: ALL                              # Grant all privileges on the database
    state: present
  # SECURITY: Never hardcode passwords. Use ansible-vault:
  #   ansible-vault encrypt_string 'mysecretpassword' --name 'pg_password'
ROLE
    echo ""

    echo "Companion file: roles/postgresql/defaults/main.yml"
    echo ""
    cat << 'DEFAULTS'
# roles/postgresql/defaults/main.yml
---
pg_database: appdb
pg_user: appuser
# pg_password should be provided via Ansible Vault or extra vars
# Do NOT put default passwords in defaults/main.yml
DEFAULTS
    echo ""

    echo "Using the role in a playbook:"
    echo ""
    cat << 'USAGE'
# playbooks/database.yml
---
- name: Setup PostgreSQL database server
  hosts: dbservers
  become: yes
  roles:
    - role: postgresql
      vars:
        pg_database: myapp_production
        pg_user: myapp_user
        pg_password: "{{ vault_pg_password }}"  # From ansible-vault
USAGE
    echo ""

    echo "Role creation and management:"
    echo "  ansible-galaxy init roles/postgresql   # Create role skeleton"
    echo "  ansible-galaxy install geerlingguy.postgresql  # Install from Galaxy"
    echo ""
    echo "Why use roles?"
    echo "  - Reusable: Same role across multiple playbooks/projects"
    echo "  - Organized: Standard directory structure (tasks, handlers, templates, vars)"
    echo "  - Shareable: Publish to Ansible Galaxy for community use"
    echo "  - Testable: Can test roles independently with Molecule"
}

# Run all exercises
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
echo "All exercises completed!"
