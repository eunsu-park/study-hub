#!/usr/bin/env bash
# =============================================================================
# 22_ansible_basics.sh — Inventory, Playbook, and Module Basics (read-only)
#
# PURPOSE: Demonstrates the Ansible control-flow in a self-contained way —
#          generates a small inventory and playbook in a tempdir, runs them
#          in CHECK MODE (no changes applied) if ansible is installed, and
#          explains the output. If ansible is not installed, the script
#          shows the commands and playbook content so the reader can follow
#          along conceptually.
#
# USAGE:
#   ./22_ansible_basics.sh [--concepts|--inventory|--playbook|--check|--all]
# =============================================================================

set -euo pipefail

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
cd "$WORKDIR"

section() { printf "\n=== %s ===\n\n" "$1"; }
explain() { printf "[INFO] %s\n" "$1"; }
show()    { printf "[CMD]  %s\n" "$1"; }

# ---------------------------------------------------------------------------
# 1. Core concepts
# ---------------------------------------------------------------------------
demo_concepts() {
    section "1. Ansible in One Screen"

    cat <<'EOF'
  Control node  — the machine you run ansible from (needs python + ansible).
  Managed node  — a target; only needs SSH + a python interpreter.
  Inventory     — a list of hosts, often with groups and per-host variables.
  Module        — a Python unit of work (e.g., apt, file, copy, service).
  Task          — a single module invocation with parameters.
  Play          — a list of tasks mapped to a group of hosts.
  Playbook      — a YAML file containing one or more plays.
  Idempotent    — re-running converges to the same state; only changed items apply.

  Ansible connects over SSH, copies the module as a tiny python script, runs it
  on the target, collects the result, and reports it back. There is no agent
  on the managed node.
EOF
}

# ---------------------------------------------------------------------------
# 2. Inventory file
# ---------------------------------------------------------------------------
demo_inventory() {
    section "2. Inventory File"

    cat > inventory.ini <<'EOF'
[webservers]
web1.example.com
web2.example.com

[dbservers]
db1.example.com ansible_port=2222

[production:children]
webservers
dbservers

[production:vars]
env=prod
ansible_user=deploy
EOF

    explain "Generated inventory.ini (groups, per-host port override, group vars):"
    show "cat inventory.ini"
    cat inventory.ini
}

# ---------------------------------------------------------------------------
# 3. Playbook
# ---------------------------------------------------------------------------
demo_playbook() {
    section "3. Playbook"

    cat > playbook.yml <<'EOF'
---
- name: Prepare web servers
  hosts: webservers
  become: true

  tasks:
    - name: Ensure nginx is installed
      ansible.builtin.package:
        name: nginx
        state: present

    - name: Deploy custom index page
      ansible.builtin.copy:
        content: "hello from {{ inventory_hostname }}\n"
        dest: /var/www/html/index.html
        owner: www-data
        group: www-data
        mode: '0644'

    - name: Enable and start nginx
      ansible.builtin.service:
        name: nginx
        state: started
        enabled: true
EOF

    explain "A playbook is YAML. Three tasks are idempotent: package install,"
    explain "file copy with owner/mode, and service start+enable."
    show "cat playbook.yml"
    cat playbook.yml

    explain "{{ inventory_hostname }} is a Jinja2 template that expands to the host name."
}

# ---------------------------------------------------------------------------
# 4. Check-mode dry run (safe)
# ---------------------------------------------------------------------------
demo_check() {
    section "4. Dry Run (check mode)"

    if ! command -v ansible-playbook >/dev/null 2>&1; then
        explain "ansible-playbook not installed — skipping actual run."
        explain "What you would run on a real control node:"
        show "ansible-playbook -i inventory.ini playbook.yml --check --diff"
        show "  # --check: predict changes, do not apply"
        show "  # --diff : show exact file-content diffs"
        return 0
    fi

    explain "Running ansible-playbook in --check mode against an invalid inventory is"
    explain "safe and demonstrates the output format without needing real SSH access."
    show "ansible-playbook -i inventory.ini playbook.yml --syntax-check"
    ansible-playbook -i inventory.ini playbook.yml --syntax-check 2>&1 || true
}

# ---------------------------------------------------------------------------
main() {
    local mode="${1:---all}"
    case "$mode" in
        --concepts)  demo_concepts ;;
        --inventory) demo_inventory ;;
        --playbook)  demo_playbook ;;
        --check)     demo_check ;;
        --all|*)
            demo_concepts
            demo_inventory
            demo_playbook
            demo_check
            ;;
    esac
}

main "$@"
