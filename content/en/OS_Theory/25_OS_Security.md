[Previous: eBPF and Kernel Tracing](./24_eBPF_Kernel_Tracing.md)

---

# 25. OS Security

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain Linux capabilities and the principle of least privilege
2. Implement seccomp-bpf syscall filtering for sandboxing
3. Configure AppArmor and SELinux mandatory access control
4. Build defense-in-depth strategies combining multiple security layers
5. Analyze common OS-level attacks and their mitigations

---

## Table of Contents

1. [Security Fundamentals](#1-security-fundamentals)
2. [Linux Capabilities](#2-linux-capabilities)
3. [seccomp-bpf](#3-seccomp-bpf)
4. [Mandatory Access Control](#4-mandatory-access-control)
5. [Sandboxing Techniques](#5-sandboxing-techniques)
6. [Kernel Security Features](#6-kernel-security-features)
7. [Attack Surface Reduction](#7-attack-surface-reduction)
8. [Exercises](#8-exercises)

---

## 1. Security Fundamentals

### 1.1 The Security Triad

```
CIA Triad:
  Confidentiality: Only authorized access to information
  Integrity:       Data is not modified without authorization
  Availability:    System is available when needed

Defense in Depth:
  Layer 1: Hardware security (TPM, secure boot)
  Layer 2: Kernel security (ASLR, capabilities, MAC)
  Layer 3: Process isolation (namespaces, seccomp)
  Layer 4: Application security (input validation, crypto)
  Layer 5: Network security (firewall, TLS)

  No single layer is enough. Each layer reduces attack surface.
```

### 1.2 Privilege Escalation

```
Attack vectors for privilege escalation:

1. SUID binaries: Programs that run as root
   find / -perm -4000 -type f 2>/dev/null

2. Kernel vulnerabilities: Exploit kernel bugs for root
   Example: Dirty COW (CVE-2016-5195)

3. Misconfigured permissions: World-writable files, weak sudo rules

4. Container escapes: Break out of container isolation

5. Supply chain: Compromised libraries or build tools

Mitigations: Capabilities, seccomp, MAC, kernel hardening
```

---

## 2. Linux Capabilities

### 2.1 Capabilities Overview

```c
#include <stdio.h>
#include <sys/capability.h>
#include <unistd.h>

/*
 * Linux Capabilities: Break root privilege into fine-grained units.
 *
 * Traditional: Process is either root (all privileges) or not.
 * Capabilities: Grant specific privileges without full root.
 *
 * Key capabilities:
 *   CAP_NET_BIND_SERVICE: Bind to ports < 1024
 *   CAP_NET_RAW:          Use raw sockets (ping)
 *   CAP_SYS_ADMIN:        Broad system administration
 *   CAP_DAC_OVERRIDE:     Bypass file permission checks
 *   CAP_SYS_PTRACE:       Trace any process
 *   CAP_NET_ADMIN:        Network configuration
 *   CAP_CHOWN:            Change file ownership
 *
 * Sets:
 *   Effective:   Currently active capabilities
 *   Permitted:   Maximum set allowed
 *   Inheritable: Passed to child processes
 *   Bounding:    Upper limit (can only shrink)
 */

void show_capabilities(void) {
    cap_t caps = cap_get_proc();
    if (caps == NULL) {
        perror("cap_get_proc");
        return;
    }

    char *text = cap_to_text(caps, NULL);
    printf("Current capabilities: %s\n", text ? text : "none");

    cap_free(text);
    cap_free(caps);
}

/* Drop all capabilities except what's needed */
void drop_privileges(void) {
    cap_t caps = cap_init();  /* Empty set */

    /* Only keep CAP_NET_BIND_SERVICE */
    cap_value_t keep[] = {CAP_NET_BIND_SERVICE};
    cap_set_flag(caps, CAP_PERMITTED, 1, keep, CAP_SET);
    cap_set_flag(caps, CAP_EFFECTIVE, 1, keep, CAP_SET);

    if (cap_set_proc(caps) != 0) {
        perror("cap_set_proc");
    }

    cap_free(caps);
    printf("Dropped to minimal capabilities\n");
    show_capabilities();
}
```

---

## 3. seccomp-bpf

### 3.1 System Call Filtering

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <linux/audit.h>
#include <sys/syscall.h>

/*
 * seccomp-bpf: Filter system calls using BPF rules.
 * Process can only make syscalls that are explicitly allowed.
 *
 * Actions:
 *   SECCOMP_RET_ALLOW:    Allow the syscall
 *   SECCOMP_RET_KILL:     Kill the process
 *   SECCOMP_RET_ERRNO:    Return error instead of executing
 *   SECCOMP_RET_TRACE:    Notify ptrace tracer
 *   SECCOMP_RET_LOG:      Allow but log
 */

void apply_seccomp_filter(void) {
    /* BPF filter: only allow read, write, exit, exit_group */
    struct sock_filter filter[] = {
        /* Load syscall number */
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 offsetof(struct seccomp_data, nr)),

        /* Allow read (0) */
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_read, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        /* Allow write (1) */
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_write, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        /* Allow exit_group (231) */
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_exit_group, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        /* Kill on any other syscall */
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL),
    };

    struct sock_fprog prog = {
        .len = sizeof(filter) / sizeof(filter[0]),
        .filter = filter,
    };

    /* Must set no_new_privs first */
    prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);

    /* Apply seccomp filter */
    if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog) != 0) {
        perror("prctl(SECCOMP)");
        exit(1);
    }

    printf("seccomp filter applied. Only read/write/exit allowed.\n");

    /* This will work: */
    write(1, "Hello from sandbox!\n", 20);

    /* This would kill the process:
     * open("/etc/passwd", O_RDONLY);  // SIGKILL!
     */
}

int main(void) {
    apply_seccomp_filter();
    return 0;
}
```

---

## 4. Mandatory Access Control

### 4.1 AppArmor

```
AppArmor: Path-based mandatory access control.

Profile for nginx:
  /etc/apparmor.d/usr.sbin.nginx

  #include <tunables/global>
  /usr/sbin/nginx {
    #include <abstractions/base>
    #include <abstractions/nameservice>

    # Network access
    network inet stream,
    network inet6 stream,

    # File access
    /etc/nginx/** r,
    /var/log/nginx/** w,
    /var/www/** r,
    /run/nginx.pid rw,

    # Deny everything else by default!
    deny /etc/shadow r,
    deny /root/** rw,
  }
```

### 4.2 SELinux

```
SELinux: Label-based mandatory access control.

Every process and file has a security context:
  user:role:type:level

Example:
  Process: system_u:system_r:httpd_t:s0
  File:    system_u:object_r:httpd_sys_content_t:s0

Policy rule:
  allow httpd_t httpd_sys_content_t : file { read open };
  → httpd_t processes can read httpd_sys_content_t files

Check context:
  $ ls -Z /var/www/html/
  system_u:object_r:httpd_sys_content_t:s0 index.html

  $ ps -Z | grep httpd
  system_u:system_r:httpd_t:s0  12345 httpd
```

---

## 5. Sandboxing Techniques

### 5.1 Comprehensive Sandbox

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <sys/resource.h>

/*
 * Multi-layer sandbox combining several techniques.
 */

void apply_sandbox(void) {
    /* Layer 1: Drop capabilities */
    /* drop_privileges(); */

    /* Layer 2: Set resource limits */
    struct rlimit rl;

    /* Limit memory to 256 MB */
    rl.rlim_cur = rl.rlim_max = 256 * 1024 * 1024;
    setrlimit(RLIMIT_AS, &rl);

    /* Limit file size to 10 MB */
    rl.rlim_cur = rl.rlim_max = 10 * 1024 * 1024;
    setrlimit(RLIMIT_FSIZE, &rl);

    /* Limit number of processes to 10 */
    rl.rlim_cur = rl.rlim_max = 10;
    setrlimit(RLIMIT_NPROC, &rl);

    /* Limit open files to 20 */
    rl.rlim_cur = rl.rlim_max = 20;
    setrlimit(RLIMIT_NOFILE, &rl);

    /* Layer 3: No new privileges */
    prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);

    /* Layer 4: Prevent core dumps (may leak sensitive data) */
    rl.rlim_cur = rl.rlim_max = 0;
    setrlimit(RLIMIT_CORE, &rl);

    /* Layer 5: Apply seccomp filter */
    /* apply_seccomp_filter(); */

    printf("Sandbox applied: memory=256M, files=20, procs=10\n");
}

int main(void) {
    printf("Before sandbox:\n");
    apply_sandbox();
    printf("After sandbox: running in restricted environment\n");

    /* Application code runs here with minimal privileges */

    return 0;
}
```

---

## 6. Kernel Security Features

### 6.1 ASLR, Stack Canaries, and More

```
Kernel hardening features:

1. ASLR (Address Space Layout Randomization):
   Randomize memory layout to prevent ROP/buffer overflow exploits.
   Check: cat /proc/sys/kernel/randomize_va_space
   Values: 0=off, 1=partial, 2=full

2. Stack Canaries:
   Random value on stack detected if overwritten.
   Compile: gcc -fstack-protector-strong

3. NX/DEP (No-Execute):
   Prevent code execution from data pages (stack, heap).
   Hardware support: AMD NX bit, Intel XD bit.

4. KASLR (Kernel ASLR):
   Randomize kernel base address at boot.

5. KPTI (Kernel Page Table Isolation):
   Separate kernel/user page tables (Meltdown mitigation).

6. SMEP/SMAP:
   Prevent kernel from executing/accessing userspace memory.
```

---

## 7. Attack Surface Reduction

### 7.1 Hardening Checklist

```
Linux Server Hardening:

Kernel:
  □ Enable ASLR (randomize_va_space = 2)
  □ Disable kernel module loading (if not needed)
  □ Enable audit subsystem
  □ Restrict dmesg access (dmesg_restrict = 1)
  □ Disable kexec (if not needed)

Process:
  □ Run services as non-root
  □ Use capabilities instead of setuid
  □ Apply seccomp profiles to all services
  □ Enable AppArmor/SELinux profiles

Filesystem:
  □ Mount /tmp with noexec,nosuid
  □ Set proper file permissions
  □ Immutable flag on critical configs
  □ Audit file access with auditd

Network:
  □ Minimal open ports
  □ Firewall (iptables/nftables)
  □ Disable unnecessary protocols
  □ Enable TCP SYN cookies
```

---

## 8. Exercises

### Exercise 1: Capability Exploration

Explore Linux capabilities:
1. List capabilities of common SUID binaries (ping, passwd, su)
2. Remove SUID from ping and add CAP_NET_RAW instead
3. Write a program that drops all capabilities except one
4. Show that the restricted program cannot perform privileged operations
5. Use `getpcaps` and `capsh` to verify capability sets

### Exercise 2: seccomp Sandbox

Build a custom seccomp sandbox:
1. Write a seccomp filter allowing only: read, write, mmap, mprotect, exit
2. Test: try to call open() and verify the process is killed
3. Change SECCOMP_RET_KILL to SECCOMP_RET_ERRNO and return EPERM
4. Build a seccomp profile for a simple HTTP server
5. Measure performance overhead of seccomp filtering

### Exercise 3: AppArmor Profile

Create AppArmor profiles for applications:
1. Generate profile for a Python script using aa-genprof
2. Test in complain mode: identify required permissions
3. Convert to enforce mode and verify restrictions
4. Attempt to access denied paths and verify blocking
5. Compare: profile complexity for simple vs complex applications

### Exercise 4: Multi-Layer Sandbox

Build a comprehensive sandbox:
1. Combine: namespaces + capabilities + seccomp + resource limits
2. The sandbox should: limit CPU, memory, network, filesystem access
3. Test with a simple workload and verify all limits work
4. Attempt sandbox escape (try to access host resources)
5. Document: which layers prevent which types of attacks

### Exercise 5: Exploit Mitigation Analysis

Analyze security mitigations:
1. Write a simple buffer overflow vulnerability (in a controlled environment)
2. Test without mitigations: -fno-stack-protector, execstack, no ASLR
3. Enable one mitigation at a time and observe the effect
4. Document: which mitigation prevents which step of exploitation
5. Create a defense matrix: attack type vs mitigation effectiveness

---

*End of Lesson 25*
