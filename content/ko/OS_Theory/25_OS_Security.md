[이전: eBPF와 커널 트레이싱](./24_eBPF_Kernel_Tracing.md)

---

# 25. OS 보안

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Linux capabilities와 최소 권한 원칙을 설명할 수 있다
2. 샌드박싱을 위한 seccomp-bpf 시스콜 필터링을 구현할 수 있다
3. AppArmor와 SELinux 강제적 접근 제어를 구성할 수 있다
4. 여러 보안 레이어를 결합한 심층 방어 전략을 구축할 수 있다
5. 일반적인 OS 수준 공격과 그 완화 방법을 분석할 수 있다

---

## 목차

1. [보안 기초](#1-보안-기초)
2. [Linux Capabilities](#2-linux-capabilities)
3. [seccomp-bpf](#3-seccomp-bpf)
4. [강제적 접근 제어](#4-강제적-접근-제어)
5. [샌드박싱 기법](#5-샌드박싱-기법)
6. [커널 보안 기능](#6-커널-보안-기능)
7. [공격 표면 축소](#7-공격-표면-축소)
8. [연습문제](#8-연습문제)

---

## 1. 보안 기초

### 1.1 보안 삼각형

```
CIA 삼각형:
  기밀성 (Confidentiality): 인가된 접근만 허용
  무결성 (Integrity):       인가 없이 데이터 수정 불가
  가용성 (Availability):    필요할 때 시스템 사용 가능

심층 방어:
  레이어 1: 하드웨어 보안 (TPM, 보안 부팅)
  레이어 2: 커널 보안 (ASLR, capabilities, MAC)
  레이어 3: 프로세스 격리 (네임스페이스, seccomp)
  레이어 4: 애플리케이션 보안 (입력 검증, 암호화)
  레이어 5: 네트워크 보안 (방화벽, TLS)

  단일 레이어로는 충분하지 않음. 각 레이어가 공격 표면을 줄임.
```

### 1.2 권한 상승

```
권한 상승을 위한 공격 벡터:

1. SUID 바이너리: root로 실행되는 프로그램
   find / -perm -4000 -type f 2>/dev/null

2. 커널 취약점: root 권한을 위한 커널 버그 악용
   예: Dirty COW (CVE-2016-5195)

3. 잘못된 권한 설정: 모든 사용자 쓰기 가능 파일, 취약한 sudo 규칙

4. 컨테이너 탈출: 컨테이너 격리 벗어나기

5. 공급망: 손상된 라이브러리 또는 빌드 도구

완화: Capabilities, seccomp, MAC, 커널 하드닝
```

---

## 2. Linux Capabilities

### 2.1 Capabilities 개요

```c
#include <stdio.h>
#include <sys/capability.h>
#include <unistd.h>

/*
 * Linux Capabilities: root 권한을 세분화된 단위로 분리.
 *
 * 전통적: 프로세스가 root(모든 권한)이거나 아니거나.
 * Capabilities: 완전한 root 없이 특정 권한을 부여.
 *
 * 주요 capabilities:
 *   CAP_NET_BIND_SERVICE: 1024 미만 포트에 바인드
 *   CAP_NET_RAW:          원시 소켓 사용 (ping)
 *   CAP_SYS_ADMIN:        광범위한 시스템 관리
 *   CAP_DAC_OVERRIDE:     파일 권한 검사 우회
 *   CAP_SYS_PTRACE:       모든 프로세스 트레이싱
 *   CAP_NET_ADMIN:        네트워크 구성
 *   CAP_CHOWN:            파일 소유권 변경
 *
 * 세트:
 *   Effective:   현재 활성화된 capabilities
 *   Permitted:   허용된 최대 세트
 *   Inheritable: 자식 프로세스에 전달
 *   Bounding:    상한선 (축소만 가능)
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

/* 필요한 것만 남기고 모든 capabilities 제거 */
void drop_privileges(void) {
    cap_t caps = cap_init();  /* 빈 세트 */

    /* CAP_NET_BIND_SERVICE만 유지 */
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

### 3.1 시스템 호출 필터링

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
 * seccomp-bpf: BPF 규칙을 사용하여 시스템 호출 필터링.
 * 프로세스는 명시적으로 허용된 시스콜만 호출 가능.
 *
 * 액션:
 *   SECCOMP_RET_ALLOW:    시스콜 허용
 *   SECCOMP_RET_KILL:     프로세스 종료
 *   SECCOMP_RET_ERRNO:    실행 대신 오류 반환
 *   SECCOMP_RET_TRACE:    ptrace 트레이서에 알림
 *   SECCOMP_RET_LOG:      허용하되 로그
 */

void apply_seccomp_filter(void) {
    /* BPF 필터: read, write, exit, exit_group만 허용 */
    struct sock_filter filter[] = {
        /* 시스콜 번호 로드 */
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 offsetof(struct seccomp_data, nr)),

        /* read (0) 허용 */
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_read, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        /* write (1) 허용 */
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_write, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        /* exit_group (231) 허용 */
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_exit_group, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

        /* 기타 모든 시스콜에서 프로세스 종료 */
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL),
    };

    struct sock_fprog prog = {
        .len = sizeof(filter) / sizeof(filter[0]),
        .filter = filter,
    };

    /* 먼저 no_new_privs를 설정해야 함 */
    prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);

    /* seccomp 필터 적용 */
    if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog) != 0) {
        perror("prctl(SECCOMP)");
        exit(1);
    }

    printf("seccomp filter applied. Only read/write/exit allowed.\n");

    /* 이것은 작동함: */
    write(1, "Hello from sandbox!\n", 20);

    /* 이것은 프로세스를 종료시킴:
     * open("/etc/passwd", O_RDONLY);  // SIGKILL!
     */
}

int main(void) {
    apply_seccomp_filter();
    return 0;
}
```

---

## 4. 강제적 접근 제어

### 4.1 AppArmor

```
AppArmor: 경로 기반 강제적 접근 제어.

nginx 프로필:
  /etc/apparmor.d/usr.sbin.nginx

  #include <tunables/global>
  /usr/sbin/nginx {
    #include <abstractions/base>
    #include <abstractions/nameservice>

    # 네트워크 접근
    network inet stream,
    network inet6 stream,

    # 파일 접근
    /etc/nginx/** r,
    /var/log/nginx/** w,
    /var/www/** r,
    /run/nginx.pid rw,

    # 기본적으로 나머지 모두 거부!
    deny /etc/shadow r,
    deny /root/** rw,
  }
```

### 4.2 SELinux

```
SELinux: 레이블 기반 강제적 접근 제어.

모든 프로세스와 파일에 보안 컨텍스트가 있음:
  user:role:type:level

예:
  프로세스: system_u:system_r:httpd_t:s0
  파일:    system_u:object_r:httpd_sys_content_t:s0

정책 규칙:
  allow httpd_t httpd_sys_content_t : file { read open };
  → httpd_t 프로세스가 httpd_sys_content_t 파일을 읽을 수 있음

컨텍스트 확인:
  $ ls -Z /var/www/html/
  system_u:object_r:httpd_sys_content_t:s0 index.html

  $ ps -Z | grep httpd
  system_u:system_r:httpd_t:s0  12345 httpd
```

---

## 5. 샌드박싱 기법

### 5.1 포괄적 샌드박스

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <sys/resource.h>

/*
 * 여러 기법을 결합한 다중 레이어 샌드박스.
 */

void apply_sandbox(void) {
    /* 레이어 1: Capabilities 제거 */
    /* drop_privileges(); */

    /* 레이어 2: 리소스 제한 설정 */
    struct rlimit rl;

    /* 메모리를 256 MB로 제한 */
    rl.rlim_cur = rl.rlim_max = 256 * 1024 * 1024;
    setrlimit(RLIMIT_AS, &rl);

    /* 파일 크기를 10 MB로 제한 */
    rl.rlim_cur = rl.rlim_max = 10 * 1024 * 1024;
    setrlimit(RLIMIT_FSIZE, &rl);

    /* 프로세스 수를 10으로 제한 */
    rl.rlim_cur = rl.rlim_max = 10;
    setrlimit(RLIMIT_NPROC, &rl);

    /* 열린 파일을 20으로 제한 */
    rl.rlim_cur = rl.rlim_max = 20;
    setrlimit(RLIMIT_NOFILE, &rl);

    /* 레이어 3: 새 권한 없음 */
    prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);

    /* 레이어 4: 코어 덤프 방지 (민감한 데이터 유출 가능) */
    rl.rlim_cur = rl.rlim_max = 0;
    setrlimit(RLIMIT_CORE, &rl);

    /* 레이어 5: seccomp 필터 적용 */
    /* apply_seccomp_filter(); */

    printf("Sandbox applied: memory=256M, files=20, procs=10\n");
}

int main(void) {
    printf("Before sandbox:\n");
    apply_sandbox();
    printf("After sandbox: running in restricted environment\n");

    /* 애플리케이션 코드가 최소 권한으로 여기서 실행 */

    return 0;
}
```

---

## 6. 커널 보안 기능

### 6.1 ASLR, 스택 카나리 등

```
커널 하드닝 기능:

1. ASLR (Address Space Layout Randomization):
   ROP/버퍼 오버플로 익스플로잇을 방지하기 위해 메모리 레이아웃 무작위화.
   확인: cat /proc/sys/kernel/randomize_va_space
   값: 0=꺼짐, 1=부분, 2=전체

2. 스택 카나리:
   스택에 놓인 임의 값으로 덮어쓰기 감지.
   컴파일: gcc -fstack-protector-strong

3. NX/DEP (No-Execute):
   데이터 페이지(스택, 힙)에서의 코드 실행 방지.
   하드웨어 지원: AMD NX 비트, Intel XD 비트.

4. KASLR (Kernel ASLR):
   부팅 시 커널 베이스 주소를 무작위화.

5. KPTI (Kernel Page Table Isolation):
   커널/사용자 페이지 테이블 분리 (Meltdown 완화).

6. SMEP/SMAP:
   커널이 사용자 공간 메모리를 실행/접근하는 것을 방지.
```

---

## 7. 공격 표면 축소

### 7.1 하드닝 체크리스트

```
Linux 서버 하드닝:

커널:
  □ ASLR 활성화 (randomize_va_space = 2)
  □ 커널 모듈 로딩 비활성화 (필요 없는 경우)
  □ 감사 서브시스템 활성화
  □ dmesg 접근 제한 (dmesg_restrict = 1)
  □ kexec 비활성화 (필요 없는 경우)

프로세스:
  □ 서비스를 비root로 실행
  □ setuid 대신 capabilities 사용
  □ 모든 서비스에 seccomp 프로필 적용
  □ AppArmor/SELinux 프로필 활성화

파일시스템:
  □ /tmp를 noexec,nosuid로 마운트
  □ 적절한 파일 권한 설정
  □ 중요 구성에 불변 플래그
  □ auditd로 파일 접근 감사

네트워크:
  □ 최소 개방 포트
  □ 방화벽 (iptables/nftables)
  □ 불필요한 프로토콜 비활성화
  □ TCP SYN 쿠키 활성화
```

---

## 8. 연습문제

### 연습문제 1: Capability 탐구

Linux capabilities를 탐구하세요:
1. 일반적인 SUID 바이너리(ping, passwd, su)의 capabilities 나열
2. ping에서 SUID를 제거하고 대신 CAP_NET_RAW 추가
3. 하나를 제외한 모든 capabilities를 제거하는 프로그램 작성
4. 제한된 프로그램이 특권 작업을 수행할 수 없음을 보여주기
5. `getpcaps`와 `capsh`를 사용하여 capability 세트 확인

### 연습문제 2: seccomp 샌드박스

커스텀 seccomp 샌드박스 구축:
1. read, write, mmap, mprotect, exit만 허용하는 seccomp 필터 작성
2. 테스트: open() 호출을 시도하고 프로세스가 종료되는지 확인
3. SECCOMP_RET_KILL을 SECCOMP_RET_ERRNO로 변경하고 EPERM 반환
4. 간단한 HTTP 서버를 위한 seccomp 프로필 구축
5. seccomp 필터링의 성능 오버헤드 측정

### 연습문제 3: AppArmor 프로필

애플리케이션을 위한 AppArmor 프로필 생성:
1. aa-genprof를 사용하여 Python 스크립트의 프로필 생성
2. complain 모드에서 테스트: 필요한 권한 식별
3. enforce 모드로 전환하고 제한 확인
4. 거부된 경로에 접근 시도하고 차단 확인
5. 비교: 단순 vs 복잡한 애플리케이션의 프로필 복잡도

### 연습문제 4: 다중 레이어 샌드박스

포괄적인 샌드박스 구축:
1. 결합: 네임스페이스 + capabilities + seccomp + 리소스 제한
2. 샌드박스는: CPU, 메모리, 네트워크, 파일시스템 접근을 제한해야 함
3. 간단한 워크로드로 테스트하고 모든 제한이 작동하는지 확인
4. 샌드박스 탈출 시도 (호스트 리소스 접근 시도)
5. 문서화: 어떤 레이어가 어떤 유형의 공격을 방지하는가

### 연습문제 5: 익스플로잇 완화 분석

보안 완화를 분석하세요:
1. 간단한 버퍼 오버플로 취약점 작성 (제어된 환경에서)
2. 완화 없이 테스트: -fno-stack-protector, execstack, ASLR 없음
3. 완화를 하나씩 활성화하고 효과 관찰
4. 문서화: 어떤 완화가 익스플로잇의 어떤 단계를 방지하는가
5. 방어 매트릭스 생성: 공격 유형 vs 완화 효과

---

*레슨 25 끝*
