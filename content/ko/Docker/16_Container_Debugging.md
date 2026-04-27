# 16. 컨테이너 디버깅(Container Debugging)

**이전**: [Podman과 OCI](./15_Podman_and_OCI.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `docker exec`를 사용하여 실행 중인 컨테이너 내부에서 대화형 디버깅을 수행한다
2. `docker logs`로 컨테이너 로그를 분석하고 프로덕션용 로그 드라이버를 구성한다
3. `docker inspect`를 사용하여 컨테이너 메타데이터와 설정을 추출한다
4. 네트워크 검사 및 진단 도구로 컨테이너 네트워킹 문제를 디버깅한다
5. `nsenter`를 사용하여 호스트에서 컨테이너 네임스페이스를 탐색한다
6. 컨테이너에서 시스템 콜 및 라이브러리 콜 추적을 위해 `strace`와 `ltrace`를 적용한다
7. 자가 복구(Self-healing) 컨테이너를 위한 헬스 체크와 재시작 정책을 구성한다
8. 멀티 컨테이너 애플리케이션을 디버깅하고 일반적인 컨테이너 문제를 해결한다

## 목차


1. [docker exec를 사용한 대화형 디버깅](#1-docker-exec를-사용한-대화형-디버깅)
2. [컨테이너 로그와 로그 드라이버](#2-컨테이너-로그와-로그-드라이버)
3. [docker inspect로 메타데이터 확인](#3-docker-inspect로-메타데이터-확인)
4. [네트워킹 문제 디버깅](#4-네트워킹-문제-디버깅)
5. [nsenter를 사용한 네임스페이스 탐색](#5-nsenter를-사용한-네임스페이스-탐색)
6. [시스템 콜 추적](#6-시스템-콜-추적)
7. [헬스 체크와 재시작 정책](#7-헬스-체크와-재시작-정책)
8. [멀티 컨테이너 애플리케이션 디버깅](#8-멀티-컨테이너-애플리케이션-디버깅)
9. [일반적인 문제와 해결 방법](#9-일반적인-문제와-해결-방법)
10. [연습 문제](#10-연습-문제)

**난이도**: ⭐⭐⭐⭐

---

컨테이너가 오작동할 때 전통적인 디버깅 접근 방식은 종종 부족합니다. 컨테이너에 SSH로 접속할 수 없고, 많은 이미지에 디버깅 도구가 없으며, 컨테이너의 일시적 특성으로 인해 증거가 사라질 수 있습니다. 이 레슨은 간단한 로그 검사부터 고급 네임스페이스 탐색, 시스템 콜 추적까지 포괄적인 컨테이너 디버깅 도구 키트를 다룹니다. 이러한 기술을 마스터하는 것은 프로덕션에서 컨테이너를 운영하는 모든 사람에게 필수적입니다.

---

## 1. docker exec를 사용한 대화형 디버깅

### 이론: `docker exec`와 `setns()` 시스템 콜

컨테이너 디버깅은 항상 해 온 같은 리눅스 프로세스 디버깅 — `strace`, `lsof`, `tcpdump`, `/proc` 검사 — 을 적절한 네임스페이스 *안에서* 수행하는 것입니다. 새로움은 네임스페이스 배관입니다.

`docker exec -it <container> sh`은 새 컨테이너를 fork하지 *않습니다*. 기존 컨테이너의 네임스페이스에 진입합니다. 메커니즘은 커널의 **`setns()`** 시스템 콜 — `/proc/<pid>/ns/<type>`을 가리키는 파일 디스크립터가 주어지면 `setns(fd, 0)`이 호출 프로세스를 그 네임스페이스로 이동.

각 실행 중인 프로세스가 `/proc/<pid>/ns/` 아래에 자기 네임스페이스 노출 —

```
$ ls -l /proc/1234/ns/
cgroup -> cgroup:[4026531835]
ipc    -> ipc:[4026531839]
mnt    -> mnt:[4026531840]
net    -> net:[4026532008]
pid    -> pid:[4026532009]
user   -> user:[4026531837]
uts    -> uts:[4026531838]
```

대괄호 안 숫자가 네임스페이스 inode. 두 프로세스가 같은 네임스페이스에 있으려면 같은 inode 가져야 함. "이 두 프로세스가 정말 같은 네트워크 네임스페이스에 있나?"를 확인하는 방법 — 둘에 대해 `readlink /proc/<pid>/ns/net`하고 비교.

`docker exec`(과 `kubectl exec`)이 동작하는 방식 —

1. 컨테이너의 PID 1 조회.
2. 각 `/proc/<pid1>/ns/<type>` 파일 디스크립터 열기.
3. 각각에 대해 `setns(fd, 0)` 호출.
4. 이제 컨테이너 네임스페이스 세계에서 요청된 명령 `execve()`.

`nsenter`는 같은 작업의 독립 CLI 버전 — `nsenter -t <pid> -p -m -u -n -i <command>`. 일부 네임스페이스만 진입하고 싶을 때(예: 네트워크 네임스페이스만 `-n` — 호스트의 `tcpdump`로 컨테이너 트래픽 스니핑에 유용).

### 이론: 임시 디버그 컨테이너 — 현대 워크플로

Distroless와 scratch 이미지는 프로덕션에 좋고 디버깅에 끔찍. 현대 Kubernetes(1.23+)가 **임시 컨테이너**(`kubectl debug`)로 해결 —

```bash
kubectl debug -it mypod --image=busybox --target=mycontainer -- sh
```

일어나는 일 —

1. kubelet이 containerd에 *기존* Pod에 *새* 컨테이너를 만들도록 요청, Pod의 네트워크와 PID 네임스페이스를, 그리고 `--target`을 통해 타깃 컨테이너의 프로세스 네임스페이스를 공유.
2. 새 컨테이너는 자체 파일시스템(busybox)을 갖지만 타깃 컨테이너의 프로세스를 보고 시그널 보낼 수 있음.
3. 타깃 이미지에 그것들이 없어도 `ps`, `cat`, `curl` 등이 있는 셸을 얻음.

Docker는 `docker run --network=container:other --pid=container:other --volumes-from=other busybox sh`으로 동등물 — 디버그 컨테이너의 네임스페이스를 기존 것에 수동 부착. K8s `kubectl debug`이 Pod 인식이 있는 같은 아이디어.

이 패턴이 "프로덕션 이미지를 최소한으로 강화"를 실용적으로 만듬. Distroless나 scratch가 *프로덕션* 이미지, *디버그* 이미지는 필요한 도구가 있는 busybox나 alpine, 필요 시 부착.

### 기본 사용법

`docker exec`는 실행 중인 컨테이너 안에서 명령어를 실행합니다:

```bash
# 실행 중인 컨테이너에서 셸 시작
docker exec -it myapp /bin/bash

# bash가 없는 경우 (최소 이미지)
docker exec -it myapp /bin/sh

# 특정 명령어 실행
docker exec myapp cat /etc/hosts

# 특정 사용자로 실행
docker exec -u root myapp whoami

# exec 세션에 환경 변수 설정
docker exec -e DEBUG=1 myapp python debug_script.py

# 작업 디렉토리 설정
docker exec -w /app/logs myapp ls -la
```

### 실행 중인 애플리케이션 디버깅

```bash
# 실행 중인 프로세스 확인
docker exec myapp ps aux

# 내부에서 리소스 사용량 확인
docker exec myapp top -bn1

# 파일시스템 검사
docker exec myapp ls -la /app/
docker exec myapp df -h
docker exec myapp du -sh /app/*

# 내부에서 네트워크 연결 확인
docker exec myapp ping -c 3 database
docker exec myapp curl -s http://localhost:8080/health

# 환경 변수 확인
docker exec myapp env

# 설정 파일 읽기
docker exec myapp cat /app/config.yaml
```

### 셸 없는 컨테이너 디버깅

일부 최소 이미지(scratch, distroless)에는 셸이 없습니다. 디버그 컨테이너를 사용하세요:

```bash
# 방법 1: Docker debug (Docker Desktop 4.27+)
docker debug myapp

# 방법 2: 정적 바이너리를 컨테이너에 복사
docker cp /usr/bin/busybox myapp:/busybox
docker exec myapp /busybox sh

# 방법 3: 호스트에서 nsenter 사용 (섹션 5 참조)

# 방법 4: 임시 디버그 컨테이너 사용 (Kubernetes)
kubectl debug -it myapp --image=busybox --target=myapp
```

### Attach vs Exec

```bash
# docker attach: 컨테이너의 메인 프로세스(PID 1)에 연결
# 주의: Ctrl+C가 컨테이너를 종료할 수 있음
docker attach myapp

# docker exec: 컨테이너 안에서 새 프로세스를 시작
# 안전: exec 세션을 종료해도 컨테이너에 영향 없음
docker exec -it myapp sh

# attach는 대화형 애플리케이션에 사용 (예: REPL)
# exec는 디버깅에 사용 (더 안전, 독립적인 프로세스)
```

---

## 2. 컨테이너 로그와 로그 드라이버

### 이론: 로깅 — stdout/stderr가 실제로 가는 곳

컨테이너 프로세스가 파일 디스크립터 1(stdout)과 2(stderr)에 씁니다. 그 FD들을 프로세스 자신이 아니라 **컨테이너 모니터**(`containerd-shim`, Podman/CRI-O의 경우 `conmon`)가 소유. 모니터가 그것들을 읽고 **로그 드라이버**를 통해 라우팅 —

| 로그 드라이버 | 로그가 가는 곳 |
|---------------|----------------|
| `json-file`(Docker 기본) | `/var/lib/docker/containers/<id>/<id>-json.log` — 디스크의 JSON 라인 |
| `journald` | systemd-journald(`journalctl`로 쿼리 가능) |
| `syslog` | 로컬 syslog 데몬 |
| `fluentd` / `gelf` / `awslogs` / `gcplogs` | 원격 집계기로 스트리밍 |
| `none` | 폐기 |

`docker logs <container>`이 json-file을 읽거나(또는 journald 쿼리, 또는 어느 드라이버든) 출력. `kubectl logs`이 kubelet을 통해 동등물.

함의 —

- **`docker logs`는 드라이버가 지원할 때만 동작.** json-file과 journald는 지원, 원격 드라이버는 안 함.
- **로깅 볼륨이 가득 차면 컨테이너가 멈춤.** 회전 없는 json-file이 고전적 장애. `--log-opt max-size=10m --log-opt max-file=3` 설정(또는 자기 관리하는 journald 사용).
- **다중 줄 스택 트레이스가 기본적으로 한 줄당 한 로그 항목.** 한 이벤트에 속하는 연속 줄을 머지할 줄 아는 로그 시퍼 사용, 또는 앱이 단일 줄 JSON 로그 내보내게.

### 기본 로그 접근

```bash
# 모든 로그 보기
docker logs myapp

# 로그 팔로우 (tail -f와 유사)
docker logs -f myapp

# 마지막 N줄 표시
docker logs --tail 100 myapp

# 특정 타임스탬프 이후 로그 표시
docker logs --since 2025-01-15T10:00:00 myapp

# 최근 30분간 로그 표시
docker logs --since 30m myapp

# 타임스탬프 표시
docker logs -t myapp

# 옵션 결합
docker logs -f --tail 50 -t myapp
```

### 로그 드라이버(Log Drivers)

Docker는 프로덕션 로그 관리를 위해 여러 로그 드라이버를 지원합니다:

```
┌──────────────────────────────────────────────────────────────┐
│                   Docker 로그 드라이버                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  json-file (기본)                                             │
│  ├─ /var/lib/docker/containers/에 JSON으로 로그 저장          │
│  ├─ docker logs 명령어 지원                                   │
│  └─ 최대 크기 및 로테이션 설정 가능                            │
│                                                               │
│  journald                                                     │
│  ├─ systemd 저널에 로그 기록                                  │
│  ├─ docker logs 명령어 지원                                   │
│  └─ 풍부한 메타데이터와 필터링                                │
│                                                               │
│  syslog                                                       │
│  ├─ syslog 데몬에 로그 기록                                   │
│  ├─ docker logs 미지원                                        │
│  └─ 표준 Unix 로깅                                            │
│                                                               │
│  fluentd                                                      │
│  ├─ Fluentd/Fluent Bit 수집기에 로그 기록                    │
│  ├─ docker logs 미지원                                        │
│  └─ 중앙 집중식 로깅 파이프라인에 최적                        │
│                                                               │
│  awslogs / gcplogs / splunk                                   │
│  ├─ 클라우드 네이티브 로그 전송                               │
│  └─ 클라우드 플랫폼과 직접 통합                               │
│                                                               │
│  none                                                         │
│  └─ 로깅 없음 (성능에 민감한 컨테이너에 사용)                 │
└──────────────────────────────────────────────────────────────┘
```

### 로그 드라이버 설정

```bash
# 컨테이너별 로그 드라이버 설정
docker run -d --name myapp \
  --log-driver json-file \
  --log-opt max-size=10m \
  --log-opt max-file=5 \
  myapp:latest

# daemon.json에서 기본 로그 드라이버 설정
```

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3",
    "labels": "production_status",
    "env": "os,customer"
  }
}
```

```yaml
# docker-compose.yml 로그 설정
version: "3.9"
services:
  app:
    image: myapp
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "5"
```

### 구조화된 로깅 모범 사례

```bash
# 애플리케이션은 stdout/stderr로 로그를 출력해야 함 (12-factor 앱)
# Docker가 stdout/stderr를 캡처하여 로그 드라이버로 라우팅

# 로그가 저장된 위치 확인
docker inspect --format='{{.LogPath}}' myapp

# 로그 파일 크기 확인
ls -lh $(docker inspect --format='{{.LogPath}}' myapp)
```

---

## 3. docker inspect로 메타데이터 확인

### 이론: `/proc/<pid>/`을 컨테이너 검사기로

모든 리눅스 프로세스가 파일을 통해 자기 상태를 노출하는 `/proc/<pid>/` 아래 디렉터리를 가집니다. *호스트*에서(실제 PID를 봄) 이것들이 컨테이너 프로세스에 진입하지 않고도 거의 모든 것을 알려 줍니다.

| 파일 | 알려주는 것 |
|------|-------------|
| `/proc/<pid>/status` | UID, GID, capability, 네임스페이스 inode, 부모 PID |
| `/proc/<pid>/cmdline` | exec된 argv |
| `/proc/<pid>/environ` | 환경 변수(권한 있을 때만) |
| `/proc/<pid>/cgroup` | 프로세스가 속한 cgroup(따라서 자원 제한) |
| `/proc/<pid>/maps` | 메모리 맵 — 모든 로드된 라이브러리, 모든 힙 세그먼트 |
| `/proc/<pid>/fd/` | 프로세스가 연 파일 디스크립터(소켓, 파일, 파이프) |
| `/proc/<pid>/root/` | 컨테이너 루트 파일시스템에 대한 마법 심링크(호스트에서 `cat /proc/<pid>/root/etc/passwd` 가능) |
| `/proc/<pid>/cwd` | 현재 작업 디렉터리 |
| `/proc/<pid>/net/tcp` | TCP 연결(프로세스 네트워크 네임스페이스) |
| `/proc/<pid>/limits` | RLIMIT 설정 |

가장 강력한 것은 `/proc/<pid>/root/`. 호스트에서, 컨테이너 안에 셸 필요 없이 컨테이너 파일시스템의 어떤 파일이든 읽을 수 있음. 셸이 없는 distroless 컨테이너에 이게 주요 검사 메커니즘 — `ls /proc/<pid>/root/app/`, `cat /proc/<pid>/root/etc/config`.

`/proc/<pid>/net/tcp`와 `/proc/<pid>/net/udp`가 호스트에서 읽지만 *컨테이너의* 네트워크 네임스페이스에서 연결 상태 줌.

### 컨테이너 검사

```bash
# 전체 JSON 출력
docker inspect myapp

# Go 템플릿을 사용한 특정 필드
docker inspect --format='{{.State.Status}}' myapp
docker inspect --format='{{.State.StartedAt}}' myapp
docker inspect --format='{{.State.Pid}}' myapp

# 네트워크 정보
docker inspect --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' myapp

# 포트 매핑
docker inspect --format='{{json .NetworkSettings.Ports}}' myapp | jq

# 볼륨 마운트
docker inspect --format='{{json .Mounts}}' myapp | jq

# 환경 변수
docker inspect --format='{{json .Config.Env}}' myapp | jq

# 재시작 횟수
docker inspect --format='{{.RestartCount}}' myapp

# OOM 종료 상태
docker inspect --format='{{.State.OOMKilled}}' myapp
```

### 유용한 검사 패턴

```bash
# 컨테이너가 오류로 종료했는지 확인
docker inspect --format='{{.State.ExitCode}}' myapp

# 컨테이너의 메인 명령어 가져오기
docker inspect --format='{{json .Config.Cmd}}' myapp

# 리소스 제한 확인
docker inspect --format='{{.HostConfig.Memory}}' myapp
docker inspect --format='{{.HostConfig.NanoCpus}}' myapp

# 컨테이너의 로그 파일 찾기
docker inspect --format='{{.LogPath}}' myapp

# 사용된 이미지 가져오기
docker inspect --format='{{.Config.Image}}' myapp

# 헬스 상태 확인
docker inspect --format='{{json .State.Health}}' myapp | jq
```

### 이미지 검사

```bash
# 이미지 검사
docker inspect nginx:alpine

# 이미지 레이어 가져오기
docker inspect --format='{{json .RootFS.Layers}}' nginx:alpine | jq

# 이미지 크기 가져오기
docker inspect --format='{{.Size}}' nginx:alpine

# 이미지 히스토리 보기 (빌드 방법)
docker history nginx:alpine
docker history --no-trunc nginx:alpine
```

### 컨테이너 비교

```bash
# 두 컨테이너 설정 비교
diff <(docker inspect container1) <(docker inspect container2)

# 특정 필드 비교
diff \
  <(docker inspect --format='{{json .Config.Env}}' prod | jq -S) \
  <(docker inspect --format='{{json .Config.Env}}' staging | jq -S)
```

---

## 4. 네트워킹 문제 디버깅

### 이론: 네트워크 디버깅 — 네임스페이스 안과 밖

네트워킹 문제는 가장 흔하고 가장 혼란스러운 컨테이너 디버깅 작업. 분할 —

- **호스트에 존재하고 `nsenter`를 통해 컨테이너 시각으로 동작하는 도구** — `tcpdump`, `iptables`, `ip route`, `ss`, `netstat`. 컨테이너에 없어도 호스트는 가짐.
- **컨테이너 안에서 실행해야 하는 도구** — 애플리케이션이 닿을 수 있는 것을 테스트하는 `curl`, 앱이 보는 방식으로 DNS를 테스트하는 `dig`, 같은 것의 `nslookup`.

컨테이너에 tcpdump 설치하지 않고 컨테이너 트래픽 캡처하는 표준 주문 —

```bash
# 컨테이너 PID 얻기
PID=$(docker inspect -f '{{.State.Pid}}' mycontainer)
# 호스트에서 컨테이너의 네트워크 네임스페이스에서 tcpdump 실행
sudo nsenter -t $PID -n tcpdump -i any -w /tmp/cap.pcap
```

DNS 문제 시 세 곳 확인 —

1. *컨테이너 안의* `/etc/resolv.conf` — 리졸버가 어떤 nameserver를 쿼리할지.
2. (사용자 정의 브리지의) `127.0.0.11`의 Docker DNS 서버 또는 K8s의 CoreDNS — 이름을 아는가?
3. 상류 DNS — 호스트가 이름을 리졸브 가능한가?

"컨테이너가 외부 세계에 못 닿음" — iptables MASQUERADE와 라우트 테이블 확인. "호스트가 컨테이너에 못 닿음" — iptables DNAT와 브리지의 `forwarding` 설정 확인.

### 네트워크 검사

```bash
# 모든 네트워크 나열
docker network ls

# 네트워크 검사
docker network inspect bridge

# 네트워크에 있는 컨테이너 찾기
docker network inspect mynet --format='{{range .Containers}}{{.Name}} {{.IPv4Address}}{{end}}'

# 컨테이너의 DNS 설정 확인
docker exec myapp cat /etc/resolv.conf

# 컨테이너의 호스트 항목 확인
docker exec myapp cat /etc/hosts
```

### 일반적인 네트워크 디버깅

```bash
# 컨테이너 간 연결 테스트
docker exec app1 ping -c 3 app2

# DNS 해석 테스트
docker exec myapp nslookup database
docker exec myapp getent hosts database

# 포트 연결 테스트
docker exec myapp nc -zv database 5432

# 컨테이너 내부에서 리스닝 포트 확인
docker exec myapp ss -tlnp
docker exec myapp netstat -tlnp

# HTTP 엔드포인트 테스트
docker exec myapp curl -v http://api:8080/health

# 네트워크 경로 추적
docker exec myapp traceroute database
```

### 네트워크 디버깅 컨테이너

애플리케이션 컨테이너에 네트워크 도구가 없는 경우, 같은 네트워크에 전용 디버그 컨테이너를 사용하세요:

```bash
# 같은 네트워크에서 디버그 컨테이너 실행
docker run --rm -it --network myapp_default \
  nicolaka/netshoot \
  bash

# netshoot 안에서 사용 가능: curl, dig, nslookup, tcpdump,
# iperf, nmap, netstat, ss, ip 등

# 패킷 캡처
docker run --rm --net container:myapp \
  nicolaka/netshoot \
  tcpdump -i eth0 -w /tmp/capture.pcap
```

### 포트 문제 디버깅

```bash
# 게시된 포트 확인
docker port myapp

# 호스트에서 포트 사용 여부 확인
ss -tlnp | grep 8080

# 포트 매핑 확인
docker inspect --format='{{json .NetworkSettings.Ports}}' myapp | jq

# 일반적인 문제: 컨테이너가 0.0.0.0이 아닌 127.0.0.1에서 리슨
docker exec myapp ss -tlnp
# 앱이 컨테이너 내부에서 127.0.0.1에 바인딩되면 외부 접근 실패
# 수정: 앱이 0.0.0.0에 바인딩하도록 설정
```

```
┌──────────────────────────────────────────────────────────────┐
│           네트워크 디버깅 의사결정 트리                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  호스트에서 컨테이너에 접근 불가?                               │
│  ├─ 확인: docker port <container>                             │
│  ├─ 확인: ss -tlnp | grep <port> (호스트)                     │
│  └─ 확인: 앱이 0.0.0.0에 바인딩 (127.0.0.1이 아닌)           │
│                                                               │
│  다른 컨테이너에서 컨테이너에 접근 불가?                        │
│  ├─ 확인: 같은 네트워크에 있는지?                              │
│  ├─ 확인: DNS 해석이 작동하는지?                               │
│  ├─ 확인: 컨테이너 간 ping                                    │
│  └─ 확인: 대상 포트가 리스닝 중인지                            │
│                                                               │
│  간헐적 연결?                                                  │
│  ├─ 확인: 리소스 제한 (OOM?)                                  │
│  ├─ 확인: 헬스 체크 실패                                      │
│  ├─ 확인: DNS 캐싱 문제                                       │
│  └─ 확인: 컨테이너 재시작 (docker events)                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. nsenter를 사용한 네임스페이스 탐색

### 컨테이너 네임스페이스 이해

컨테이너는 격리를 위해 Linux 네임스페이스를 사용합니다. `nsenter`를 사용하면 호스트에서 컨테이너의 네임스페이스에 들어갈 수 있습니다:

```
┌──────────────────────────────────────────────────────────────┐
│                  Linux 네임스페이스                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  PID  ─── 프로세스 격리 (컨테이너가 자체 PID를 볼 수 있음)     │
│  NET  ─── 네트워크 격리 (자체 인터페이스, IP, 라우트)          │
│  MNT  ─── 마운트 격리 (자체 파일시스템 뷰)                    │
│  UTS  ─── 호스트이름 격리                                      │
│  IPC  ─── 프로세스 간 통신 격리                                │
│  USER ─── 사용자/그룹 ID 격리 (루트리스 컨테이너)              │
│  CGROUP ─ Cgroup 뷰 격리                                      │
└──────────────────────────────────────────────────────────────┘
```

### nsenter 사용

```bash
# 호스트에서 컨테이너의 PID 가져오기
PID=$(docker inspect --format='{{.State.Pid}}' myapp)

# 컨테이너의 모든 네임스페이스에 들어가기
sudo nsenter -t $PID -m -u -i -n -p -- /bin/sh

# 네트워크 네임스페이스만 들어가기
sudo nsenter -t $PID -n -- ip addr

# PID 네임스페이스만 들어가기
sudo nsenter -t $PID -p -- ps aux

# 마운트 네임스페이스에 들어가서 컨테이너의 파일시스템 보기
sudo nsenter -t $PID -m -- ls /app

# 컨테이너의 네트워크 네임스페이스에서 iptables 규칙 확인
sudo nsenter -t $PID -n -- iptables -L -n
```

### 실용적인 nsenter 예시

```bash
# distroless 컨테이너 디버깅 (내부에 셸 없음)
PID=$(docker inspect --format='{{.State.Pid}}' distroless-app)

# 컨테이너의 네임스페이스에서 호스트 도구 사용
sudo nsenter -t $PID -n -- ss -tlnp        # 네트워크 소켓
sudo nsenter -t $PID -n -- ip route         # 라우팅 테이블
sudo nsenter -t $PID -m -- cat /app/config  # 설정 읽기
sudo nsenter -t $PID -p -- kill -SIGUSR1 1  # PID 1에 시그널 전송

# 컨테이너와 호스트 네임스페이스 비교
echo "호스트 네트워크:"
ip addr show
echo "컨테이너 네트워크:"
sudo nsenter -t $PID -n -- ip addr show
```

### /proc 탐색

```bash
# 호스트에서 컨테이너의 프로세스 정보
PID=$(docker inspect --format='{{.State.Pid}}' myapp)

# 컨테이너의 환경 변수 보기
sudo cat /proc/$PID/environ | tr '\0' '\n'

# 컨테이너의 파일 디스크립터 보기
sudo ls -la /proc/$PID/fd/

# 컨테이너의 메모리 맵 보기
sudo cat /proc/$PID/maps

# 컨테이너의 cgroup 제한 보기
sudo cat /proc/$PID/cgroup

# 컨테이너의 리소스 제한 확인
sudo cat /proc/$PID/limits
```

---

## 6. 시스템 콜 추적

### 이론: `strace`, `lsof`, 안에서의 검사

이것들이 리눅스 프로세스 디버깅의 일꾼이며, 모두 컨테이너 안에서 동작 — 그저 설치(또는 호스트에서 `nsenter`로 실행)되어야 함.

- **`strace -p <pid>`** — 프로세스가 만드는 모든 시스템 콜을 인자와 반환값과 함께 표시. 느림(가로채기 오버헤드 상당)이지만 "프로세스가 실제로 무엇을 하려 하나?"에 답하는 가장 정확한 방법. `-e trace=network`나 `-e trace=file`로 노이즈 줄임. `strace -f`이 다중 프로세스 앱의 fork 따라감.
- **`lsof -p <pid>`** — 경로/소켓/파이프와 함께 열린 파일 디스크립터 나열. 파일 누출, 프로세스가 실제로 바인딩된 포트, 어떤 설정 파일이 열려 있는지 찾기.
- **`pgrep`, `pkill`, `ps -ef`** — 기본 프로세스 검사. 컨테이너 PID 네임스페이스 안에서 PID는 1(엔트리포인트)부터 시작.
- **`top`, `htop`** — 컨테이너 시각의 자원 사용.
- **`/proc/self/status`와 `/proc/self/cgroup`** — *현재 실행 중인 셸*의 UID, capability, cgroup 확인. 디버그 명령이 상속하는 것.

이미지가 distroless(셸 없음, `strace` 없음, `lsof` 없음)일 때 워크플로는 `/proc/<pid>/`로 호스트에서 디버깅, 또는 **임시 디버그 컨테이너** 부착으로 이동.

### 컨테이너에서 strace

`strace`는 프로세스가 수행하는 시스템 콜을 추적합니다. 권한 오류, 파일 접근 문제, 프로세스 행업(hanging)을 디버깅하는 데 매우 유용합니다:

```bash
# 실행 중인 컨테이너에 strace 설치
docker exec myapp apt-get update && docker exec myapp apt-get install -y strace

# 프로세스의 모든 시스템 콜 추적
docker exec myapp strace -p 1

# 타임스탬프와 함께 추적
docker exec myapp strace -tt -p 1

# 특정 시스템 콜 추적
docker exec myapp strace -e trace=open,read,write -p 1

# 네트워크 관련 콜 추적
docker exec myapp strace -e trace=network -p 1

# 파일 관련 콜 추적
docker exec myapp strace -e trace=file -p 1

# 추적 결과를 파일로 저장
docker exec myapp strace -o /tmp/trace.log -p 1
docker cp myapp:/tmp/trace.log ./trace.log
```

### 호스트에서 strace

```bash
# nsenter를 사용한 추적 (컨테이너 내부에 strace 불필요)
PID=$(docker inspect --format='{{.State.Pid}}' myapp)

# SYS_PTRACE 권한(capability) 필요
docker run --cap-add=SYS_PTRACE ...

# 또는 호스트에서 추적
sudo strace -p $PID -e trace=network

# 컨테이너의 모든 프로세스 추적
sudo strace -p $PID -f -e trace=file
```

### 라이브러리 콜을 위한 ltrace

```bash
# ltrace는 라이브러리 콜을 추적 (예: malloc, printf)
docker exec myapp ltrace -p 1

# 특정 라이브러리 추적
docker exec myapp ltrace -e malloc+free -p 1
```

### strace를 사용한 실용적 디버깅

```bash
# "Permission denied" 오류 디버깅
docker exec myapp strace -e trace=open,openat,access -f -p 1
# 출력에서 EACCES 또는 EPERM 찾기

# "Connection refused" 오류 디버깅
docker exec myapp strace -e trace=connect -f -p 1
# ECONNREFUSED 찾기

# 느린 시작 디버깅
docker exec myapp strace -T -e trace=file -f -p 1
# -T는 각 시스템 콜에 소요된 시간 표시

# "No such file or directory" 디버깅
docker exec myapp strace -e trace=openat,stat -f -p 1
# ENOENT 찾기
```

---

## 7. 헬스 체크와 재시작 정책

### 이론: 헬스 체크와 재시작 루프

컨테이너의 재시작 정책이 "프로세스가 종료됨"을 "오케스트레이터가 재시작함"으로 바꿈. 이게 자기 의존성에 대한 서비스 거부가 아닌 자가 치유로 동작하려면 —

- **헬스 체크가 "살아 있음"을 정의.** Dockerfile의 `HEALTHCHECK CMD curl -f http://localhost/ || exit 1`, 또는 K8s의 `livenessProbe`. 주기적 실행. `start_period`이 첫 N초 스킵(느린 시작 위해). `retries`가 unhealthy 선언 전 연속 실패 카운트.
- **K8s의 Liveness vs Readiness.** Liveness 실패 → 컨테이너 재시작. Readiness 실패 → Service 로드 밸런싱에서 제거하지만 재시작 안 함. 흔한 버그 — 너무 많이 하는 단일 프로브(예: DB 쿼리) 사용해 DB가 잠시 느릴 때 healthy 앱을 무너뜨림.
- **백오프가 중요.** 모든 시작에서 즉시 크래시되는 컨테이너가 백오프 없이 호스트를 소비. Docker와 K8s 모두 지수 백오프 구현(K8s의 `CrashLoopBackOff`) — 즉시 재시작, 그 다음 10s, 20s, 40s, ... 5분까지. 잘못 구성된 Pod이 "CrashLoopBackOff에 갇힘"인 이유 — 갇힌 게 아니라 클러스터를 태우지 않도록 천천히 재시작되는 것.

"컨테이너가 계속 재시작" 디버깅 시 첫 질문 —

1. *종료 코드*가 무엇인가? `docker inspect -f '{{.State.ExitCode}}' <id>` 또는 `kubectl describe pod`. 0 = 깨끗한 종료(정책이 `always`라 오케스트레이터가 재시작), 0 아님 = 크래시, 137 = SIGKILL(아마 OOM), 143 = SIGTERM(보통 셧다운).
2. *마지막* 컨테이너가 무엇을 로깅했나? `docker logs --previous` 또는 `kubectl logs --previous`. 현재 컨테이너는 신선, *이전* 것의 로그에 실제 사망 메시지.
3. 호스트의 `dmesg`이 무엇이라 하나? OOM kill이 거기 죽은 PID와 메모리 합과 함께 나타남.

### Dockerfile HEALTHCHECK

```dockerfile
FROM nginx:alpine

# 기본 헬스 체크
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
  CMD curl -f http://localhost/ || exit 1
```

```dockerfile
FROM python:3.12-slim

# Python API를 위한 헬스 체크
HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

COPY . /app
CMD ["python", "/app/main.py"]
```

### 헬스 체크 매개변수

```
┌──────────────────────────────────────────────────────────────┐
│                헬스 체크 매개변수                               │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  --interval=30s     체크 실행 빈도                             │
│  --timeout=5s       단일 체크의 최대 시간                      │
│  --retries=3        "unhealthy" 전 연속 실패 횟수             │
│  --start-period=0s  컨테이너 시작 유예 기간                    │
│                                                               │
│  종료 코드:                                                    │
│  0 = healthy (정상)                                           │
│  1 = unhealthy (비정상)                                       │
│  2 = reserved (예약됨, 사용 금지)                              │
│                                                               │
│  상태 전환:                                                    │
│  starting ──(통과)──► healthy ──(실패 x retries)──► unhealthy │
│           ──(실패)──► starting (start-period 이내)             │
└──────────────────────────────────────────────────────────────┘
```

### 헬스 상태 모니터링

```bash
# 헬스 상태 확인
docker inspect --format='{{json .State.Health}}' myapp | jq

# 헬스 이벤트 감시
docker events --filter event=health_status

# 헬스 상태별 컨테이너 나열
docker ps --filter health=unhealthy
docker ps --filter health=healthy

# 헬스 체크 로그
docker inspect --format='{{json .State.Health.Log}}' myapp | jq
```

### 재시작 정책(Restart Policies)

```bash
# 재시작 안 함 (기본)
docker run --restart=no myapp

# 항상 재시작
docker run --restart=always myapp

# 실패 시 재시작 (최대 재시도 횟수 포함)
docker run --restart=on-failure:5 myapp

# 수동 중지 전까지 재시작
docker run --restart=unless-stopped myapp
```

```
┌──────────────────────────────────────────────────────────────┐
│                   재시작 정책                                  │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  no              자동 재시작 안 함                              │
│  always          항상 재시작 (수동 중지 + 재부팅 후에도)       │
│  on-failure[:N]  비정상 종료 코드에서만 재시작                  │
│                  선택: 최대 N번 재시도                          │
│  unless-stopped  always와 유사, 수동 중지 후에는 아님          │
│                                                               │
│  권장:                                                         │
│  ├─ 개발: no (기본)                                           │
│  ├─ 프로덕션: unless-stopped 또는 on-failure:10               │
│  └─ 시스템 서비스: always                                      │
└──────────────────────────────────────────────────────────────┘
```

### Docker Compose 헬스 체크

```yaml
# docker-compose.yml
version: "3.9"
services:
  app:
    image: myapp
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 30s
    restart: unless-stopped
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: secret
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
```

---

## 8. 멀티 컨테이너 애플리케이션 디버깅

### docker compose logs

```bash
# 모든 서비스의 로그 보기
docker compose logs

# 특정 서비스의 로그 팔로우
docker compose logs -f app db

# 서비스별 마지막 N줄 표시
docker compose logs --tail 50

# 타임스탬프 표시
docker compose logs -t
```

### Docker Events

```bash
# 모든 Docker 이벤트 실시간 모니터링
docker events

# 컨테이너별 필터
docker events --filter container=myapp

# 이벤트 유형별 필터
docker events --filter event=start --filter event=stop --filter event=die

# 시간 범위별 필터
docker events --since "2025-01-15T10:00:00" --until "2025-01-15T11:00:00"

# 파싱을 위한 JSON 형식
docker events --format '{{json .}}' | jq
```

### 리소스 모니터링

```bash
# 모든 컨테이너의 실시간 리소스 사용량
docker stats

# 일회성 리소스 스냅샷
docker stats --no-stream

# 출력 형식 지정
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# 특정 컨테이너 확인
docker stats app db redis

# 과도한 리소스를 사용하는 컨테이너 찾기
docker stats --no-stream --format '{{.Name}}\t{{.CPUPerc}}' | sort -t$'\t' -k2 -rn
```

### Compose 시작 순서 디버깅

```bash
# 비정상인 서비스 확인
docker compose ps

# depends_on 헬스 조건 테스트
docker compose up -d db
docker compose ps db  # healthy 대기
docker compose up -d app

# 시작 순서 관찰
docker compose up 2>&1 | grep -E "(Creating|Started|healthy|unhealthy)"
```

### 서비스 간 통신 검사

```bash
# Compose 네트워크 확인
docker network inspect $(docker compose config --format json | jq -r '.networks | keys[0]')

# Compose 내에서 DNS 해석
docker compose exec app nslookup db
docker compose exec app getent hosts db

# 서비스 간 요청 추적
docker compose exec app curl -v http://db:5432
```

---

## 9. 일반적인 문제와 해결 방법

### 문제 1: 컨테이너가 즉시 종료됨

```bash
# 종료 코드 확인
docker inspect --format='{{.State.ExitCode}}' myapp
# 0 = 정상 종료, 137 = 강제 종료 (OOM 또는 SIGKILL), 1 = 애플리케이션 오류

# 오류 로그 확인
docker logs myapp

# 대화형으로 실행하여 무슨 일이 발생하는지 확인
docker run -it myapp /bin/sh

# 일반적인 원인:
# - CMD/ENTRYPOINT가 백그라운드에서 실행 (exec 형식 사용)
# - 종속성 또는 설정 파일 누락
# - 권한 오류
```

### 문제 2: 컨테이너 OOM 종료

```bash
# OOM 종료 여부 확인
docker inspect --format='{{.State.OOMKilled}}' myapp

# 메모리 제한 확인
docker inspect --format='{{.HostConfig.Memory}}' myapp

# 실제 메모리 사용량 확인
docker stats --no-stream myapp

# 해결 방법:
# - 메모리 제한 증가: docker run -m 2g myapp
# - 애플리케이션 메모리 튜닝 (JVM 힙, Python 등)
# - docker events --filter event=oom으로 모니터링
```

### 문제 3: 권한 거부(Permission Denied)

```bash
# 실행 중인 사용자 확인
docker exec myapp whoami
docker exec myapp id

# 파일 권한 확인
docker exec myapp ls -la /app/data/

# 볼륨 마운트 권한 확인
docker inspect --format='{{json .Mounts}}' myapp | jq

# 해결 방법:
# - 컨테이너 UID를 볼륨 소유자와 일치시키기
# - --user 플래그 사용: docker run --user 1000:1000 myapp
# - Dockerfile에서 수정: RUN chown -R appuser:appuser /app
```

### 문제 4: DNS 해석 실패

```bash
# DNS 설정 확인
docker exec myapp cat /etc/resolv.conf

# 해석 테스트
docker exec myapp nslookup google.com

# Docker DNS 설정 확인
docker inspect --format='{{json .HostConfig.Dns}}' myapp

# 해결 방법:
# - 커스텀 DNS: docker run --dns 8.8.8.8 myapp
# - /etc/docker/daemon.json에서 Docker 데몬 DNS 확인
# - 컨테이너가 같은 네트워크에 있는지 확인
```

### 문제 5: 느린 컨테이너 시작

```bash
# 시작 시간 측정
time docker run --rm myapp echo "started"

# strace로 프로파일링
docker run --cap-add=SYS_PTRACE myapp strace -c -f /app/entrypoint.sh

# 일반적인 원인과 해결 방법:
# - 큰 이미지 풀: 더 작은 베이스 이미지 사용
# - 종속성 다운로드: 이미지 레이어에 캐시
# - 데이터베이스 마이그레이션: init 컨테이너 또는 헬스 체크 사용
# - DNS 타임아웃: DNS 서버를 명시적으로 구성
```

### 디버깅 체크리스트

```
┌──────────────────────────────────────────────────────────────┐
│              컨테이너 디버깅 체크리스트                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  □ 1. 컨테이너 상태 확인:     docker ps -a                    │
│  □ 2. 종료 코드 확인:         docker inspect (ExitCode)       │
│  □ 3. 로그 확인:              docker logs --tail 100          │
│  □ 4. 이벤트 확인:            docker events                   │
│  □ 5. 리소스 사용량 확인:     docker stats                    │
│  □ 6. OOM 확인:               docker inspect (OOMKilled)     │
│  □ 7. 마운트 확인:            docker inspect (Mounts)         │
│  □ 8. 네트워크 확인:          docker network inspect          │
│  □ 9. 헬스 확인:              docker inspect (Health)         │
│  □ 10. 대화형 디버그:         docker exec -it ... sh          │
│  □ 11. 작동하는 것과 비교:    diff <(inspect A) <(B)          │
│  □ 12. 시스템 콜:             strace -p PID                   │
└──────────────────────────────────────────────────────────────┘
```

---

## 10. 연습 문제

### 연습 1: 로그 조사 (초급)

nginx 컨테이너를 시작하고, 트래픽을 생성한 후, `docker logs`를 사용하여 접근 패턴을 분석하세요.

```bash
# 1. JSON 파일 로깅과 크기 제한으로 nginx 시작
# 2. curl을 사용하여 20개의 HTTP 요청 생성
# 3. docker logs --since를 사용하여 최근 항목 찾기
# 4. docker inspect를 사용하여 로그 파일 경로 찾기
```

<details>
<summary>풀이</summary>

```bash
# nginx 시작
docker run -d --name web \
  --log-opt max-size=5m --log-opt max-file=3 \
  -p 8080:80 nginx:alpine

# 트래픽 생성
for i in $(seq 1 20); do
  curl -s http://localhost:8080 > /dev/null
  curl -s http://localhost:8080/nonexistent > /dev/null 2>&1
done

# 최근 로그 보기
docker logs --since 5m web

# 404 오류 필터링
docker logs web 2>&1 | grep "404"

# 로그 파일 찾기
docker inspect --format='{{.LogPath}}' web

# 정리
docker rm -f web
```

</details>

### 연습 2: 컨테이너 검사 (초급)

특정 리소스 제한과 환경 변수로 컨테이너를 실행한 다음, `docker inspect`를 사용하여 모든 설정 세부사항을 추출하세요.

<details>
<summary>풀이</summary>

```bash
# 설정과 함께 실행
docker run -d --name inspect-test \
  -m 256m --cpus=0.5 \
  -e APP_ENV=production \
  -e APP_DEBUG=false \
  -p 9090:80 \
  -v testdata:/data \
  --restart=on-failure:3 \
  nginx:alpine

# 다양한 필드 검사
echo "상태: $(docker inspect --format='{{.State.Status}}' inspect-test)"
echo "PID: $(docker inspect --format='{{.State.Pid}}' inspect-test)"
echo "메모리 제한: $(docker inspect --format='{{.HostConfig.Memory}}' inspect-test)"
echo "CPU: $(docker inspect --format='{{.HostConfig.NanoCpus}}' inspect-test)"
echo "재시작 정책: $(docker inspect --format='{{.HostConfig.RestartPolicy.Name}}' inspect-test)"
echo "IP: $(docker inspect --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' inspect-test)"

# 환경 변수
docker inspect --format='{{json .Config.Env}}' inspect-test | jq

# 마운트
docker inspect --format='{{json .Mounts}}' inspect-test | jq

# 정리
docker rm -f inspect-test
docker volume rm testdata
```

</details>

### 연습 3: 네트워크 디버깅 (중급)

데이터베이스에 연결할 수 없는 앱이 있는 멀티 컨테이너 환경을 설정하고, 문제를 디버깅하세요.

<details>
<summary>풀이</summary>

```bash
# 두 개의 별도 네트워크 생성 (잘못된 설정 시뮬레이션)
docker network create frontend
docker network create backend

# backend 네트워크에서 데이터베이스 시작
docker run -d --name db \
  --network backend \
  -e POSTGRES_PASSWORD=secret \
  postgres:16-alpine

# frontend 네트워크에서 앱 시작 (잘못됨 -- db에 접근 불가)
docker run -d --name app \
  --network frontend \
  alpine sleep 3600

# 디버그: app에서 db에 접근 시도
docker exec app ping -c 1 db
# ping: bad address 'db' -- DNS 해석 실패

# 네트워크 확인
docker network inspect frontend --format='{{range .Containers}}{{.Name}} {{end}}'
docker network inspect backend --format='{{range .Containers}}{{.Name}} {{end}}'

# 수정: app을 backend 네트워크에 연결
docker network connect backend app

# 수정 확인
docker exec app ping -c 1 db
# 이제 작동해야 함

# 정리
docker rm -f app db
docker network rm frontend backend
```

</details>

### 연습 4: 헬스 체크 디버깅 (중급)

처음에 실패하는 헬스 체크가 있는 컨테이너를 만들고, 왜 실패하는지 디버깅하고, 수정한 후 헬스 상태 전환을 확인하세요.

<details>
<summary>풀이</summary>

```bash
# 실패할 헬스 체크가 있는 컨테이너 생성
docker run -d --name health-test \
  --health-cmd="curl -f http://localhost:80/" \
  --health-interval=5s \
  --health-retries=3 \
  --health-start-period=5s \
  alpine sleep 3600

# 헬스 상태 변경 감시
docker events --filter container=health-test --filter event=health_status &
EVENT_PID=$!

# 대기 후 상태 확인
sleep 20
docker inspect --format='{{.State.Health.Status}}' health-test
# unhealthy -- alpine에 curl과 웹 서버가 없기 때문

# 세부사항에 대한 헬스 로그 확인
docker inspect --format='{{json .State.Health.Log}}' health-test | jq

# 이벤트 감시자 정리
kill $EVENT_PID 2>/dev/null

# 수정: 적절한 헬스 체크가 있는 컨테이너 생성
docker rm -f health-test
docker run -d --name health-test \
  --health-cmd="wget -qO- http://localhost:80/ || exit 1" \
  --health-interval=5s \
  --health-retries=3 \
  --health-start-period=10s \
  nginx:alpine

# 대기 후 확인
sleep 15
docker inspect --format='{{.State.Health.Status}}' health-test
# healthy

# 정리
docker rm -f health-test
```

</details>

### 연습 5: 전체 디버깅 워크플로우 (고급)

멀티 컨테이너 애플리케이션(web + API + 데이터베이스)에 문제가 있습니다. 하나의 컨테이너가 충돌하고, 다른 하나는 연결 문제가 있습니다. 이 레슨의 모든 디버깅 도구를 사용하여 진단하고 수정하세요.

<details>
<summary>풀이</summary>

```yaml
# docker-compose.yml -- 의도적으로 깨진 설정
version: "3.9"
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: myapp
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      retries: 5

  api:
    image: python:3.12-slim
    command: >
      sh -c "pip install flask psycopg2-binary &&
             python -c \"
      from flask import Flask
      app = Flask(__name__)
      @app.route('/health')
      def health(): return 'ok'
      app.run(host='0.0.0.0', port=5000)
      \""
    depends_on:
      db:
        condition: service_healthy

  web:
    image: nginx:alpine
    ports:
      - "8080:80"
```

```bash
# 1단계: 스택 시작
docker compose up -d

# 2단계: 상태 확인
docker compose ps

# 3단계: 오류 로그 확인
docker compose logs api
docker compose logs db

# 4단계: 헬스 확인
docker inspect --format='{{json .State.Health.Status}}' $(docker compose ps -q db)

# 5단계: 네트워크 연결 확인
docker compose exec web ping -c 1 api
docker compose exec api ping -c 1 db

# 6단계: API가 리스닝하는지 확인
docker compose exec api ss -tlnp

# 7단계: 리소스 사용량 확인
docker stats --no-stream

# 8단계: 컨테이너 이벤트 확인
docker events --since 5m --filter label=com.docker.compose.project

# 9단계: 발견된 문제 수정 후 재배포
docker compose down
# docker-compose.yml 수정
docker compose up -d

# 10단계: 모든 것이 작동하는지 확인
curl http://localhost:8080
docker compose ps
docker compose logs
```

</details>

---

**이전**: [Podman과 OCI](./15_Podman_and_OCI.md)
