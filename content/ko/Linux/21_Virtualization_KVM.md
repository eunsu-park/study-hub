# 가상화 (KVM)

**이전**: [커널 관리](./20_Kernel_Management.md) | **다음**: [Ansible 기초](./22_Ansible_Basics.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. KVM/QEMU 아키텍처와 libvirt가 통합 관리 레이어를 제공하는 방식을 설명한다
2. 하드웨어 가상화 지원을 확인하고 KVM 가상화 스택을 설치한다
3. CPU, 메모리, 디스크, 네트워크 옵션을 지정하여 virt-install로 가상 머신을 생성한다
4. virsh 명령으로 VM 생명주기(시작, 중지, 스냅샷, 마이그레이션)를 관리한다
5. NAT, 브릿지(Bridged), 격리(Isolated) 네트워크를 포함한 가상 네트워킹을 설정한다
6. 호스트 간 실시간(Live) 및 오프라인 VM 마이그레이션을 수행한다
7. virsh와 qemu-img로 스토리지 풀(Storage Pool), 볼륨, 디스크 이미지를 관리한다

**난이도**: ⭐⭐⭐⭐ (고급)

---

## 목차

1. [KVM/QEMU 개요](#1-kvmqemu-개요)
2. [설치 및 설정](#2-설치-및-설정)
3. [VM 생성](#3-vm-생성)
4. [virsh 명령어](#4-virsh-명령어)
5. [네트워크 설정](#5-네트워크-설정)
6. [스토리지 관리](#6-스토리지-관리)
7. [스냅샷과 마이그레이션](#7-스냅샷과-마이그레이션)

---

KVM은 Linux 커널 자체를 하이퍼바이저(Hypervisor)로 만들어, 독점 솔루션의 비용 없이 가상 머신에 거의 네이티브에 가까운 성능을 제공합니다. 개발 환경 구축, 서버 통합, 프라이빗 클라우드 인프라 구성 등 어떤 목적이든 KVM은 Linux 가상화의 기반입니다. 이 레슨에서는 OpenStack과 주요 호스팅 업체들이 실제로 사용하는 도구들로 VM을 생성, 관리, 마이그레이션하는 방법을 다룹니다.

## 1. KVM/QEMU 개요

### 가상화 유형

```
┌─────────────────────────────────────────────────────────────┐
│  Type 1 (Bare-metal)          Type 2 (Hosted)               │
│  ┌─────────────────┐          ┌─────────────────┐          │
│  │ VM │ VM │ VM   │          │ VM │ VM │ VM   │          │
│  ├─────────────────┤          ├─────────────────┤          │
│  │   Hypervisor    │          │   Hypervisor    │          │
│  ├─────────────────┤          ├─────────────────┤          │
│  │    Hardware     │          │   Host OS       │          │
│  └─────────────────┘          ├─────────────────┤          │
│  ESXi, Xen, Hyper-V           │    Hardware     │          │
│                               └─────────────────┘          │
│                               VirtualBox, VMware Workstation│
│                                                             │
│  KVM은 Linux 커널이 Hypervisor 역할 (Type 1에 가까움)       │
└─────────────────────────────────────────────────────────────┘
```

### KVM/QEMU 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                       가상 머신                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Guest OS (Linux, Windows, etc.)                     │   │
│  └─────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  QEMU: 장치 에뮬레이션 (I/O, 네트워크, 디스크)              │
├─────────────────────────────────────────────────────────────┤
│  KVM 커널 모듈: CPU/메모리 가상화 (하드웨어 지원)           │
├─────────────────────────────────────────────────────────────┤
│  Linux 커널                                                  │
├─────────────────────────────────────────────────────────────┤
│  하드웨어 (VT-x/AMD-V, VT-d)                                │
└─────────────────────────────────────────────────────────────┘
```

### libvirt 관리 스택

```
┌─────────────────────────────────────────────────────────────┐
│  관리 도구                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ virsh    │  │virt-manager│ │Cockpit   │  │ API      │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │             │             │          │
│       └─────────────┴─────────────┴─────────────┘          │
│                           │                                 │
│                    ┌──────┴──────┐                         │
│                    │   libvirt   │                         │
│                    │ (libvirtd)  │                         │
│                    └──────┬──────┘                         │
│                           │                                 │
│  ┌────────────────────────┼────────────────────────┐       │
│  │            │           │           │            │       │
│  │  ┌─────┐   │   ┌─────┐│   ┌─────┐│   ┌─────┐  │       │
│  │  │ KVM │   │   │ QEMU││   │ LXC ││   │ Xen │  │       │
│  │  └─────┘   │   └─────┘│   └─────┘│   └─────┘  │       │
│  └────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 설치 및 설정

### 하드웨어 가상화 지원 확인

```bash
# CPU 가상화 지원 확인
grep -E '(vmx|svm)' /proc/cpuinfo

# vmx: Intel VT-x
# svm: AMD-V

# 또는
lscpu | grep Virtualization

# KVM 모듈 확인
lsmod | grep kvm
```

### 패키지 설치

```bash
# Ubuntu/Debian
sudo apt install qemu-kvm libvirt-daemon-system libvirt-clients \
    bridge-utils virt-manager virtinst

# RHEL/CentOS
sudo yum install qemu-kvm libvirt libvirt-python \
    libguestfs-tools virt-install virt-manager

# Fedora
sudo dnf install @virtualization
```

### 서비스 시작

```bash
# libvirtd 시작
sudo systemctl enable --now libvirtd

# 상태 확인
sudo systemctl status libvirtd

# 사용자를 libvirt 그룹에 추가 (재로그인 필요)
sudo usermod -aG libvirt $USER
sudo usermod -aG kvm $USER
```

### 연결 확인

```bash
# 로컬 연결 테스트
virsh -c qemu:///system list --all

# 또는 sudo 없이
virsh list --all

# 시스템 정보
virsh nodeinfo
```

---

## 3. VM 생성

### virt-install을 이용한 VM 생성

```bash
# 기본 VM 생성
virt-install \
    --name ubuntu-vm \
    --ram 2048 \
    --vcpus 2 \
    --disk path=/var/lib/libvirt/images/ubuntu-vm.qcow2,size=20 \
    --os-variant ubuntu22.04 \
    --network bridge=virbr0 \
    --graphics vnc,listen=0.0.0.0 \
    --cdrom /path/to/ubuntu-22.04.iso \
    --boot cdrom,hd
```

### 상세 옵션

```bash
virt-install \
    --name centos-vm \
    --memory 4096 \
    --vcpus 4,maxvcpus=8 \
    --cpu host-passthrough \
    --disk path=/var/lib/libvirt/images/centos-vm.qcow2,size=40,format=qcow2,bus=virtio \
    --disk path=/var/lib/libvirt/images/centos-data.qcow2,size=100,format=qcow2 \
    --os-variant centos-stream9 \
    --network network=default,model=virtio \
    --graphics spice,listen=0.0.0.0 \
    --video qxl \
    --channel spicevmc \
    --location /path/to/CentOS-Stream-9.iso \
    --extra-args "console=ttyS0,115200n8 serial" \
    --initrd-inject /path/to/kickstart.cfg \
    --extra-args "ks=file:/kickstart.cfg" \
    --noautoconsole
```

### OS 변형 목록 확인

```bash
# 지원하는 OS 목록
osinfo-query os

# 특정 OS 검색
osinfo-query os | grep -i ubuntu
osinfo-query os | grep -i centos
```

### XML 정의로 VM 생성

```bash
# VM 정의 XML 예시 (/tmp/vm-definition.xml)
virsh define /tmp/vm-definition.xml

# 기존 VM의 XML 덤프
virsh dumpxml ubuntu-vm > ubuntu-vm.xml
```

```xml
<!-- vm-definition.xml 예시 -->
<domain type='kvm'>
  <name>test-vm</name>
  <memory unit='GiB'>2</memory>
  <vcpu placement='static'>2</vcpu>
  <os>
    <type arch='x86_64' machine='pc-q35-6.2'>hvm</type>
    <boot dev='hd'/>
  </os>
  <features>
    <acpi/>
    <apic/>
  </features>
  <cpu mode='host-passthrough'/>
  <devices>
    <emulator>/usr/bin/qemu-system-x86_64</emulator>
    <disk type='file' device='disk'>
      <driver name='qemu' type='qcow2'/>
      <source file='/var/lib/libvirt/images/test-vm.qcow2'/>
      <target dev='vda' bus='virtio'/>
    </disk>
    <interface type='network'>
      <source network='default'/>
      <model type='virtio'/>
    </interface>
    <graphics type='vnc' port='-1' autoport='yes'/>
    <console type='pty'/>
  </devices>
</domain>
```

---

## 4. virsh 명령어

### VM 상태 관리

```bash
# VM 목록
virsh list             # 실행 중인 VM
virsh list --all       # 모든 VM
virsh list --inactive  # 중지된 VM

# VM 시작/중지
virsh start vm-name
virsh shutdown vm-name     # 정상 종료
virsh destroy vm-name      # 강제 종료 (전원 차단)
virsh reboot vm-name

# VM 일시 정지/재개
virsh suspend vm-name
virsh resume vm-name

# 자동 시작 설정
virsh autostart vm-name
virsh autostart --disable vm-name
```

### VM 정보 조회

```bash
# 기본 정보
virsh dominfo vm-name

# 상세 설정 (XML)
virsh dumpxml vm-name

# CPU 정보
virsh vcpuinfo vm-name

# 메모리 정보
virsh dommemstat vm-name

# 블록 장치 정보
virsh domblklist vm-name
virsh domblkinfo vm-name vda

# 네트워크 인터페이스
virsh domiflist vm-name
virsh domifstat vm-name vnet0
```

### VM 콘솔 접속

```bash
# 시리얼 콘솔 (게스트 설정 필요)
virsh console vm-name

# VNC/SPICE 포트 확인
virsh domdisplay vm-name
virsh vncdisplay vm-name

# virt-viewer로 접속
virt-viewer vm-name
```

### VM 리소스 변경

```bash
# CPU 수 변경 (온라인)
virsh setvcpus vm-name 4 --live

# 메모리 변경 (온라인, 사전 설정 필요)
virsh setmem vm-name 4G --live

# 최대값 설정 (오프라인)
virsh setmaxmem vm-name 8G --config
virsh setvcpus vm-name 8 --maximum --config

# XML 직접 편집
virsh edit vm-name
```

### VM 삭제

```bash
# VM 정의 삭제 (디스크 유지)
virsh undefine vm-name

# VM과 관련 스토리지 삭제
virsh undefine vm-name --remove-all-storage

# NVRAM 포함 삭제 (UEFI)
virsh undefine vm-name --nvram
```

---

## 5. 네트워크 설정

### 기본 NAT 네트워크

```bash
# 기본 네트워크 확인
virsh net-list --all

# 기본 네트워크 시작
virsh net-start default
virsh net-autostart default

# 네트워크 정보
virsh net-info default
virsh net-dumpxml default
```

### 브릿지 네트워크 설정

```bash
# Ubuntu: netplan 설정
# /etc/netplan/01-bridge.yaml
```

```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    enp3s0:
      dhcp4: no
  bridges:
    br0:
      interfaces:
        - enp3s0
      dhcp4: yes
      parameters:
        stp: false
        forward-delay: 0
```

```bash
# 적용
sudo netplan apply
```

```bash
# RHEL/CentOS: NetworkManager
sudo nmcli connection add type bridge con-name br0 ifname br0
sudo nmcli connection add type ethernet con-name br0-slave-enp3s0 ifname enp3s0 master br0
sudo nmcli connection modify br0 ipv4.method auto
sudo nmcli connection up br0
```

### libvirt 브릿지 네트워크 정의

```xml
<!-- /tmp/bridge-network.xml -->
<network>
  <name>host-bridge</name>
  <forward mode='bridge'/>
  <bridge name='br0'/>
</network>
```

```bash
# 네트워크 생성
virsh net-define /tmp/bridge-network.xml
virsh net-start host-bridge
virsh net-autostart host-bridge
```

### 격리된 네트워크

```xml
<!-- /tmp/isolated-network.xml -->
<network>
  <name>isolated</name>
  <bridge name='virbr1'/>
  <ip address='192.168.100.1' netmask='255.255.255.0'>
    <dhcp>
      <range start='192.168.100.10' end='192.168.100.254'/>
    </dhcp>
  </ip>
</network>
```

### VM에 네트워크 인터페이스 추가

```bash
# 인터페이스 추가 (온라인)
virsh attach-interface vm-name network default --model virtio --live --config

# 인터페이스 제거
virsh detach-interface vm-name network --mac 52:54:00:xx:xx:xx --live --config

# 인터페이스 목록
virsh domiflist vm-name
```

---

## 6. 스토리지 관리

### 스토리지 풀

```bash
# 기본 풀 (디렉토리 기반)
virsh pool-list --all

# 풀 생성 (디렉토리)
virsh pool-define-as mypool dir --target /var/lib/libvirt/images/mypool
virsh pool-build mypool
virsh pool-start mypool
virsh pool-autostart mypool

# LVM 풀
virsh pool-define-as lvm-pool logical --source-name vg_vms --target /dev/vg_vms

# 풀 정보
virsh pool-info default
```

### 볼륨 관리

```bash
# 볼륨 목록
virsh vol-list default

# 볼륨 생성
virsh vol-create-as default disk1.qcow2 20G --format qcow2

# 볼륨 정보
virsh vol-info disk1.qcow2 --pool default
virsh vol-dumpxml disk1.qcow2 --pool default

# 볼륨 삭제
virsh vol-delete disk1.qcow2 --pool default

# 볼륨 복제
virsh vol-clone disk1.qcow2 disk1-clone.qcow2 --pool default

# 볼륨 리사이즈
virsh vol-resize disk1.qcow2 30G --pool default
```

### 디스크 이미지 관리 (qemu-img)

```bash
# 이미지 생성
qemu-img create -f qcow2 disk.qcow2 20G

# 이미지 정보
qemu-img info disk.qcow2

# 이미지 변환
qemu-img convert -f raw -O qcow2 disk.raw disk.qcow2
qemu-img convert -f vmdk -O qcow2 disk.vmdk disk.qcow2

# 이미지 리사이즈
qemu-img resize disk.qcow2 +10G

# 스파스 이미지 압축
qemu-img convert -O qcow2 -c disk.qcow2 disk-compressed.qcow2
```

### VM에 디스크 추가

```bash
# 디스크 추가 (온라인)
virsh attach-disk vm-name /var/lib/libvirt/images/extra.qcow2 vdb \
    --driver qemu --subdriver qcow2 --live --config

# 디스크 제거
virsh detach-disk vm-name vdb --live --config

# 블록 장치 목록
virsh domblklist vm-name
```

---

## 7. 스냅샷과 마이그레이션

### 스냅샷 관리

```bash
# 스냅샷 생성
virsh snapshot-create-as vm-name snap1 "First snapshot" --atomic

# 디스크만 스냅샷 (메모리 제외)
virsh snapshot-create-as vm-name snap-disk --disk-only --atomic

# 스냅샷 목록
virsh snapshot-list vm-name

# 스냅샷 정보
virsh snapshot-info vm-name snap1

# 스냅샷으로 복원
virsh snapshot-revert vm-name snap1

# 스냅샷 삭제
virsh snapshot-delete vm-name snap1

# 현재 스냅샷 확인
virsh snapshot-current vm-name
```

### 외부 스냅샷

```bash
# 외부 스냅샷 생성 (운영 환경 권장)
virsh snapshot-create-as vm-name snap-external \
    --diskspec vda,snapshot=external \
    --disk-only --atomic

# 외부 스냅샷 커밋 (병합)
virsh blockcommit vm-name vda --active --pivot

# 블록 작업 정보
virsh blockjob vm-name vda --info
```

### 라이브 마이그레이션

```bash
# 대상 호스트 준비
# - 동일한 libvirt 버전
# - 공유 스토리지 (NFS, GlusterFS, Ceph 등)
# - 네트워크 연결

# 마이그레이션 실행
virsh migrate --live vm-name qemu+ssh://target-host/system

# 옵션과 함께
virsh migrate --live --persistent --undefinesource \
    --copy-storage-all \
    vm-name qemu+ssh://target-host/system

# 터널 마이그레이션 (NAT 환경)
virsh migrate --live --p2p --tunnelled \
    vm-name qemu+ssh://target-host/system

# 마이그레이션 상태 확인
virsh domjobinfo vm-name
```

### 오프라인 마이그레이션

```bash
# 원본에서 XML 덤프
virsh dumpxml vm-name > vm.xml

# 디스크 이미지 복사
rsync -av /var/lib/libvirt/images/vm-disk.qcow2 target-host:/var/lib/libvirt/images/

# 대상에서 정의
virsh define vm.xml

# 원본에서 삭제
virsh undefine vm-name
```

---

## 연습 문제

### 문제 1: VM 생성

virt-install을 사용하여 다음 사양의 VM을 생성하세요:
- 이름: test-server
- 메모리: 2GB
- CPU: 2개
- 디스크: 20GB (qcow2)
- 네트워크: default (NAT)
- 그래픽: VNC

### 문제 2: 네트워크 설정

격리된 내부 네트워크를 생성하세요:
- 이름: internal
- 대역: 10.10.10.0/24
- DHCP: 10.10.10.100-200
- NAT 없음

### 문제 3: 스냅샷 관리

1. VM의 스냅샷을 생성하세요
2. VM에 변경을 가하세요 (파일 생성 등)
3. 스냅샷으로 복원하세요
4. 변경이 취소되었는지 확인하세요

---

## 정답

### 문제 1 정답

```bash
virt-install \
    --name test-server \
    --memory 2048 \
    --vcpus 2 \
    --disk path=/var/lib/libvirt/images/test-server.qcow2,size=20,format=qcow2 \
    --os-variant generic \
    --network network=default \
    --graphics vnc,listen=0.0.0.0 \
    --cdrom /path/to/installer.iso \
    --boot cdrom,hd
```

### 문제 2 정답

```xml
<!-- /tmp/internal-net.xml -->
<network>
  <name>internal</name>
  <bridge name='virbr-int'/>
  <ip address='10.10.10.1' netmask='255.255.255.0'>
    <dhcp>
      <range start='10.10.10.100' end='10.10.10.200'/>
    </dhcp>
  </ip>
</network>
```

```bash
virsh net-define /tmp/internal-net.xml
virsh net-start internal
virsh net-autostart internal
```

### 문제 3 정답

```bash
# 1. 스냅샷 생성
virsh snapshot-create-as vm-name before-change "Before changes"

# 2. VM 내에서 변경
virsh console vm-name
# (게스트 내에서) touch /tmp/test-file

# 3. 스냅샷으로 복원
virsh snapshot-revert vm-name before-change

# 4. 확인 (파일이 없어야 함)
virsh console vm-name
# (게스트 내에서) ls /tmp/test-file  # 파일 없음
```

---

## 다음 단계

- [22_Ansible_기초.md](./22_Ansible_기초.md) - 인프라 자동화

---

## 참고 자료

- [libvirt Documentation](https://libvirt.org/docs.html)
- [KVM Documentation](https://www.linux-kvm.org/page/Documents)
- [Red Hat Virtualization Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/9/html/configuring_and_managing_virtualization/index)
- `man virsh`, `man virt-install`, `man qemu-img`
