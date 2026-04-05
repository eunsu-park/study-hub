# 26. 임베디드 Rust

**이전**: [WebAssembly](./09_WebAssembly.md) | **다음**: [네트워크 프로그래밍](./11_Network_Programming.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 표준 라이브러리 없이 베어메탈 환경을 위한 `no_std` Rust 코드 작성하기
2. 이식 가능한 드라이버 개발을 위한 `embedded-hal` 트레이트 사용하기
3. RTIC(실시간 인터럽트 기반 동시성) 프레임워크로 실시간 애플리케이션 빌드하기
4. 직접 하드웨어 레지스터 접근을 위한 메모리 매핑 I/O 수행하기
5. `probe-rs`로 임베디드 장치 디버깅 및 플래싱하기

---

Rust는 임베디드 시스템에서 점점 더 채택되고 있습니다. 소유권 모델은 C 펌웨어를 괴롭히는 버퍼 오버플로, use-after-free, 데이터 경쟁 등의 버그 전체 클래스를 방지합니다. 제로 비용 추상화와 런타임 오버헤드 없이 컴파일 타임 안전 보장을 제공하면서 수작업 C에 비견되는 바이너리를 생성합니다.

## 목차
1. [왜 임베디드에 Rust인가?](#1-왜-임베디드에-rust인가)
2. [no_std 기초](#2-no_std-기초)
3. [타겟 아키텍처 설정](#3-타겟-아키텍처-설정)
4. [메모리 레이아웃과 링커 스크립트](#4-메모리-레이아웃과-링커-스크립트)
5. [embedded-hal: 하드웨어 추상화 레이어](#5-embedded-hal-하드웨어-추상화-레이어)
6. [GPIO, SPI, I2C, UART](#6-gpio-spi-i2c-uart)
7. [메모리 매핑 I/O](#7-메모리-매핑-io)
8. [인터럽트와 예외 처리](#8-인터럽트와-예외-처리)
9. [RTIC 프레임워크](#9-rtic-프레임워크)
10. [probe-rs: 플래시와 디버그](#10-probe-rs-플래시와-디버그)
11. [임베디드 패턴](#11-임베디드-패턴)
12. [연습문제](#12-연습문제)

---

## 1. 왜 임베디드에 Rust인가?

### C와의 비교

| 측면 | C | Rust |
|------|---|------|
| 메모리 안전성 | 수동 | 컴파일 타임 보장 |
| 데이터 경쟁 방지 | 없음 | 소유권 시스템 |
| 버퍼 오버플로 | 흔함 | 불가능 (안전한 코드) |
| 널 역참조 | 흔함 | 널 없음 (`Option`) |
| 바이너리 크기 | 작음 | 비슷함 |
| 런타임 오버헤드 | 없음 | 없음 |
| 에코시스템 성숙도 | 수십 년 | 빠르게 성장 중 |

### 임베디드 Rust 에코시스템

```
┌─────────────────────────────────────────┐
│  애플리케이션                           │
├─────────────────────────────────────────┤
│  BSP (보드 지원 패키지)                 │
│  예: stm32f4xx-hal, rp-pico            │
├─────────────────────────────────────────┤
│  HAL (하드웨어 추상화 레이어)           │
│  예: stm32f4xx-hal                     │
├─────────────────────────────────────────┤
│  PAC (주변장치 접근 크레이트)           │
│  예: stm32f4 (SVD에서 자동 생성)       │
├─────────────────────────────────────────┤
│  embedded-hal 트레이트                  │
├─────────────────────────────────────────┤
│  cortex-m / cortex-m-rt                 │
│  (코어 지원 & 런타임)                   │
└─────────────────────────────────────────┘
```

---

## 2. no_std 기초

임베디드 시스템에는 운영체제가 없습니다 — 힙도, 스레드도, 파일시스템도 없습니다. `#![no_std]`는 컴파일러에게 표준 라이브러리를 링킹하지 말도록 지시합니다:

```rust
#![no_std]
#![no_main]

use core::panic::PanicInfo;

// no_std에서는 자체 패닉 핸들러를 정의해야 함
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}  // 패닉 시 정지
}

// main 함수 없음 — 진입점은 런타임 크레이트가 정의
```

### no_std에서 사용 가능한 것

```rust
#![no_std]

// core::는 항상 사용 가능 (할당 없음, OS 없음)
use core::fmt;           // 포매팅 (write!, format_args!)
use core::ops;           // 연산자 오버로딩
use core::iter;          // Iterator 트레이트와 어댑터
use core::option::Option;
use core::result::Result;
use core::cell::Cell;    // 내부 가변성
use core::ptr;           // 원시 포인터 연산
use core::mem;           // 메모리 연산
use core::sync::atomic;  // 원자적 연산

// alloc::는 글로벌 할당자가 있으면 사용 가능
// extern crate alloc;
// use alloc::vec::Vec;
// use alloc::string::String;
// use alloc::boxed::Box;
```

### alloc을 사용한 no_std

임베디드 타겟에 힙이 있다면 (많은 경우 있음):

```rust
#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use alloc::string::String;
use alloc::format;

// 글로벌 할당자 정의 필요
use embedded_alloc::LlffHeap as Heap;

#[global_allocator]
static HEAP: Heap = Heap::empty();

fn init_heap() {
    // 정적 버퍼로 힙 초기화
    const HEAP_SIZE: usize = 4096;
    static mut HEAP_MEM: [u8; HEAP_SIZE] = [0; HEAP_SIZE];
    unsafe {
        HEAP.init(HEAP_MEM.as_ptr() as usize, HEAP_SIZE);
    }
}
```

---

## 3. 타겟 아키텍처 설정

### 일반적인 임베디드 타겟

```bash
# ARM Cortex-M0/M0+ (하드웨어 FPU 없음)
rustup target add thumbv6m-none-eabi

# ARM Cortex-M3
rustup target add thumbv7m-none-eabi

# ARM Cortex-M4/M7 (하드웨어 FPU 포함)
rustup target add thumbv7em-none-eabihf

# RISC-V 32비트
rustup target add riscv32imac-unknown-none-elf

# RISC-V 64비트
rustup target add riscv64gc-unknown-none-elf
```

### 프로젝트 구성

```toml
# .cargo/config.toml
[target.thumbv7em-none-eabihf]
runner = "probe-rs run --chip STM32F411CEUx"
rustflags = ["-C", "link-arg=-Tlink.x"]

[build]
target = "thumbv7em-none-eabihf"
```

```toml
# Cargo.toml
[package]
name = "blinky"
version = "0.1.0"
edition = "2021"

[dependencies]
cortex-m = "0.7"
cortex-m-rt = "0.7"
panic-halt = "0.2"

# 특정 칩을 위한 HAL
stm32f4xx-hal = { version = "0.21", features = ["stm32f411"] }

[profile.release]
opt-level = "s"
debug = true       # 디버깅을 위한 디버그 정보 유지
lto = true
```

---

## 4. 메모리 레이아웃과 링커 스크립트

임베디드 시스템은 명시적인 메모리 레이아웃이 필요합니다:

```
/* memory.x — 링커 스크립트 */
MEMORY
{
  FLASH : ORIGIN = 0x08000000, LENGTH = 512K
  RAM   : ORIGIN = 0x20000000, LENGTH = 128K
}

/* 선택사항: 특정 섹션 배치 */
SECTIONS
{
  .data : ALIGN(4)
  {
    *(.data .data.*);
  } > RAM AT > FLASH
}
```

```rust
// cortex-m-rt 크레이트가 벡터 테이블과 시작 코드를 처리:
// 1. FLASH에서 RAM으로 .data 복사
// 2. .bss를 0으로 초기화
// 3. main() 호출

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;

#[entry]
fn main() -> ! {
    // 진입점 — 절대 반환하지 않음 (-> !)
    loop {
        cortex_m::asm::wfi();  // 인터럽트 대기 (저전력)
    }
}
```

### 정적 변수

```rust
use core::cell::RefCell;
use cortex_m::interrupt::Mutex;

// 임베디드에서 전역 가변 상태는 신중한 처리가 필요
// 임계 구역과 함께 Mutex<RefCell<T>> 사용
static COUNTER: Mutex<RefCell<u32>> = Mutex::new(RefCell::new(0));

fn increment() {
    cortex_m::interrupt::free(|cs| {
        let mut counter = COUNTER.borrow(cs).borrow_mut();
        *counter += 1;
    });
}

fn get_count() -> u32 {
    cortex_m::interrupt::free(|cs| {
        *COUNTER.borrow(cs).borrow()
    })
}
```

---

## 5. embedded-hal: 하드웨어 추상화 레이어

`embedded-hal`은 일반적인 하드웨어 주변장치에 대한 트레이트를 정의합니다. 이 트레이트를 기반으로 작성된 드라이버는 모든 칩에서 작동합니다:

```rust
// embedded-hal 트레이트 (간략화)
pub trait OutputPin {
    type Error;
    fn set_high(&mut self) -> Result<(), Self::Error>;
    fn set_low(&mut self) -> Result<(), Self::Error>;
}

pub trait InputPin {
    type Error;
    fn is_high(&mut self) -> Result<bool, Self::Error>;
    fn is_low(&mut self) -> Result<bool, Self::Error>;
}

pub trait DelayNs {
    fn delay_ns(&mut self, ns: u32);
    fn delay_us(&mut self, us: u32) {
        self.delay_ns(us * 1000);
    }
    fn delay_ms(&mut self, ms: u32) {
        self.delay_ns(ms * 1_000_000);
    }
}
```

### 이식 가능한 드라이버 작성

```rust
use embedded_hal::digital::OutputPin;
use embedded_hal::delay::DelayNs;

/// 일반 LED 드라이버 — 모든 칩에서 작동
pub struct Led<P: OutputPin> {
    pin: P,
    is_on: bool,
}

impl<P: OutputPin> Led<P> {
    pub fn new(pin: P) -> Self {
        Led { pin, is_on: false }
    }

    pub fn on(&mut self) -> Result<(), P::Error> {
        self.pin.set_high()?;
        self.is_on = true;
        Ok(())
    }

    pub fn off(&mut self) -> Result<(), P::Error> {
        self.pin.set_low()?;
        self.is_on = false;
        Ok(())
    }

    pub fn toggle(&mut self) -> Result<(), P::Error> {
        if self.is_on { self.off() } else { self.on() }
    }

    pub fn blink<D: DelayNs>(
        &mut self,
        delay: &mut D,
        duration_ms: u32,
    ) -> Result<(), P::Error> {
        self.on()?;
        delay.delay_ms(duration_ms);
        self.off()?;
        delay.delay_ms(duration_ms);
        Ok(())
    }
}
```

---

## 6. GPIO, SPI, I2C, UART

### GPIO: 블링키 예제 (STM32F4)

```rust
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;
use stm32f4xx_hal::{pac, prelude::*, timer::MonoTimerUs};

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let cp = cortex_m::Peripherals::take().unwrap();

    // 클럭 구성
    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.sysclk(84.MHz()).freeze();

    // GPIO 핀을 출력으로 구성
    let gpioa = dp.GPIOA.split();
    let mut led = gpioa.pa5.into_push_pull_output();

    // 딜레이 프로바이더 생성
    let mut delay = cp.SYST.delay(&clocks);

    loop {
        led.set_high();
        delay.delay_ms(500u32);
        led.set_low();
        delay.delay_ms(500u32);
    }
}
```

### SPI 통신

```rust
use embedded_hal::spi::SpiDevice;

/// SPI로 레지스터 읽기
fn read_register<SPI: SpiDevice>(spi: &mut SPI, reg: u8) -> Result<u8, SPI::Error> {
    let mut buf = [reg | 0x80, 0x00];  // 읽기 비트 + 레지스터 주소
    spi.transfer_in_place(&mut buf)?;
    Ok(buf[1])
}

/// SPI로 레지스터 쓰기
fn write_register<SPI: SpiDevice>(
    spi: &mut SPI,
    reg: u8,
    value: u8,
) -> Result<(), SPI::Error> {
    spi.write(&[reg & 0x7F, value])?;  // 쓰기 비트 + 레지스터 + 값
    Ok(())
}
```

### I2C: 온도 센서 읽기

```rust
use embedded_hal::i2c::I2c;

const SENSOR_ADDR: u8 = 0x48;  // TMP102 주소

fn read_temperature<I: I2c>(i2c: &mut I) -> Result<f32, I::Error> {
    let mut buf = [0u8; 2];
    i2c.write_read(SENSOR_ADDR, &[0x00], &mut buf)?;

    // 원시 바이트를 온도로 변환
    let raw = ((buf[0] as i16) << 4) | ((buf[1] as i16) >> 4);
    Ok(raw as f32 * 0.0625)
}

fn configure_sensor<I: I2c>(i2c: &mut I) -> Result<(), I::Error> {
    // 구성 레지스터 쓰기
    i2c.write(SENSOR_ADDR, &[0x01, 0x60, 0xA0])?;
    Ok(())
}
```

### UART

```rust
use embedded_hal::serial::{Read, Write};
use core::fmt::Write as FmtWrite;

fn uart_echo<S: Read<u8> + Write<u8>>(serial: &mut S) {
    loop {
        if let Ok(byte) = serial.read() {
            let _ = serial.write(byte);
        }
    }
}

// UART로 포매팅된 텍스트 쓰기
struct UartWriter<S> {
    serial: S,
}

impl<S: Write<u8>> FmtWrite for UartWriter<S> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for byte in s.bytes() {
            self.serial.write(byte).map_err(|_| core::fmt::Error)?;
        }
        Ok(())
    }
}
```

---

## 7. 메모리 매핑 I/O

메모리 매핑 I/O를 통한 직접 하드웨어 레지스터 접근:

```rust
use core::ptr::{read_volatile, write_volatile};

// STM32F4 GPIO 레지스터 블록 (간략화)
const GPIOA_BASE: usize = 0x4002_0000;
const GPIOA_MODER: *mut u32 = GPIOA_BASE as *mut u32;           // 모드 레지스터
const GPIOA_ODR: *mut u32 = (GPIOA_BASE + 0x14) as *mut u32;   // 출력 데이터
const GPIOA_BSRR: *mut u32 = (GPIOA_BASE + 0x18) as *mut u32;  // 비트 세트/리셋

unsafe fn configure_pa5_output() {
    // 현재 MODER 값 읽기
    let mut moder = read_volatile(GPIOA_MODER);
    // 비트 10:11 (PA5 모드) 지우고 01 (출력)로 설정
    moder &= !(0b11 << 10);
    moder |= 0b01 << 10;
    write_volatile(GPIOA_MODER, moder);
}

unsafe fn set_pa5_high() {
    // BSRR: 비트 5에 1을 써서 PA5를 HIGH로 설정 (원자적)
    write_volatile(GPIOA_BSRR, 1 << 5);
}

unsafe fn set_pa5_low() {
    // BSRR: 비트 21 (16 + 5)에 1을 써서 PA5 리셋 (원자적)
    write_volatile(GPIOA_BSRR, 1 << (5 + 16));
}
```

### volatile-register 크레이트 사용

```rust
use volatile_register::{RO, RW, WO};

#[repr(C)]
struct GpioRegisters {
    moder: RW<u32>,     // 모드 레지스터
    otyper: RW<u32>,    // 출력 타입
    ospeedr: RW<u32>,   // 출력 속도
    pupdr: RW<u32>,     // 풀업/풀다운
    idr: RO<u32>,       // 입력 데이터 (읽기 전용)
    odr: RW<u32>,       // 출력 데이터
    bsrr: WO<u32>,      // 비트 세트/리셋 (쓰기 전용)
    lckr: RW<u32>,      // 잠금
    afrl: RW<u32>,      // 대체 기능 하위
    afrh: RW<u32>,      // 대체 기능 상위
}

fn gpio_example() {
    let gpio = unsafe { &*(0x4002_0000 as *const GpioRegisters) };

    // 입력 읽기
    let pin_state = gpio.idr.read() & (1 << 0);

    // 출력 수정 (읽기-수정-쓰기)
    unsafe {
        gpio.moder.modify(|v| (v & !(0b11 << 10)) | (0b01 << 10));
    }

    // 원자적 세트 (쓰기 전용 레지스터)
    unsafe { gpio.bsrr.write(1 << 5); }
}
```

---

## 8. 인터럽트와 예외 처리

```rust
#![no_std]
#![no_main]

use core::cell::RefCell;
use cortex_m::interrupt::Mutex;
use cortex_m_rt::{entry, exception};
use stm32f4xx_hal::{pac, prelude::*, timer::{Event, CounterUs}};
use panic_halt as _;

// 메인과 인터럽트 핸들러 간 공유 상태
static TIMER: Mutex<RefCell<Option<CounterUs<pac::TIM2>>>> =
    Mutex::new(RefCell::new(None));
static COUNTER: Mutex<RefCell<u32>> = Mutex::new(RefCell::new(0));

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.sysclk(84.MHz()).freeze();

    // 타이머 인터럽트 구성
    let mut timer = dp.TIM2.counter(&clocks);
    timer.start(1.secs()).unwrap();
    timer.listen(Event::Update);

    // 인터럽트 핸들러를 위해 전역에 타이머 저장
    cortex_m::interrupt::free(|cs| {
        TIMER.borrow(cs).replace(Some(timer));
    });

    // NVIC에서 TIM2 인터럽트 활성화
    unsafe {
        cortex_m::peripheral::NVIC::unmask(pac::Interrupt::TIM2);
    }

    loop {
        cortex_m::asm::wfi();  // 인터럽트까지 슬립
        let count = cortex_m::interrupt::free(|cs| {
            *COUNTER.borrow(cs).borrow()
        });
        // count는 인터럽트를 통해 매 초마다 증가
    }
}

#[cortex_m_rt::interrupt]
fn TIM2() {
    cortex_m::interrupt::free(|cs| {
        if let Some(ref mut timer) = *TIMER.borrow(cs).borrow_mut() {
            timer.clear_interrupt(Event::Update);
        }
        let mut counter = COUNTER.borrow(cs).borrow_mut();
        *counter += 1;
    });
}
```

---

## 9. RTIC 프레임워크

RTIC(Real-Time Interrupt-driven Concurrency)는 컴파일 타임 보장이 있는 동시 임베디드 애플리케이션 구축 프레임워크입니다:

```rust
#![no_std]
#![no_main]

use panic_halt as _;
use rtic::app;
use stm32f4xx_hal::{pac, prelude::*, gpio::{Output, PushPull, PA5}};

#[app(device = stm32f4xx_hal::pac, peripherals = true, dispatchers = [SPI1])]
mod app {
    use super::*;
    use systick_monotonic::Systick;

    #[monotonic(binds = SysTick, default = true)]
    type Mono = Systick<1000>;  // 1kHz 틱 속도

    #[shared]
    struct Shared {
        counter: u32,
    }

    #[local]
    struct Local {
        led: PA5<Output<PushPull>>,
    }

    #[init]
    fn init(cx: init::Context) -> (Shared, Local, init::Monotonics) {
        let dp = cx.device;
        let rcc = dp.RCC.constrain();
        let clocks = rcc.cfgr.sysclk(84.MHz()).freeze();

        let gpioa = dp.GPIOA.split();
        let led = gpioa.pa5.into_push_pull_output();

        let mono = Systick::new(cx.core.SYST, clocks.sysclk().raw());

        // 첫 번째 블링크 스케줄
        blink::spawn_after(1.secs()).ok();

        (
            Shared { counter: 0 },
            Local { led },
            init::Monotonics(mono),
        )
    }

    // 태스크: 매 초마다 LED 토글
    #[task(local = [led], shared = [counter])]
    fn blink(mut cx: blink::Context) {
        cx.local.led.toggle();

        cx.shared.counter.lock(|counter| {
            *counter += 1;
        });

        // 재스케줄
        blink::spawn_after(1.secs()).ok();
    }

    // 하드웨어 태스크: 버튼 누름 처리
    #[task(binds = EXTI0, shared = [counter])]
    fn button_press(mut cx: button_press::Context) {
        cx.shared.counter.lock(|counter| {
            *counter = 0;  // 버튼 누름 시 리셋
        });
    }
}
```

---

## 10. probe-rs: 플래시와 디버그

probe-rs는 임베디드 Rust를 플래싱하고 디버깅하기 위한 현대적인 도구입니다:

```bash
# probe-rs 설치
cargo install probe-rs-tools

# 연결된 프로브 목록
probe-rs list

# 플래시 및 실행 (RTT 출력 포함)
cargo run --release  # .cargo/config.toml의 runner 사용

# 또는 직접:
probe-rs run --chip STM32F411CEUx target/thumbv7em-none-eabihf/release/blinky

# GDB로 디버그
probe-rs debug --chip STM32F411CEUx target/thumbv7em-none-eabihf/release/blinky
```

### RTT(실시간 전송) 로깅

```rust
use rtt_target::{rprintln, rtt_init_print};

#[entry]
fn main() -> ! {
    rtt_init_print!();

    rprintln!("임베디드 Rust에서 안녕하세요!");

    let mut counter = 0u32;
    loop {
        rprintln!("카운터: {counter}");
        counter += 1;
        cortex_m::asm::delay(8_000_000);  // 8MHz에서 약 1초
    }
}
```

### defmt: 효율적인 로깅

```rust
use defmt::*;
use defmt_rtt as _;

#[entry]
fn main() -> ! {
    info!("애플리케이션 시작");

    let sensor_value: u16 = 1023;
    debug!("센서 값: {}", sensor_value);

    if sensor_value > 900 {
        warn!("센서 값 높음: {}", sensor_value);
    }

    loop {
        trace!("메인 루프 반복");
        cortex_m::asm::wfi();
    }
}
```

---

## 11. 임베디드 패턴

### 주변장치에 대한 싱글턴 패턴

```rust
use core::sync::atomic::{AtomicBool, Ordering};

static UART_TAKEN: AtomicBool = AtomicBool::new(false);

struct Uart {
    // UART 주변장치 필드
}

impl Uart {
    /// UART 주변장치 가져오기 — 한 번만 호출 가능
    fn take() -> Option<Self> {
        if UART_TAKEN.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            Some(Uart { /* ... */ })
        } else {
            None  // 이미 가져감
        }
    }
}
```

### 프로토콜 처리를 위한 상태 머신

```rust
enum ProtocolState {
    Idle,
    ReceivingHeader { bytes_received: usize },
    ReceivingPayload { header: [u8; 4], bytes_received: usize },
    Complete { header: [u8; 4], payload: [u8; 256] },
    Error,
}

struct ProtocolHandler {
    state: ProtocolState,
    buffer: [u8; 256],
}

impl ProtocolHandler {
    fn feed_byte(&mut self, byte: u8) {
        self.state = match core::mem::replace(&mut self.state, ProtocolState::Error) {
            ProtocolState::Idle if byte == 0xAA => {
                ProtocolState::ReceivingHeader { bytes_received: 0 }
            }
            ProtocolState::ReceivingHeader { bytes_received } => {
                self.buffer[bytes_received] = byte;
                if bytes_received + 1 >= 4 {
                    let mut header = [0u8; 4];
                    header.copy_from_slice(&self.buffer[..4]);
                    ProtocolState::ReceivingPayload { header, bytes_received: 0 }
                } else {
                    ProtocolState::ReceivingHeader { bytes_received: bytes_received + 1 }
                }
            }
            // ... 더 많은 상태들
            _ => ProtocolState::Idle,
        };
    }
}
```

---

## 12. 연습문제

1. **no_std 라이브러리**: `core`만 사용하여 링 버퍼(`CircularBuffer<T, const N: usize>`)를 구현하세요. `push`, `pop`, `is_full`, `is_empty`, `Iterator` 구현을 포함하세요.

2. **이식 가능한 센서 드라이버**: I2C 트레이트를 사용하는 BMP280 온도/압력 센서용 embedded-hal 드라이버를 작성하세요. 드라이버는 `embedded_hal::i2c::I2c`를 구현하는 모든 칩에서 작동해야 합니다.

3. **LED 패턴 생성기**: embedded-hal의 `OutputPin`과 `DelayNs` 트레이트를 사용하여 LED 패턴(블링크, 브리드, 모스 부호, 체이스)을 재생하는 라이브러리를 만드세요. 모의 구현으로 테스트하세요.

4. **상태 머신 프로토콜**: UART 프로토콜 파서를 상태 머신으로 구현하세요: 싱크 바이트 → 헤더(4바이트) → 페이로드(가변) → CRC → 완료. 모든 오류 케이스를 처리하세요.

5. **메모리 매핑 레지스터 DSL**: 다음과 같은 설명에서 안전한 레지스터 접근 타입을 생성하는 매크로를 만드세요: `register_block!(GPIOA at 0x4002_0000 { MODER: RW @ 0x00, IDR: RO @ 0x10, ODR: RW @ 0x14, BSRR: WO @ 0x18 })`.

---

## 참고 자료

- [The Embedded Rust Book](https://docs.rust-embedded.org/book/)
- [Discovery Book](https://docs.rust-embedded.org/discovery/)
- [embedded-hal documentation](https://docs.rs/embedded-hal/latest/embedded_hal/)
- [RTIC documentation](https://rtic.rs/)
- [probe-rs documentation](https://probe.rs/)
- [defmt documentation](https://defmt.ferrous-systems.com/)

---

**이전**: [WebAssembly](./09_WebAssembly.md) | **다음**: [네트워크 프로그래밍](./11_Network_Programming.md)
