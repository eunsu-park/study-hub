# 26. Embedded Rust

**Previous**: [WebAssembly](./09_WebAssembly.md) | **Next**: [Network Programming](./11_Network_Programming.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Write `no_std` Rust code for bare-metal environments without the standard library
2. Use `embedded-hal` traits for portable driver development
3. Build real-time applications with the RTIC (Real-Time Interrupt-driven Concurrency) framework
4. Perform memory-mapped I/O for direct hardware register access
5. Debug and flash embedded devices using `probe-rs`

---

Rust is increasingly adopted for embedded systems. Its ownership model prevents entire classes of bugs — buffer overflows, use-after-free, data races — that plague C firmware. With zero-cost abstractions and no runtime overhead, Rust produces binaries comparable to hand-written C while providing compile-time safety guarantees.

## Table of Contents
1. [Why Rust for Embedded?](#1-why-rust-for-embedded)
2. [no_std Fundamentals](#2-no_std-fundamentals)
3. [Target Architecture Setup](#3-target-architecture-setup)
4. [Memory Layout and Linker Scripts](#4-memory-layout-and-linker-scripts)
5. [embedded-hal: The Hardware Abstraction Layer](#5-embedded-hal-the-hardware-abstraction-layer)
6. [GPIO, SPI, I2C, and UART](#6-gpio-spi-i2c-and-uart)
7. [Memory-Mapped I/O](#7-memory-mapped-io)
8. [Interrupts and Exception Handling](#8-interrupts-and-exception-handling)
9. [RTIC Framework](#9-rtic-framework)
10. [probe-rs: Flash and Debug](#10-probe-rs-flash-and-debug)
11. [Embedded Patterns](#11-embedded-patterns)
12. [Exercises](#12-exercises)

---

## 1. Why Rust for Embedded?

### Comparison with C

| Aspect | C | Rust |
|--------|---|------|
| Memory safety | Manual | Compile-time guaranteed |
| Data race prevention | None | Ownership system |
| Buffer overflows | Common | Impossible (safe code) |
| Null dereferences | Common | No null (`Option`) |
| Binary size | Small | Comparable |
| Runtime overhead | None | None |
| Ecosystem maturity | Decades | Growing rapidly |

### The Embedded Rust Ecosystem

```
┌─────────────────────────────────────────┐
│  Your Application                       │
├─────────────────────────────────────────┤
│  BSP (Board Support Package)            │
│  e.g., stm32f4xx-hal, rp-pico          │
├─────────────────────────────────────────┤
│  HAL (Hardware Abstraction Layer)       │
│  e.g., stm32f4xx-hal                   │
├─────────────────────────────────────────┤
│  PAC (Peripheral Access Crate)          │
│  e.g., stm32f4 (auto-generated from SVD)│
├─────────────────────────────────────────┤
│  embedded-hal traits                    │
├─────────────────────────────────────────┤
│  cortex-m / cortex-m-rt                 │
│  (Core support & runtime)               │
└─────────────────────────────────────────┘
```

---

## 2. no_std Fundamentals

In embedded systems, there is no operating system — no heap, no threads, no filesystem. `#![no_std]` tells the compiler not to link the standard library:

```rust
#![no_std]
#![no_main]

use core::panic::PanicInfo;

// With no_std, you must define your own panic handler
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}  // Halt on panic
}

// No main function — the entry point is defined by the runtime crate
```

### What's Available in no_std

```rust
#![no_std]

// core:: is always available (no allocation, no OS)
use core::fmt;           // Formatting (write!, format_args!)
use core::ops;           // Operator overloading
use core::iter;          // Iterator trait and adapters
use core::option::Option;
use core::result::Result;
use core::cell::Cell;    // Interior mutability
use core::ptr;           // Raw pointer operations
use core::mem;           // Memory operations
use core::sync::atomic;  // Atomic operations

// alloc:: is available if you have a global allocator
// extern crate alloc;
// use alloc::vec::Vec;
// use alloc::string::String;
// use alloc::boxed::Box;
```

### no_std with alloc

If your embedded target has a heap (many do):

```rust
#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use alloc::string::String;
use alloc::format;

// You need to define a global allocator
use embedded_alloc::LlffHeap as Heap;

#[global_allocator]
static HEAP: Heap = Heap::empty();

fn init_heap() {
    // Initialize the heap with a static buffer
    const HEAP_SIZE: usize = 4096;
    static mut HEAP_MEM: [u8; HEAP_SIZE] = [0; HEAP_SIZE];
    unsafe {
        HEAP.init(HEAP_MEM.as_ptr() as usize, HEAP_SIZE);
    }
}
```

---

## 3. Target Architecture Setup

### Common Embedded Targets

```bash
# ARM Cortex-M0/M0+ (no hardware FPU)
rustup target add thumbv6m-none-eabi

# ARM Cortex-M3
rustup target add thumbv7m-none-eabi

# ARM Cortex-M4/M7 (with hardware FPU)
rustup target add thumbv7em-none-eabihf

# RISC-V 32-bit
rustup target add riscv32imac-unknown-none-elf

# RISC-V 64-bit
rustup target add riscv64gc-unknown-none-elf
```

### Project Configuration

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

# HAL for your specific chip
stm32f4xx-hal = { version = "0.21", features = ["stm32f411"] }

[profile.release]
opt-level = "s"
debug = true       # Keep debug info for debugging
lto = true
```

---

## 4. Memory Layout and Linker Scripts

Embedded systems need explicit memory layout:

```
/* memory.x — linker script */
MEMORY
{
  FLASH : ORIGIN = 0x08000000, LENGTH = 512K
  RAM   : ORIGIN = 0x20000000, LENGTH = 128K
}

/* Optional: place specific sections */
SECTIONS
{
  .data : ALIGN(4)
  {
    *(.data .data.*);
  } > RAM AT > FLASH
}
```

```rust
// The cortex-m-rt crate handles the vector table and startup code:
// 1. Copies .data from FLASH to RAM
// 2. Zeros .bss
// 3. Calls main()

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;

#[entry]
fn main() -> ! {
    // Entry point — must never return (-> !)
    loop {
        cortex_m::asm::wfi();  // Wait for interrupt (low power)
    }
}
```

### Static Variables

```rust
use core::cell::RefCell;
use cortex_m::interrupt::Mutex;

// Global mutable state in embedded requires careful handling
// Use Mutex<RefCell<T>> with critical sections
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

## 5. embedded-hal: The Hardware Abstraction Layer

`embedded-hal` defines traits for common hardware peripherals. Drivers written against these traits work on any chip:

```rust
// embedded-hal traits (simplified)
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

### Writing a Portable Driver

```rust
use embedded_hal::digital::OutputPin;
use embedded_hal::delay::DelayNs;

/// A driver for a generic LED — works on any chip
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

## 6. GPIO, SPI, I2C, and UART

### GPIO: Blinky Example (STM32F4)

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

    // Configure clocks
    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.sysclk(84.MHz()).freeze();

    // Configure GPIO pin as output
    let gpioa = dp.GPIOA.split();
    let mut led = gpioa.pa5.into_push_pull_output();

    // Create a delay provider
    let mut delay = cp.SYST.delay(&clocks);

    loop {
        led.set_high();
        delay.delay_ms(500u32);
        led.set_low();
        delay.delay_ms(500u32);
    }
}
```

### SPI Communication

```rust
use embedded_hal::spi::SpiDevice;

/// Read a register over SPI
fn read_register<SPI: SpiDevice>(spi: &mut SPI, reg: u8) -> Result<u8, SPI::Error> {
    let mut buf = [reg | 0x80, 0x00];  // Read bit + register address
    spi.transfer_in_place(&mut buf)?;
    Ok(buf[1])
}

/// Write a register over SPI
fn write_register<SPI: SpiDevice>(
    spi: &mut SPI,
    reg: u8,
    value: u8,
) -> Result<(), SPI::Error> {
    spi.write(&[reg & 0x7F, value])?;  // Write bit + register + value
    Ok(())
}
```

### I2C: Reading a Temperature Sensor

```rust
use embedded_hal::i2c::I2c;

const SENSOR_ADDR: u8 = 0x48;  // TMP102 address

fn read_temperature<I: I2c>(i2c: &mut I) -> Result<f32, I::Error> {
    let mut buf = [0u8; 2];
    i2c.write_read(SENSOR_ADDR, &[0x00], &mut buf)?;

    // Convert raw bytes to temperature
    let raw = ((buf[0] as i16) << 4) | ((buf[1] as i16) >> 4);
    Ok(raw as f32 * 0.0625)
}

fn configure_sensor<I: I2c>(i2c: &mut I) -> Result<(), I::Error> {
    // Write configuration register
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

// Write formatted text over UART
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

## 7. Memory-Mapped I/O

Direct hardware register access via memory-mapped I/O:

```rust
use core::ptr::{read_volatile, write_volatile};

// STM32F4 GPIO register block (simplified)
const GPIOA_BASE: usize = 0x4002_0000;
const GPIOA_MODER: *mut u32 = GPIOA_BASE as *mut u32;           // Mode register
const GPIOA_ODR: *mut u32 = (GPIOA_BASE + 0x14) as *mut u32;   // Output data
const GPIOA_BSRR: *mut u32 = (GPIOA_BASE + 0x18) as *mut u32;  // Bit set/reset

unsafe fn configure_pa5_output() {
    // Read current MODER value
    let mut moder = read_volatile(GPIOA_MODER);
    // Clear bits 10:11 (PA5 mode) and set to 01 (output)
    moder &= !(0b11 << 10);
    moder |= 0b01 << 10;
    write_volatile(GPIOA_MODER, moder);
}

unsafe fn set_pa5_high() {
    // BSRR: write 1 to bit 5 to set PA5 high (atomic)
    write_volatile(GPIOA_BSRR, 1 << 5);
}

unsafe fn set_pa5_low() {
    // BSRR: write 1 to bit 21 (16 + 5) to reset PA5 (atomic)
    write_volatile(GPIOA_BSRR, 1 << (5 + 16));
}
```

### Using volatile-register Crate

```rust
use volatile_register::{RO, RW, WO};

#[repr(C)]
struct GpioRegisters {
    moder: RW<u32>,     // Mode register
    otyper: RW<u32>,    // Output type
    ospeedr: RW<u32>,   // Output speed
    pupdr: RW<u32>,     // Pull-up/pull-down
    idr: RO<u32>,       // Input data (read-only)
    odr: RW<u32>,       // Output data
    bsrr: WO<u32>,      // Bit set/reset (write-only)
    lckr: RW<u32>,      // Lock
    afrl: RW<u32>,      // Alternate function low
    afrh: RW<u32>,      // Alternate function high
}

fn gpio_example() {
    let gpio = unsafe { &*(0x4002_0000 as *const GpioRegisters) };

    // Read input
    let pin_state = gpio.idr.read() & (1 << 0);

    // Modify output (read-modify-write)
    unsafe {
        gpio.moder.modify(|v| (v & !(0b11 << 10)) | (0b01 << 10));
    }

    // Atomic set (write-only register)
    unsafe { gpio.bsrr.write(1 << 5); }
}
```

---

## 8. Interrupts and Exception Handling

```rust
#![no_std]
#![no_main]

use core::cell::RefCell;
use cortex_m::interrupt::Mutex;
use cortex_m_rt::{entry, exception};
use stm32f4xx_hal::{pac, prelude::*, timer::{Event, CounterUs}};
use panic_halt as _;

// Shared state between main and interrupt handler
static TIMER: Mutex<RefCell<Option<CounterUs<pac::TIM2>>>> =
    Mutex::new(RefCell::new(None));
static COUNTER: Mutex<RefCell<u32>> = Mutex::new(RefCell::new(0));

#[entry]
fn main() -> ! {
    let dp = pac::Peripherals::take().unwrap();
    let rcc = dp.RCC.constrain();
    let clocks = rcc.cfgr.sysclk(84.MHz()).freeze();

    // Configure timer interrupt
    let mut timer = dp.TIM2.counter(&clocks);
    timer.start(1.secs()).unwrap();
    timer.listen(Event::Update);

    // Store timer in global for the interrupt handler
    cortex_m::interrupt::free(|cs| {
        TIMER.borrow(cs).replace(Some(timer));
    });

    // Enable the TIM2 interrupt in the NVIC
    unsafe {
        cortex_m::peripheral::NVIC::unmask(pac::Interrupt::TIM2);
    }

    loop {
        cortex_m::asm::wfi();  // Sleep until interrupt
        let count = cortex_m::interrupt::free(|cs| {
            *COUNTER.borrow(cs).borrow()
        });
        // count increments every second via the interrupt
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

## 9. RTIC Framework

RTIC (Real-Time Interrupt-driven Concurrency) provides a framework for building concurrent embedded applications with compile-time guarantees:

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
    type Mono = Systick<1000>;  // 1kHz tick rate

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

        // Schedule the first blink
        blink::spawn_after(1.secs()).ok();

        (
            Shared { counter: 0 },
            Local { led },
            init::Monotonics(mono),
        )
    }

    // Task: toggle LED every second
    #[task(local = [led], shared = [counter])]
    fn blink(mut cx: blink::Context) {
        cx.local.led.toggle();

        cx.shared.counter.lock(|counter| {
            *counter += 1;
        });

        // Reschedule
        blink::spawn_after(1.secs()).ok();
    }

    // Hardware task: handle button press
    #[task(binds = EXTI0, shared = [counter])]
    fn button_press(mut cx: button_press::Context) {
        cx.shared.counter.lock(|counter| {
            *counter = 0;  // Reset on button press
        });
    }
}
```

---

## 10. probe-rs: Flash and Debug

probe-rs is a modern tool for flashing and debugging embedded Rust:

```bash
# Install probe-rs
cargo install probe-rs-tools

# List connected probes
probe-rs list

# Flash and run (with RTT output)
cargo run --release  # Uses runner from .cargo/config.toml

# Or directly:
probe-rs run --chip STM32F411CEUx target/thumbv7em-none-eabihf/release/blinky

# Debug with GDB
probe-rs debug --chip STM32F411CEUx target/thumbv7em-none-eabihf/release/blinky
```

### RTT (Real-Time Transfer) Logging

```rust
use rtt_target::{rprintln, rtt_init_print};

#[entry]
fn main() -> ! {
    rtt_init_print!();

    rprintln!("Hello from embedded Rust!");

    let mut counter = 0u32;
    loop {
        rprintln!("Counter: {counter}");
        counter += 1;
        cortex_m::asm::delay(8_000_000);  // ~1 second at 8MHz
    }
}
```

### defmt: Efficient Logging

```rust
use defmt::*;
use defmt_rtt as _;

#[entry]
fn main() -> ! {
    info!("Starting application");

    let sensor_value: u16 = 1023;
    debug!("Sensor reading: {}", sensor_value);

    if sensor_value > 900 {
        warn!("Sensor value high: {}", sensor_value);
    }

    loop {
        trace!("Main loop iteration");
        cortex_m::asm::wfi();
    }
}
```

---

## 11. Embedded Patterns

### Singleton Pattern for Peripherals

```rust
use core::sync::atomic::{AtomicBool, Ordering};

static UART_TAKEN: AtomicBool = AtomicBool::new(false);

struct Uart {
    // UART peripheral fields
}

impl Uart {
    /// Take the UART peripheral — can only be called once
    fn take() -> Option<Self> {
        if UART_TAKEN.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            Some(Uart { /* ... */ })
        } else {
            None  // Already taken
        }
    }
}
```

### State Machine for Protocol Handling

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
            // ... more states
            _ => ProtocolState::Idle,
        };
    }
}
```

---

## 12. Exercises

1. **no_std library**: Create a `no_std` Rust library that implements a ring buffer (`CircularBuffer<T, const N: usize>`) using only `core`. Include `push`, `pop`, `is_full`, `is_empty`, and `Iterator` implementation.

2. **Portable sensor driver**: Write an embedded-hal driver for a BMP280 temperature/pressure sensor using the I2C trait. The driver should work on any chip that implements `embedded_hal::i2c::I2c`.

3. **LED pattern generator**: Using embedded-hal's `OutputPin` and `DelayNs` traits, create a library that plays LED patterns (blink, breathe, morse code, chase). Test against a mock implementation.

4. **State machine protocol**: Implement a UART protocol parser as a state machine that handles: sync byte → header (4 bytes) → payload (variable) → CRC → done. Handle all error cases.

5. **Memory-mapped register DSL**: Create a macro that generates safe register access types from a description like: `register_block!(GPIOA at 0x4002_0000 { MODER: RW @ 0x00, IDR: RO @ 0x10, ODR: RW @ 0x14, BSRR: WO @ 0x18 })`.

---

## References

- [The Embedded Rust Book](https://docs.rust-embedded.org/book/)
- [Discovery Book](https://docs.rust-embedded.org/discovery/)
- [embedded-hal documentation](https://docs.rs/embedded-hal/latest/embedded_hal/)
- [RTIC documentation](https://rtic.rs/)
- [probe-rs documentation](https://probe.rs/)
- [defmt documentation](https://defmt.ferrous-systems.com/)

---

**Previous**: [WebAssembly](./09_WebAssembly.md) | **Next**: [Network Programming](./11_Network_Programming.md)
