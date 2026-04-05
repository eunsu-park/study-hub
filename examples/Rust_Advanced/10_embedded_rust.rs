// 10_embedded_rust.rs — Embedded Rust patterns (simulated, no hardware required)
//
// Run: rustc 10_embedded_rust.rs && ./10_embedded_rust
//
// Demonstrates no_std patterns, hardware abstraction, and embedded idioms
// without requiring actual embedded hardware.

use std::fmt;

fn main() {
    println!("=== Register-level Access ===");
    register_demo();

    println!("\n=== GPIO Abstraction ===");
    gpio_demo();

    println!("\n=== Embedded HAL Pattern ===");
    hal_demo();

    println!("\n=== Ring Buffer (no-alloc) ===");
    ring_buffer_demo();

    println!("\n=== State Machine (typestate) ===");
    typestate_demo();
}

// --- Memory-mapped register simulation ---

struct Register {
    name: &'static str,
    value: u32,
}

impl Register {
    fn new(name: &'static str) -> Self {
        Register { name, value: 0 }
    }

    fn read(&self) -> u32 {
        self.value
    }

    fn write(&mut self, val: u32) {
        println!("  [{:>10}] write: {:#010x}", self.name, val);
        self.value = val;
    }

    fn set_bits(&mut self, mask: u32) {
        self.value |= mask;
        println!("  [{:>10}] set bits: {:#010x} → {:#010x}", self.name, mask, self.value);
    }

    fn clear_bits(&mut self, mask: u32) {
        self.value &= !mask;
        println!("  [{:>10}] clear bits: {:#010x} → {:#010x}", self.name, mask, self.value);
    }
}

fn register_demo() {
    let mut gpio_moder = Register::new("GPIOA_MODE");
    let mut gpio_odr = Register::new("GPIOA_ODR");

    // Configure pin 5 as output (bits 10-11 = 01)
    gpio_moder.set_bits(1 << 10);
    gpio_moder.clear_bits(1 << 11);

    // Set pin 5 high
    gpio_odr.set_bits(1 << 5);
    println!("  Pin 5 state: {}", if gpio_odr.read() & (1 << 5) != 0 { "HIGH" } else { "LOW" });

    // Clear pin 5
    gpio_odr.clear_bits(1 << 5);
    println!("  Pin 5 state: {}", if gpio_odr.read() & (1 << 5) != 0 { "HIGH" } else { "LOW" });
}

// --- GPIO abstraction ---

#[derive(Debug, Clone, Copy)]
enum PinState { High, Low }

impl fmt::Display for PinState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { PinState::High => write!(f, "HIGH"), PinState::Low => write!(f, "LOW") }
    }
}

struct OutputPin {
    pin: u8,
    state: PinState,
}

impl OutputPin {
    fn new(pin: u8) -> Self {
        println!("  Configured pin {pin} as OUTPUT");
        OutputPin { pin, state: PinState::Low }
    }

    fn set_high(&mut self) {
        self.state = PinState::High;
        println!("  Pin {} → {}", self.pin, self.state);
    }

    fn set_low(&mut self) {
        self.state = PinState::Low;
        println!("  Pin {} → {}", self.pin, self.state);
    }

    fn toggle(&mut self) {
        match self.state {
            PinState::High => self.set_low(),
            PinState::Low => self.set_high(),
        }
    }
}

struct InputPin {
    pin: u8,
    state: PinState,
}

impl InputPin {
    fn new(pin: u8, initial: PinState) -> Self {
        println!("  Configured pin {pin} as INPUT");
        InputPin { pin, state: initial }
    }

    fn is_high(&self) -> bool {
        matches!(self.state, PinState::High)
    }

    fn simulate_change(&mut self, state: PinState) {
        self.state = state;
        println!("  [SIM] Pin {} external → {}", self.pin, state);
    }
}

fn gpio_demo() {
    let mut led = OutputPin::new(13);
    let mut button = InputPin::new(2, PinState::Low);

    // Simulate button press → toggle LED
    button.simulate_change(PinState::High);
    if button.is_high() {
        led.toggle();
    }

    button.simulate_change(PinState::Low);
    button.simulate_change(PinState::High);
    if button.is_high() {
        led.toggle();
    }
}

// --- Embedded HAL trait pattern ---

trait DigitalWrite {
    fn set_high(&mut self);
    fn set_low(&mut self);
}

trait DelayMs {
    fn delay_ms(&mut self, ms: u32);
}

struct SimulatedDelay;

impl DelayMs for SimulatedDelay {
    fn delay_ms(&mut self, ms: u32) {
        println!("  [delay {ms}ms]");
    }
}

struct Led {
    pin: OutputPin,
}

impl Led {
    fn new(pin: u8) -> Self {
        Led { pin: OutputPin::new(pin) }
    }

    fn blink(&mut self, delay: &mut impl DelayMs, times: u32) {
        for i in 1..=times {
            println!("  Blink {i}/{times}");
            self.pin.set_high();
            delay.delay_ms(500);
            self.pin.set_low();
            delay.delay_ms(500);
        }
    }
}

fn hal_demo() {
    let mut led = Led::new(13);
    let mut delay = SimulatedDelay;
    led.blink(&mut delay, 3);
}

// --- Ring buffer (fixed-size, no heap allocation) ---

struct RingBuffer<const N: usize> {
    buffer: [u8; N],
    head: usize,
    tail: usize,
    count: usize,
}

impl<const N: usize> RingBuffer<N> {
    fn new() -> Self {
        RingBuffer { buffer: [0; N], head: 0, tail: 0, count: 0 }
    }

    fn push(&mut self, value: u8) -> bool {
        if self.count == N { return false; }
        self.buffer[self.tail] = value;
        self.tail = (self.tail + 1) % N;
        self.count += 1;
        true
    }

    fn pop(&mut self) -> Option<u8> {
        if self.count == 0 { return None; }
        let value = self.buffer[self.head];
        self.head = (self.head + 1) % N;
        self.count -= 1;
        Some(value)
    }

    fn len(&self) -> usize { self.count }
    fn is_full(&self) -> bool { self.count == N }
}

fn ring_buffer_demo() {
    let mut buf = RingBuffer::<4>::new();

    for b in b"Hello" {
        if buf.push(*b) {
            println!("  Pushed '{}', len={}", *b as char, buf.len());
        } else {
            println!("  Buffer full, dropped '{}'", *b as char);
        }
    }

    while let Some(b) = buf.pop() {
        println!("  Popped '{}', len={}", b as char, buf.len());
    }
}

// --- Typestate pattern (compile-time state machine) ---

struct Uninit;
struct Ready;
struct Running;

struct Peripheral<State> {
    name: String,
    _state: std::marker::PhantomData<State>,
}

impl Peripheral<Uninit> {
    fn new(name: &str) -> Self {
        println!("  Created peripheral: {name} (uninitialized)");
        Peripheral { name: name.to_string(), _state: std::marker::PhantomData }
    }

    fn init(self) -> Peripheral<Ready> {
        println!("  {} → Ready", self.name);
        Peripheral { name: self.name, _state: std::marker::PhantomData }
    }
}

impl Peripheral<Ready> {
    fn start(self) -> Peripheral<Running> {
        println!("  {} → Running", self.name);
        Peripheral { name: self.name, _state: std::marker::PhantomData }
    }
}

impl Peripheral<Running> {
    fn read(&self) -> u32 {
        println!("  {} read → 42", self.name);
        42
    }

    fn stop(self) -> Peripheral<Ready> {
        println!("  {} → Stopped (Ready)", self.name);
        Peripheral { name: self.name, _state: std::marker::PhantomData }
    }
}

fn typestate_demo() {
    let adc = Peripheral::<Uninit>::new("ADC1");
    let adc = adc.init();      // Uninit → Ready
    let adc = adc.start();     // Ready → Running
    let val = adc.read();      // Can only read when Running
    println!("  Value: {val}");
    let _adc = adc.stop();     // Running → Ready

    // These would NOT compile:
    // Peripheral::<Uninit>::new("X").start(); // Can't start uninitialized
    // Peripheral::<Ready>::new("X").read();   // Can't read when not running
}
