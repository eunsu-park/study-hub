// Exercise: Embedded Rust
// Practice no_std programming and embedded patterns.
//
// Run: rustc 26_embedded.rs && ./26_embedded

// Exercise 1: no_std ring buffer
// Implement a fixed-size circular buffer using only core.

struct RingBuffer<T, const N: usize> {
    data: [Option<T>; N],
    head: usize,
    tail: usize,
    len: usize,
}

impl<T: Copy + Default, const N: usize> RingBuffer<T, N> {
    fn new() -> Self {
        RingBuffer {
            data: [None; N],
            head: 0,
            tail: 0,
            len: 0,
        }
    }

    fn push(&mut self, item: T) -> Result<(), T> {
        if self.len == N {
            return Err(item);  // Buffer full
        }
        self.data[self.tail] = Some(item);
        self.tail = (self.tail + 1) % N;
        self.len += 1;
        Ok(())
    }

    fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let item = self.data[self.head].take();
        self.head = (self.head + 1) % N;
        self.len -= 1;
        item
    }

    fn is_empty(&self) -> bool { self.len == 0 }
    fn is_full(&self) -> bool { self.len == N }
    fn len(&self) -> usize { self.len }
}

// Exercise 2: State machine protocol parser
#[derive(Debug, PartialEq)]
enum ParseState {
    WaitingSync,
    ReadingHeader(usize),
    ReadingPayload(usize, usize),  // (payload_len, bytes_read)
    Complete,
    Error,
}

struct ProtocolParser {
    state: ParseState,
    header: [u8; 4],
    payload: Vec<u8>,
}

impl ProtocolParser {
    fn new() -> Self {
        ProtocolParser {
            state: ParseState::WaitingSync,
            header: [0; 4],
            payload: Vec::new(),
        }
    }

    fn feed(&mut self, byte: u8) {
        self.state = match &self.state {
            ParseState::WaitingSync => {
                if byte == 0xAA {
                    ParseState::ReadingHeader(0)
                } else {
                    ParseState::WaitingSync
                }
            }
            ParseState::ReadingHeader(n) => {
                let n = *n;
                self.header[n] = byte;
                if n + 1 >= 4 {
                    let payload_len = self.header[3] as usize;
                    self.payload = Vec::with_capacity(payload_len);
                    if payload_len == 0 {
                        ParseState::Complete
                    } else {
                        ParseState::ReadingPayload(payload_len, 0)
                    }
                } else {
                    ParseState::ReadingHeader(n + 1)
                }
            }
            ParseState::ReadingPayload(len, read) => {
                let len = *len;
                let read = *read;
                self.payload.push(byte);
                if read + 1 >= len {
                    ParseState::Complete
                } else {
                    ParseState::ReadingPayload(len, read + 1)
                }
            }
            _ => ParseState::Error,
        };
    }

    fn is_complete(&self) -> bool {
        self.state == ParseState::Complete
    }

    fn reset(&mut self) {
        self.state = ParseState::WaitingSync;
        self.header = [0; 4];
        self.payload.clear();
    }
}

fn main() {
    // Test Exercise 1
    let mut buf: RingBuffer<i32, 4> = RingBuffer::new();
    assert!(buf.is_empty());
    buf.push(1).unwrap();
    buf.push(2).unwrap();
    buf.push(3).unwrap();
    buf.push(4).unwrap();
    assert!(buf.is_full());
    assert!(buf.push(5).is_err());

    assert_eq!(buf.pop(), Some(1));
    assert_eq!(buf.pop(), Some(2));
    buf.push(5).unwrap();
    assert_eq!(buf.pop(), Some(3));
    assert_eq!(buf.pop(), Some(4));
    assert_eq!(buf.pop(), Some(5));
    assert!(buf.is_empty());
    println!("RingBuffer works correctly!");

    // Test Exercise 2
    let mut parser = ProtocolParser::new();
    // Send: sync(0xAA) + header(01 02 03 02) + payload(FF EE)
    let packet = [0xAA, 0x01, 0x02, 0x03, 0x02, 0xFF, 0xEE];
    for &byte in &packet {
        parser.feed(byte);
    }
    assert!(parser.is_complete());
    assert_eq!(parser.payload, vec![0xFF, 0xEE]);
    println!("Protocol parser works correctly!");
    println!("Header: {:02X?}", parser.header);
    println!("Payload: {:02X?}", parser.payload);

    println!("\nAll exercises passed!");
}
