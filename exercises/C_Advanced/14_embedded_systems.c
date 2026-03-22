/*
 * Exercises for Lesson 14: Embedded Systems Programming
 * Topic: C_Advanced
 * Solutions to practice problems from the lesson.
 * Compile: gcc -Wall -Wextra -std=c11 -o ex14 14_embedded_systems.c
 * Note: Simulates embedded concepts on a desktop system.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* === Exercise 1: Register Manipulation === */
/* Problem: Simulate hardware register access using bitfield manipulation. */

/* Simulated memory-mapped register (in real embedded code, this would
 * be a volatile pointer to a hardware address) */
typedef volatile uint32_t reg32_t;

/* Register bit manipulation macros */
#define BIT(n)              (1U << (n))
#define SET_BIT(reg, n)     ((reg) |= BIT(n))
#define CLR_BIT(reg, n)     ((reg) &= ~BIT(n))
#define TOG_BIT(reg, n)     ((reg) ^= BIT(n))
#define GET_BIT(reg, n)     (((reg) >> (n)) & 1U)
#define SET_FIELD(reg, mask, shift, val) \
    ((reg) = ((reg) & ~((mask) << (shift))) | (((val) & (mask)) << (shift)))
#define GET_FIELD(reg, mask, shift) \
    (((reg) >> (shift)) & (mask))

/* Simulated GPIO control register */
#define GPIO_MODE_MASK  0x3U   /* 2 bits per pin */
#define GPIO_MODE_INPUT  0x0
#define GPIO_MODE_OUTPUT 0x1
#define GPIO_MODE_ALT    0x2
#define GPIO_MODE_ANALOG 0x3

void print_reg(const char *name, uint32_t reg) {
    printf("  %s = 0x%08X (", name, reg);
    for (int i = 31; i >= 0; i--) {
        printf("%d", (reg >> i) & 1);
        if (i > 0 && i % 4 == 0) printf("_");
    }
    printf(")\n");
}

void exercise_1(void) {
    printf("=== Exercise 1: Register Manipulation ===\n");

    uint32_t gpio_mode = 0x00000000;
    uint32_t gpio_output = 0x00000000;

    /* Configure pin 0 as output (bits 1:0) */
    SET_FIELD(gpio_mode, GPIO_MODE_MASK, 0, GPIO_MODE_OUTPUT);
    printf("Pin 0 set to OUTPUT:\n");
    print_reg("GPIO_MODE", gpio_mode);

    /* Configure pin 3 as alternate function (bits 7:6) */
    SET_FIELD(gpio_mode, GPIO_MODE_MASK, 6, GPIO_MODE_ALT);
    printf("Pin 3 set to ALT:\n");
    print_reg("GPIO_MODE", gpio_mode);

    /* Read back pin 3 mode */
    uint32_t pin3_mode = GET_FIELD(gpio_mode, GPIO_MODE_MASK, 6);
    printf("Pin 3 mode readback: %u (ALT=%u)\n", pin3_mode, GPIO_MODE_ALT);

    /* Toggle output pins */
    SET_BIT(gpio_output, 0);   /* Pin 0 high */
    SET_BIT(gpio_output, 4);   /* Pin 4 high */
    printf("Pins 0,4 set high:\n");
    print_reg("GPIO_OUT", gpio_output);

    TOG_BIT(gpio_output, 0);   /* Pin 0 toggle -> low */
    printf("Pin 0 toggled:\n");
    print_reg("GPIO_OUT", gpio_output);

    /*
     * In real embedded code:
     * - Registers are at fixed memory addresses (#define GPIOA ((volatile uint32_t*)0x40020000))
     * - volatile prevents compiler optimization of register reads/writes
     * - Bit manipulation must be atomic or protected by interrupts-off sections
     */
}

/* === Exercise 2: Protocol Parser === */
/* Problem: Parse a simple binary protocol frame:
 *   [SYNC(2)] [LEN(1)] [CMD(1)] [PAYLOAD(LEN)] [CRC(1)]
 *   SYNC = 0xAA 0x55
 */

#define PROTO_SYNC1 0xAA
#define PROTO_SYNC2 0x55
#define MAX_PAYLOAD 64

typedef struct {
    uint8_t cmd;
    uint8_t payload[MAX_PAYLOAD];
    uint8_t length;
    uint8_t crc;
} ProtoFrame;

typedef enum {
    PARSE_SYNC1,
    PARSE_SYNC2,
    PARSE_LEN,
    PARSE_CMD,
    PARSE_PAYLOAD,
    PARSE_CRC
} ParseState;

typedef struct {
    ParseState state;
    ProtoFrame frame;
    uint8_t payload_idx;
    uint8_t calc_crc;
} ProtoParser;

void parser_init(ProtoParser *p) {
    memset(p, 0, sizeof(*p));
    p->state = PARSE_SYNC1;
}

/* Simple XOR checksum */
static uint8_t update_crc(uint8_t crc, uint8_t byte) {
    return crc ^ byte;
}

/* Feed one byte to the parser. Returns 1 if a complete frame is ready. */
int parser_feed(ProtoParser *p, uint8_t byte) {
    switch (p->state) {
        case PARSE_SYNC1:
            if (byte == PROTO_SYNC1) p->state = PARSE_SYNC2;
            break;
        case PARSE_SYNC2:
            p->state = (byte == PROTO_SYNC2) ? PARSE_LEN : PARSE_SYNC1;
            break;
        case PARSE_LEN:
            if (byte > MAX_PAYLOAD) { p->state = PARSE_SYNC1; break; }
            p->frame.length = byte;
            p->calc_crc = byte;
            p->state = PARSE_CMD;
            break;
        case PARSE_CMD:
            p->frame.cmd = byte;
            p->calc_crc = update_crc(p->calc_crc, byte);
            p->payload_idx = 0;
            p->state = (p->frame.length > 0) ? PARSE_PAYLOAD : PARSE_CRC;
            break;
        case PARSE_PAYLOAD:
            p->frame.payload[p->payload_idx++] = byte;
            p->calc_crc = update_crc(p->calc_crc, byte);
            if (p->payload_idx >= p->frame.length) p->state = PARSE_CRC;
            break;
        case PARSE_CRC:
            p->frame.crc = byte;
            p->state = PARSE_SYNC1;
            return (p->calc_crc == byte) ? 1 : -1;  /* 1=valid, -1=CRC error */
    }
    return 0;
}

void exercise_2(void) {
    printf("\n=== Exercise 2: Protocol Parser ===\n");

    ProtoParser parser;
    parser_init(&parser);

    /* Construct a valid frame: SYNC(AA 55) LEN(3) CMD(01) DATA(DE AD BE) CRC */
    uint8_t payload[] = {0xDE, 0xAD, 0xBE};
    uint8_t crc = 0x03 ^ 0x01 ^ 0xDE ^ 0xAD ^ 0xBE;  /* XOR of LEN+CMD+DATA */

    uint8_t frame[] = {
        0xAA, 0x55,          /* sync */
        0x03,                /* length */
        0x01,                /* command */
        0xDE, 0xAD, 0xBE,   /* payload */
        crc                  /* CRC */
    };

    printf("Feeding valid frame (%zu bytes):\n", sizeof(frame));
    for (size_t i = 0; i < sizeof(frame); i++) {
        int result = parser_feed(&parser, frame[i]);
        if (result == 1) {
            printf("  Frame received! CMD=0x%02X, LEN=%d, DATA=",
                   parser.frame.cmd, parser.frame.length);
            for (int j = 0; j < parser.frame.length; j++) {
                printf("%02X ", parser.frame.payload[j]);
            }
            printf("\n");
        } else if (result == -1) {
            printf("  CRC error!\n");
        }
    }

    /* Test with corrupted frame */
    parser_init(&parser);
    frame[6] = 0xFF;  /* corrupt payload byte */
    printf("\nFeeding corrupted frame:\n");
    for (size_t i = 0; i < sizeof(frame); i++) {
        int result = parser_feed(&parser, frame[i]);
        if (result == -1) {
            printf("  CRC error detected (expected)\n");
        }
    }

    (void)payload;  /* suppress unused warning */
}

/* === Exercise 3: Bitfield Device Configuration === */
/* Problem: Use C bitfields to define a device configuration register. */

typedef struct {
    uint32_t enabled    : 1;   /* Bit 0: device enable */
    uint32_t mode       : 3;   /* Bits 1-3: operating mode (0-7) */
    uint32_t speed      : 4;   /* Bits 4-7: clock divider (0-15) */
    uint32_t irq_enable : 1;   /* Bit 8: interrupt enable */
    uint32_t dma_enable : 1;   /* Bit 9: DMA enable */
    uint32_t parity     : 2;   /* Bits 10-11: parity mode */
    uint32_t data_bits  : 2;   /* Bits 12-13: data width */
    uint32_t stop_bits  : 1;   /* Bit 14: stop bits (0=1, 1=2) */
    uint32_t reserved   : 17;  /* Bits 15-31: reserved */
} DeviceConfig;

/* Mode constants */
#define MODE_SPI    0
#define MODE_I2C    1
#define MODE_UART   2
#define MODE_PWM    3

/* Parity constants */
#define PARITY_NONE 0
#define PARITY_EVEN 1
#define PARITY_ODD  2

void print_config(const DeviceConfig *cfg) {
    printf("  enabled=%u, mode=%u, speed=%u, irq=%u, dma=%u\n",
           cfg->enabled, cfg->mode, cfg->speed,
           cfg->irq_enable, cfg->dma_enable);
    printf("  parity=%u, data_bits=%u, stop_bits=%u\n",
           cfg->parity, cfg->data_bits, cfg->stop_bits);
    /* Show raw value by casting */
    uint32_t raw;
    memcpy(&raw, cfg, sizeof(raw));
    print_reg("RAW", raw);
}

void exercise_3(void) {
    printf("\n=== Exercise 3: Bitfield Device Configuration ===\n");

    printf("sizeof(DeviceConfig) = %zu bytes\n", sizeof(DeviceConfig));

    /* Configure for UART: 8N1, 9600 baud (speed divider=4), with IRQ */
    DeviceConfig uart_cfg = {
        .enabled    = 1,
        .mode       = MODE_UART,
        .speed      = 4,
        .irq_enable = 1,
        .dma_enable = 0,
        .parity     = PARITY_NONE,
        .data_bits  = 3,  /* 3 = 8 bits */
        .stop_bits  = 0,  /* 0 = 1 stop bit */
        .reserved   = 0,
    };

    printf("\nUART 8N1 configuration:\n");
    print_config(&uart_cfg);

    /* Modify at runtime */
    uart_cfg.dma_enable = 1;
    uart_cfg.parity = PARITY_EVEN;
    printf("\nAfter enabling DMA and even parity:\n");
    print_config(&uart_cfg);

    /*
     * Bitfield caveats:
     * - Bit order is compiler/platform dependent (not portable)
     * - Cannot take address of a bitfield member
     * - Prefer explicit bit masks for truly portable register access
     * - Useful for readability in non-portable embedded code
     */
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();

    printf("\nAll exercises completed!\n");
    return 0;
}
