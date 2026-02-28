/*
 * Exercises for Lesson 19: Advanced Embedded Protocols
 * Topic: C_Programming
 * Solutions to practice problems from the lesson.
 *
 * Note: The original exercises target Arduino hardware (PWM, I2C).
 * These solutions simulate the protocol behavior in standard C,
 * demonstrating the logic and data flow without requiring hardware.
 *
 * Compile: gcc -Wall -Wextra -std=c11 -o ex19 19_advanced_embedded_protocols.c
 */
#include <stdio.h>
#include <stdint.h>

/* === Exercise 1: PWM LED Control === */
/* Problem: Increase LED brightness by 25% each button press, wrap at 100%. */

/*
 * PWM (Pulse Width Modulation) basics:
 * - analogWrite(pin, value) where value is 0-255
 * - 0 = 0% duty cycle (LED off), 255 = 100% duty cycle (LED full brightness)
 * - 25% steps in 256 range: 0, 64, 128, 192, 256->0 (wrap)
 *
 * Arduino solution:
 *   const int LED_PIN = 9;
 *   const int BUTTON_PIN = 2;
 *   int brightness = 0;
 *   bool lastButtonState = HIGH;
 *
 *   void setup() {
 *       pinMode(LED_PIN, OUTPUT);
 *       pinMode(BUTTON_PIN, INPUT_PULLUP);
 *   }
 *
 *   void loop() {
 *       bool buttonState = digitalRead(BUTTON_PIN);
 *       if (lastButtonState == HIGH && buttonState == LOW) {
 *           brightness = (brightness + 64) % 256;
 *           analogWrite(LED_PIN, brightness);
 *           delay(50);  // Debouncing
 *       }
 *       lastButtonState = buttonState;
 *   }
 */

/* Simulated PWM state */
typedef struct {
    int pin;
    int brightness;  /* 0-255 (8-bit PWM resolution) */
} PwmLed;

static void pwm_set(PwmLed *led, int brightness) {
    led->brightness = brightness % 256;
    float percent = (float)led->brightness / 255.0f * 100.0f;
    printf("  analogWrite(pin %d, %3d) -> %.0f%% brightness",
           led->pin, led->brightness, percent);

    /* Visual bar: each '#' represents ~5% */
    printf("  [");
    int bars = led->brightness * 20 / 255;
    for (int i = 0; i < 20; i++) {
        putchar(i < bars ? '#' : ' ');
    }
    printf("]\n");
}

void exercise_1(void) {
    printf("=== Exercise 1: PWM LED Control ===\n");

    PwmLed led = { .pin = 9, .brightness = 0 };

    printf("Simulating button presses (25%% brightness steps):\n\n");

    /* Simulate 6 button presses to show full cycle and wrap-around */
    for (int press = 0; press < 6; press++) {
        printf("Button press %d:\n", press + 1);

        /* Each press increases brightness by 64 (25% of 256) */
        led.brightness = (led.brightness + 64) % 256;
        pwm_set(&led, led.brightness);

        /*
         * Debouncing note: The 50ms delay after button press prevents
         * the mechanical switch bounce from being read as multiple presses.
         * A physical button can bounce for 5-20ms, causing multiple
         * HIGH->LOW transitions from a single press.
         */
    }

    printf("\n  Note: brightness values cycle: 64 -> 128 -> 192 -> 0 -> 64 -> ...\n");
    printf("  The modulo 256 ensures wrap-around from full to off.\n");
}

/* === Exercise 2: I2C Scanner === */
/* Problem: Scan all I2C addresses (1-126) and report connected devices. */

/*
 * I2C (Inter-Integrated Circuit) basics:
 * - Two-wire protocol: SDA (data) and SCL (clock)
 * - 7-bit addressing: addresses 0x01 to 0x7E (1-126)
 * - Address 0x00 is general call, 0x7F is reserved
 * - Master initiates communication by sending the device address
 * - If a device acknowledges (ACK), it exists at that address
 *
 * Arduino solution:
 *   #include <Wire.h>
 *   void setup() {
 *       Wire.begin();
 *       Serial.begin(9600);
 *       Serial.println("I2C Scanner");
 *   }
 *   void loop() {
 *       int deviceCount = 0;
 *       for (uint8_t addr = 1; addr < 127; addr++) {
 *           Wire.beginTransmission(addr);
 *           uint8_t error = Wire.endTransmission();
 *           if (error == 0) {
 *               Serial.print("Found device at 0x");
 *               Serial.println(addr, HEX);
 *               deviceCount++;
 *           }
 *       }
 *       Serial.print("Found ");
 *       Serial.print(deviceCount);
 *       Serial.println(" device(s)");
 *       delay(5000);
 *   }
 */

/* Simulated I2C bus with some common devices */
typedef struct {
    uint8_t address;
    const char *device_name;
} I2cDevice;

/* Common I2C device addresses for simulation */
static const I2cDevice simulated_bus[] = {
    {0x27, "LCD 16x2 (PCF8574)"},
    {0x3C, "OLED Display (SSD1306)"},
    {0x48, "Temperature Sensor (TMP102)"},
    {0x50, "EEPROM (AT24C32)"},
    {0x68, "RTC (DS3231) / MPU-6050"},
    {0x76, "Pressure Sensor (BMP280)"},
};

static int simulated_i2c_probe(uint8_t addr) {
    /* Check if the address matches any simulated device */
    int num_devices = (int)(sizeof(simulated_bus) / sizeof(simulated_bus[0]));
    for (int i = 0; i < num_devices; i++) {
        if (simulated_bus[i].address == addr) {
            return 0; /* ACK: device found */
        }
    }
    return 2; /* NACK: no device at this address */
}

static const char *get_device_name(uint8_t addr) {
    int num_devices = (int)(sizeof(simulated_bus) / sizeof(simulated_bus[0]));
    for (int i = 0; i < num_devices; i++) {
        if (simulated_bus[i].address == addr) {
            return simulated_bus[i].device_name;
        }
    }
    return "Unknown";
}

void exercise_2(void) {
    printf("\n=== Exercise 2: I2C Scanner ===\n");

    printf("Scanning I2C bus (addresses 0x01 - 0x7E)...\n\n");

    int device_count = 0;

    /* Scan all valid 7-bit I2C addresses */
    for (uint8_t addr = 1; addr < 127; addr++) {
        /*
         * Wire.beginTransmission(addr) starts an I2C transaction.
         * Wire.endTransmission() sends the address on the bus and returns:
         *   0: ACK received (device present)
         *   1: Data too long for buffer
         *   2: NACK on address (no device)
         *   3: NACK on data
         *   4: Other error
         */
        uint8_t error = (uint8_t)simulated_i2c_probe(addr);

        if (error == 0) {
            printf("  Found device at 0x%02X - %s\n", addr, get_device_name(addr));
            device_count++;
        }
    }

    printf("\nFound %d device(s) on the I2C bus.\n", device_count);

    /* Print the full I2C address map (useful for debugging) */
    printf("\nI2C Address Map (simulated):\n");
    printf("     0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F\n");
    for (int row = 0; row < 8; row++) {
        printf("%02X: ", row * 16);
        for (int col = 0; col < 16; col++) {
            uint8_t addr = (uint8_t)(row * 16 + col);
            if (addr < 1 || addr > 126) {
                printf("   "); /* Reserved address */
            } else if (simulated_i2c_probe(addr) == 0) {
                printf("%02X ", addr); /* Device found */
            } else {
                printf("-- "); /* No device */
            }
        }
        printf("\n");
    }
}

int main(void) {
    exercise_1();
    exercise_2();

    printf("\nAll exercises completed!\n");
    return 0;
}
