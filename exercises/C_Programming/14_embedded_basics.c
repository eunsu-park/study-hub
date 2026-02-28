/*
 * Exercises for Lesson 14: Embedded Basics
 * Topic: C_Programming
 * Solutions to practice problems from the lesson.
 *
 * Note: The original exercises target Arduino hardware. These solutions
 * simulate the behavior in standard C using printf for serial output
 * and timing functions for delays, making them compilable and testable
 * on any desktop system.
 *
 * Compile: gcc -Wall -Wextra -std=c11 -o ex14 14_embedded_basics.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* --- Simulated Arduino API for desktop testing --- */

/* Simulate delay (prints action instead of actually sleeping) */
static void sim_delay(int ms) {
    printf("  [delay %d ms]\n", ms);
}

/* Simulate LED state */
static void sim_led(int pin, int state) {
    printf("  LED (pin %d): %s\n", pin, state ? "ON" : "OFF");
}

/* === Exercise 1: Heartbeat LED === */
/* Problem: Blink LED twice quickly (heartbeat), then pause. */
void exercise_1(void) {
    printf("=== Exercise 1: Heartbeat LED ===\n");

    /*
     * Arduino version:
     *   void loop() {
     *       digitalWrite(LED_PIN, HIGH); delay(100);
     *       digitalWrite(LED_PIN, LOW);  delay(100);
     *       digitalWrite(LED_PIN, HIGH); delay(100);
     *       digitalWrite(LED_PIN, LOW);  delay(700);
     *   }
     *
     * The pattern: two quick blinks (100ms on/off each) followed by
     * a longer pause (700ms), mimicking a heartbeat rhythm.
     */

    int led_pin = 13;
    int num_beats = 3; /* Simulate 3 heartbeat cycles */

    for (int cycle = 0; cycle < num_beats; cycle++) {
        printf("Heartbeat cycle %d:\n", cycle + 1);

        /* First beat - quick blink */
        sim_led(led_pin, 1);   /* ON */
        sim_delay(100);
        sim_led(led_pin, 0);   /* OFF */
        sim_delay(100);

        /* Second beat - quick blink */
        sim_led(led_pin, 1);   /* ON */
        sim_delay(100);
        sim_led(led_pin, 0);   /* OFF */

        /* Pause between heartbeats */
        sim_delay(700);
        printf("\n");
    }
}

/* === Exercise 2: Countdown === */
/* Problem: Countdown from 10 to 1 on serial, then blink LED 3 times at 0. */
void exercise_2(void) {
    printf("\n=== Exercise 2: Countdown ===\n");

    /*
     * Arduino version:
     *   void loop() {
     *       for (int i = 10; i >= 1; i--) {
     *           Serial.println(i);
     *           delay(1000);
     *       }
     *       Serial.println("Liftoff!");
     *       for (int j = 0; j < 3; j++) {
     *           digitalWrite(LED_PIN, HIGH); delay(200);
     *           digitalWrite(LED_PIN, LOW);  delay(200);
     *       }
     *       while (1) {} // Stop
     *   }
     */

    int led_pin = 13;

    /* Countdown phase */
    for (int i = 10; i >= 1; i--) {
        printf("Serial output: %d\n", i);
        /* In Arduino: delay(1000); */
    }
    printf("Serial output: Liftoff!\n");

    /* Blink LED 3 times */
    printf("\nBlinking LED 3 times:\n");
    for (int j = 0; j < 3; j++) {
        sim_led(led_pin, 1);
        sim_delay(200);
        sim_led(led_pin, 0);
        sim_delay(200);
    }
}

/* === Exercise 3: Random Blink === */
/* Problem: Blink LED at random intervals using random(). */
void exercise_3(void) {
    printf("\n=== Exercise 3: Random Blink ===\n");

    /*
     * Arduino version:
     *   void loop() {
     *       int randomDelay = random(100, 1000);
     *       digitalWrite(LED_PIN, HIGH);
     *       delay(randomDelay);
     *       digitalWrite(LED_PIN, LOW);
     *       delay(randomDelay);
     *   }
     *
     * random(min, max) returns a value in [min, max-1].
     */

    int led_pin = 13;
    srand((unsigned int)time(NULL)); /* Seed RNG (Arduino uses randomSeed()) */

    printf("Simulating 5 random blink cycles:\n");
    for (int i = 0; i < 5; i++) {
        /* random(100, 1000): value between 100 and 999 ms */
        int random_delay = 100 + rand() % 900;

        printf("Cycle %d (delay=%d ms):\n", i + 1, random_delay);
        sim_led(led_pin, 1);
        sim_delay(random_delay);
        sim_led(led_pin, 0);
        sim_delay(random_delay);
    }
}

/* === Exercise 4: Binary Counter === */
/* Problem: Use 4 LEDs to display 0~15 in binary. */
void exercise_4(void) {
    printf("\n=== Exercise 4: Binary Counter ===\n");

    /*
     * Arduino version (4 LEDs on pins 2-5):
     *   const int ledPins[] = {2, 3, 4, 5};
     *
     *   void setup() {
     *       for (int i = 0; i < 4; i++)
     *           pinMode(ledPins[i], OUTPUT);
     *   }
     *
     *   void loop() {
     *       for (int val = 0; val <= 15; val++) {
     *           for (int bit = 0; bit < 4; bit++) {
     *               digitalWrite(ledPins[bit], (val >> bit) & 1);
     *           }
     *           delay(500);
     *       }
     *   }
     *
     * Each LED represents one bit:
     *   LED1 (pin 2) = bit 0 (LSB)
     *   LED2 (pin 3) = bit 1
     *   LED3 (pin 4) = bit 2
     *   LED4 (pin 5) = bit 3 (MSB)
     */

    int led_pins[] = {2, 3, 4, 5};

    printf("%-6s  LED4  LED3  LED2  LED1\n", "Value");
    printf("%-6s  (b3)  (b2)  (b1)  (b0)\n", "");
    printf("------  ----  ----  ----  ----\n");

    for (int val = 0; val <= 15; val++) {
        printf("%-6d", val);
        /* Print bits from MSB (bit 3) to LSB (bit 0) */
        for (int bit = 3; bit >= 0; bit--) {
            int state = (val >> bit) & 1;
            printf("  %4s", state ? "ON" : "OFF");
        }
        printf("\n");

        /*
         * Verification for specific values from the exercise:
         * val= 0: 0000 -> all LEDs OFF
         * val= 5: 0101 -> LED4=OFF LED3=ON LED2=OFF LED1=ON
         * val=15: 1111 -> all LEDs ON
         */
    }
}

int main(void) {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();

    printf("\nAll exercises completed!\n");
    return 0;
}
