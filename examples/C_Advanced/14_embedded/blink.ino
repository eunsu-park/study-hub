// blink.ino
// LED Blink - The most basic Arduino program

const int LED_PIN = 13;  // Built-in LED (or use LED_BUILTIN)

void setup() {
    // Set pin mode to output
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    // Turn LED on
    digitalWrite(LED_PIN, HIGH);
    delay(1000);  // Wait 1 second

    // Turn LED off
    digitalWrite(LED_PIN, LOW);
    delay(1000);  // Wait 1 second
}

/*
 * Running on Wokwi:
 * 1. Visit https://wokwi.com
 * 2. New Project -> Arduino Uno
 * 3. Paste this code
 * 4. Start Simulation
 *
 * The built-in LED will blink at 1-second intervals.
 */
