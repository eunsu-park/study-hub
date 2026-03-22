// button_led.ino
// Controlling LED with a Button

const int BUTTON_PIN = 2;  // Button pin
const int LED_PIN = 13;    // LED pin

bool ledState = false;
bool lastButtonState = HIGH;

void setup() {
    pinMode(BUTTON_PIN, INPUT_PULLUP);  // Use internal pull-up
    pinMode(LED_PIN, OUTPUT);

    Serial.begin(9600);
    Serial.println("Button LED Control");
    Serial.println("Press button to toggle LED");
}

void loop() {
    bool currentButtonState = digitalRead(BUTTON_PIN);

    // When button is pressed (HIGH -> LOW)
    if (lastButtonState == HIGH && currentButtonState == LOW) {
        ledState = !ledState;  // Toggle LED state
        digitalWrite(LED_PIN, ledState);

        Serial.print("LED ");
        Serial.println(ledState ? "ON" : "OFF");
    }

    lastButtonState = currentButtonState;
    delay(50);  // Simple debouncing
}

/*
 * Wokwi Circuit Setup:
 * 1. Add Arduino Uno
 * 2. Add Pushbutton
 * 3. Connections:
 *    - One side of button -> Pin 2
 *    - Other side of button -> GND
 *
 * The LED toggles on and off each time the button is pressed.
 */
