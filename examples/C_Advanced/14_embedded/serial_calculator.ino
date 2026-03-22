// serial_calculator.ino
// Serial Monitor Calculator

void setup() {
    Serial.begin(9600);

    Serial.println("=================================");
    Serial.println("   Simple Serial Calculator");
    Serial.println("=================================");
    Serial.println("Enter expression (e.g., 10 + 5)");
    Serial.println("Operators: +, -, *, /, %");
    Serial.println("Type 'quit' to exit");
    Serial.println("---------------------------------");
}

float calculate(float a, char op, float b) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/':
            if (b == 0) {
                Serial.println("Error: Division by zero");
                return 0;
            }
            return a / b;
        case '%':
            return (int)a % (int)b;
        default:
            Serial.print("Unknown operator: ");
            Serial.println(op);
            return 0;
    }
}

void processExpression(char* expr) {
    float num1, num2;
    char op;

    // Parse expression: "num1 op num2"
    int parsed = sscanf(expr, "%f %c %f", &num1, &op, &num2);

    if (parsed == 3) {
        float result = calculate(num1, op, num2);

        Serial.print(num1);
        Serial.print(" ");
        Serial.print(op);
        Serial.print(" ");
        Serial.print(num2);
        Serial.print(" = ");
        Serial.println(result);
    } else {
        Serial.println("Invalid format. Use: num1 op num2");
    }
}

char inputBuffer[32];
int inputIndex = 0;

void loop() {
    while (Serial.available()) {
        char c = Serial.read();

        if (c == '\n' || c == '\r') {
            if (inputIndex > 0) {
                inputBuffer[inputIndex] = '\0';

                // Check for quit command
                if (strcmp(inputBuffer, "quit") == 0) {
                    Serial.println("Goodbye!");
                    while (1);  // Halt
                }

                processExpression(inputBuffer);
                inputIndex = 0;

                Serial.println("---------------------------------");
            }
        } else if (inputIndex < 31) {
            inputBuffer[inputIndex++] = c;
        }
    }
}

/*
 * How to Use:
 * 1. Open Serial Monitor (Tools -> Serial Monitor)
 * 2. Set Baud rate to 9600
 * 3. Enter expression: "10 + 5" and press Enter
 * 4. Check result: "10 + 5 = 15.00"
 *
 * On Wokwi:
 * - Click the Serial Monitor tab
 * - Enter expression and press Enter
 */
