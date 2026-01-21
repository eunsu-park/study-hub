# 프로젝트 16: 시리얼 통신

UART 시리얼 통신을 이용한 PC와의 양방향 데이터 교환을 배웁니다.

## 학습 목표
- UART 통신 원리 이해
- 시리얼 데이터 송수신
- 문자열 파싱
- 명령어 인터프리터 구현
- 디버깅 기법

## 사전 지식
- Arduino 기본 구조
- 문자열 처리
- GPIO 제어

---

## 1. UART 통신 개념

### UART란?

**UART (Universal Asynchronous Receiver/Transmitter)**는 가장 기본적인 시리얼 통신 방식입니다.

```
UART 통신 구조:

Arduino                           PC
┌─────────┐                    ┌─────────┐
│         │───── TX ─────────→│ RX      │
│  MCU    │                    │  USB    │
│         │←───── RX ─────────│ TX      │
│         │───── GND ─────────│ GND     │
└─────────┘                    └─────────┘

- TX (Transmit): 데이터 송신
- RX (Receive): 데이터 수신
- 비동기 통신: 클럭 신호 없이 약속된 속도로 통신
```

### 통신 파라미터

```
Baud Rate (보레이트):
- 초당 전송 비트 수
- 일반적인 값: 9600, 19200, 38400, 57600, 115200
- 송신측과 수신측이 같아야 함

데이터 프레임:
┌─────┬────────────┬────────┬─────┐
│Start│  Data bits │ Parity │Stop │
│ bit │ (5-9 bits) │ (opt)  │bit  │
└─────┴────────────┴────────┴─────┘

일반적인 설정: 8N1
- 8 데이터 비트
- N (No parity): 패리티 없음
- 1 스톱 비트
```

### 9600 Baud 전송 예시

```
문자 'A' (ASCII 65 = 0b01000001) 전송:

HIGH ─────┐     ┌─────┐                 ┌─────────
          │     │     │                 │
LOW       └─────┘     └─────────────────┘
          │Start│  0  1  0  0  0  0  0  1 │ Stop │
          │ bit │      Data bits (LSB first)│ bit │
                        ← 'A' = 0x41 →

시간: 1/9600 ≈ 104μs per bit
1 문자 = 10 bits = 약 1.04ms
초당 최대 약 960 문자 전송 가능
```

---

## 2. Arduino Serial 라이브러리

### 기본 함수

```cpp
// 시리얼 초기화
Serial.begin(baudrate);      // 9600, 115200 등
Serial.begin(9600, config);  // 설정 포함 (SERIAL_8N1 등)

// 송신 (출력)
Serial.print(data);          // 줄바꿈 없이
Serial.println(data);        // 줄바꿈 포함
Serial.write(byte);          // 1바이트 전송
Serial.write(buffer, len);   // 버퍼 전송

// 수신 (입력)
Serial.available();          // 수신 대기 중인 바이트 수
Serial.read();               // 1바이트 읽기 (-1 if empty)
Serial.peek();               // 읽지 않고 확인
Serial.readBytes(buf, len);  // 여러 바이트 읽기
Serial.readString();         // 문자열로 읽기
Serial.readStringUntil(ch);  // 특정 문자까지 읽기

// 기타
Serial.flush();              // 송신 완료까지 대기
Serial.setTimeout(ms);       // 타임아웃 설정 (기본 1000ms)
```

### 기본 출력

```cpp
// serial_output.ino
// 다양한 형식의 시리얼 출력

void setup() {
    Serial.begin(9600);

    // 연결 대기 (선택적)
    while (!Serial) {
        ; // USB 연결 대기
    }

    Serial.println("=== Serial Output Demo ===");
}

void loop() {
    // 문자열 출력
    Serial.println("Hello, World!");

    // 숫자 출력
    int num = 42;
    Serial.print("Number: ");
    Serial.println(num);

    // 다양한 진법
    Serial.print("Decimal: ");
    Serial.println(255, DEC);    // 255

    Serial.print("Binary:  ");
    Serial.println(255, BIN);    // 11111111

    Serial.print("Octal:   ");
    Serial.println(255, OCT);    // 377

    Serial.print("Hex:     ");
    Serial.println(255, HEX);    // FF

    // 실수 출력
    float pi = 3.14159;
    Serial.print("Pi: ");
    Serial.println(pi, 4);       // 소수점 4자리

    // 포맷 문자열 (sprintf 사용)
    char buffer[50];
    sprintf(buffer, "x=%d, y=%d, val=%.2f", 10, 20, 3.14);
    Serial.println(buffer);

    delay(3000);
}
```

### 기본 입력

```cpp
// serial_input.ino
// 시리얼 입력 처리

void setup() {
    Serial.begin(9600);
    Serial.println("Type something and press Enter:");
}

void loop() {
    // 수신 데이터가 있는지 확인
    if (Serial.available() > 0) {
        // 한 문자 읽기
        char c = Serial.read();

        // 에코 (받은 문자 그대로 출력)
        Serial.print("Received: '");
        Serial.print(c);
        Serial.print("' (ASCII ");
        Serial.print((int)c);
        Serial.println(")");
    }
}
```

---

## 3. 문자열 수신 처리

### 줄 단위 읽기

```cpp
// serial_readline.ino
// 한 줄씩 읽기

String inputString = "";
bool stringComplete = false;

void setup() {
    Serial.begin(9600);
    inputString.reserve(200);  // 메모리 예약
    Serial.println("Enter a line:");
}

void loop() {
    // 한 줄 완성되면 처리
    if (stringComplete) {
        Serial.print("You entered: ");
        Serial.println(inputString);

        // 초기화
        inputString = "";
        stringComplete = false;
    }
}

// 시리얼 이벤트 (자동 호출)
void serialEvent() {
    while (Serial.available()) {
        char c = (char)Serial.read();

        if (c == '\n') {
            stringComplete = true;
        } else if (c != '\r') {  // CR 무시
            inputString += c;
        }
    }
}
```

### char 배열로 읽기 (메모리 효율적)

```cpp
// serial_char_array.ino
// char 배열 사용 (String 대신)

#define MAX_INPUT 64

char inputBuffer[MAX_INPUT];
int inputIndex = 0;
bool lineReady = false;

void setup() {
    Serial.begin(9600);
    Serial.println("Enter command:");
}

void loop() {
    // 데이터 수신
    while (Serial.available() && !lineReady) {
        char c = Serial.read();

        if (c == '\n' || c == '\r') {
            if (inputIndex > 0) {
                inputBuffer[inputIndex] = '\0';  // 문자열 종료
                lineReady = true;
            }
        } else if (inputIndex < MAX_INPUT - 1) {
            inputBuffer[inputIndex++] = c;
        }
    }

    // 완성된 라인 처리
    if (lineReady) {
        Serial.print("Command: ");
        Serial.println(inputBuffer);

        // 처리 후 초기화
        inputIndex = 0;
        lineReady = false;
    }
}
```

---

## 4. 명령어 파싱

### 단순 명령어 처리

```cpp
// serial_commands.ino
// 간단한 명령어 처리

const int LED_PIN = 13;

void setup() {
    Serial.begin(9600);
    pinMode(LED_PIN, OUTPUT);

    Serial.println("=== LED Control ===");
    Serial.println("Commands: ON, OFF, BLINK, STATUS");
}

void processCommand(String cmd) {
    cmd.trim();  // 앞뒤 공백 제거
    cmd.toUpperCase();  // 대문자로 변환

    if (cmd == "ON") {
        digitalWrite(LED_PIN, HIGH);
        Serial.println("LED turned ON");
    }
    else if (cmd == "OFF") {
        digitalWrite(LED_PIN, LOW);
        Serial.println("LED turned OFF");
    }
    else if (cmd == "BLINK") {
        Serial.println("Blinking 5 times...");
        for (int i = 0; i < 5; i++) {
            digitalWrite(LED_PIN, HIGH);
            delay(200);
            digitalWrite(LED_PIN, LOW);
            delay(200);
        }
        Serial.println("Done");
    }
    else if (cmd == "STATUS") {
        Serial.print("LED is ");
        Serial.println(digitalRead(LED_PIN) ? "ON" : "OFF");
    }
    else {
        Serial.print("Unknown command: ");
        Serial.println(cmd);
    }
}

void loop() {
    if (Serial.available()) {
        String input = Serial.readStringUntil('\n');
        processCommand(input);
    }
}
```

### 인자가 있는 명령어

```cpp
// serial_args.ino
// 인자가 있는 명령어 처리

const int LED_PINS[] = {9, 10, 11, 12};
const int NUM_LEDS = 4;

void setup() {
    Serial.begin(9600);
    for (int i = 0; i < NUM_LEDS; i++) {
        pinMode(LED_PINS[i], OUTPUT);
    }

    Serial.println("=== Multi-LED Control ===");
    Serial.println("Commands:");
    Serial.println("  SET <led> <state>  - Set LED 0-3 to 0/1");
    Serial.println("  PATTERN <value>    - Set pattern 0-15");
    Serial.println("  DELAY <ms>         - Set delay time");
    Serial.println("  HELP               - Show this help");
}

void setLED(int led, int state) {
    if (led >= 0 && led < NUM_LEDS) {
        digitalWrite(LED_PINS[led], state ? HIGH : LOW);
        Serial.print("LED ");
        Serial.print(led);
        Serial.print(" set to ");
        Serial.println(state ? "ON" : "OFF");
    } else {
        Serial.println("Invalid LED number (0-3)");
    }
}

void setPattern(int pattern) {
    for (int i = 0; i < NUM_LEDS; i++) {
        digitalWrite(LED_PINS[i], (pattern >> i) & 1);
    }
    Serial.print("Pattern set to ");
    Serial.print(pattern);
    Serial.print(" (0b");
    for (int i = 3; i >= 0; i--) {
        Serial.print((pattern >> i) & 1);
    }
    Serial.println(")");
}

void processCommand(char* input) {
    char* cmd = strtok(input, " ");
    if (cmd == NULL) return;

    // 대문자로 변환
    for (int i = 0; cmd[i]; i++) {
        cmd[i] = toupper(cmd[i]);
    }

    if (strcmp(cmd, "SET") == 0) {
        char* ledStr = strtok(NULL, " ");
        char* stateStr = strtok(NULL, " ");

        if (ledStr && stateStr) {
            int led = atoi(ledStr);
            int state = atoi(stateStr);
            setLED(led, state);
        } else {
            Serial.println("Usage: SET <led> <state>");
        }
    }
    else if (strcmp(cmd, "PATTERN") == 0) {
        char* valStr = strtok(NULL, " ");
        if (valStr) {
            int pattern = atoi(valStr);
            setPattern(pattern & 0x0F);
        } else {
            Serial.println("Usage: PATTERN <0-15>");
        }
    }
    else if (strcmp(cmd, "HELP") == 0) {
        Serial.println("\nCommands:");
        Serial.println("  SET <led> <state>");
        Serial.println("  PATTERN <value>");
        Serial.println("  HELP");
    }
    else {
        Serial.print("Unknown: ");
        Serial.println(cmd);
    }
}

char inputBuffer[64];
int inputIndex = 0;

void loop() {
    while (Serial.available()) {
        char c = Serial.read();

        if (c == '\n' || c == '\r') {
            if (inputIndex > 0) {
                inputBuffer[inputIndex] = '\0';
                processCommand(inputBuffer);
                inputIndex = 0;
            }
        } else if (inputIndex < 63) {
            inputBuffer[inputIndex++] = c;
        }
    }
}
```

---

## 5. 실습: 시리얼 모니터 계산기

```cpp
// serial_calculator.ino
// 시리얼로 수식을 입력받아 계산

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

    // 수식 파싱: "num1 op num2"
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

                // 종료 명령 확인
                if (strcmp(inputBuffer, "quit") == 0) {
                    Serial.println("Goodbye!");
                    while (1);  // 정지
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
```

---

## 6. 데이터 프로토콜 설계

### 간단한 프로토콜 예시

```cpp
// serial_protocol.ino
// 간단한 통신 프로토콜 구현

// 프로토콜 형식:
// <STX><TYPE><LENGTH><DATA><CHECKSUM><ETX>
// STX = 0x02 (Start of Text)
// ETX = 0x03 (End of Text)

#define STX 0x02
#define ETX 0x03

// 메시지 타입
#define MSG_LED_SET     0x01
#define MSG_LED_GET     0x02
#define MSG_TEMP_GET    0x03
#define MSG_ACK         0x10
#define MSG_ERROR       0xFF

const int LED_PIN = 13;

void sendMessage(byte type, byte* data, byte length) {
    byte checksum = type ^ length;
    for (int i = 0; i < length; i++) {
        checksum ^= data[i];
    }

    Serial.write(STX);
    Serial.write(type);
    Serial.write(length);
    Serial.write(data, length);
    Serial.write(checksum);
    Serial.write(ETX);
}

void sendAck() {
    byte data[] = {0x00};
    sendMessage(MSG_ACK, data, 1);
}

void sendError(byte code) {
    byte data[] = {code};
    sendMessage(MSG_ERROR, data, 1);
}

void processMessage(byte type, byte* data, byte length) {
    switch (type) {
        case MSG_LED_SET:
            if (length >= 1) {
                digitalWrite(LED_PIN, data[0] ? HIGH : LOW);
                sendAck();
            } else {
                sendError(0x01);  // Invalid length
            }
            break;

        case MSG_LED_GET:
            {
                byte state = digitalRead(LED_PIN);
                sendMessage(MSG_LED_GET, &state, 1);
            }
            break;

        case MSG_TEMP_GET:
            {
                // 임의의 온도 값 (실제로는 센서에서 읽음)
                byte temp[] = {25, 50};  // 25.50도
                sendMessage(MSG_TEMP_GET, temp, 2);
            }
            break;

        default:
            sendError(0x02);  // Unknown type
    }
}

// 수신 상태 머신
enum RxState { WAIT_STX, WAIT_TYPE, WAIT_LENGTH, WAIT_DATA, WAIT_CHECKSUM, WAIT_ETX };
RxState rxState = WAIT_STX;

byte rxType, rxLength, rxChecksum;
byte rxData[32];
byte rxIndex;

void setup() {
    Serial.begin(9600);
    pinMode(LED_PIN, OUTPUT);
}

void loop() {
    while (Serial.available()) {
        byte b = Serial.read();

        switch (rxState) {
            case WAIT_STX:
                if (b == STX) rxState = WAIT_TYPE;
                break;

            case WAIT_TYPE:
                rxType = b;
                rxChecksum = b;
                rxState = WAIT_LENGTH;
                break;

            case WAIT_LENGTH:
                rxLength = b;
                rxChecksum ^= b;
                rxIndex = 0;
                rxState = (rxLength > 0) ? WAIT_DATA : WAIT_CHECKSUM;
                break;

            case WAIT_DATA:
                rxData[rxIndex++] = b;
                rxChecksum ^= b;
                if (rxIndex >= rxLength) {
                    rxState = WAIT_CHECKSUM;
                }
                break;

            case WAIT_CHECKSUM:
                if (b == rxChecksum) {
                    rxState = WAIT_ETX;
                } else {
                    sendError(0x03);  // Checksum error
                    rxState = WAIT_STX;
                }
                break;

            case WAIT_ETX:
                if (b == ETX) {
                    processMessage(rxType, rxData, rxLength);
                }
                rxState = WAIT_STX;
                break;
        }
    }
}
```

### JSON 형식 사용

```cpp
// serial_json.ino
// JSON 형식 통신 (ArduinoJson 라이브러리 필요)

// Library Manager에서 "ArduinoJson" 설치

#include <ArduinoJson.h>

const int LED_PIN = 13;
int brightness = 0;
String deviceName = "Arduino";

void setup() {
    Serial.begin(9600);
    pinMode(LED_PIN, OUTPUT);

    Serial.println("{\"status\":\"ready\",\"device\":\"Arduino\"}");
}

void processJson(const char* json) {
    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, json);

    if (error) {
        Serial.print("{\"error\":\"");
        Serial.print(error.c_str());
        Serial.println("\"}");
        return;
    }

    const char* cmd = doc["cmd"];

    if (strcmp(cmd, "set_led") == 0) {
        bool state = doc["state"];
        digitalWrite(LED_PIN, state ? HIGH : LOW);
        Serial.println("{\"result\":\"ok\"}");
    }
    else if (strcmp(cmd, "get_status") == 0) {
        StaticJsonDocument<200> response;
        response["led"] = digitalRead(LED_PIN);
        response["uptime"] = millis() / 1000;
        response["device"] = deviceName;

        serializeJson(response, Serial);
        Serial.println();
    }
    else if (strcmp(cmd, "set_name") == 0) {
        deviceName = doc["name"].as<String>();
        Serial.println("{\"result\":\"ok\"}");
    }
    else {
        Serial.println("{\"error\":\"unknown command\"}");
    }
}

char inputBuffer[256];
int inputIndex = 0;

void loop() {
    while (Serial.available()) {
        char c = Serial.read();

        if (c == '\n') {
            inputBuffer[inputIndex] = '\0';
            if (inputIndex > 0) {
                processJson(inputBuffer);
            }
            inputIndex = 0;
        } else if (inputIndex < 255) {
            inputBuffer[inputIndex++] = c;
        }
    }
}

// 사용 예:
// {"cmd":"set_led","state":true}
// {"cmd":"get_status"}
// {"cmd":"set_name","name":"MyDevice"}
```

---

## 7. 디버깅 기법

### 디버그 매크로

```cpp
// debug_macros.ino
// 디버그 출력 매크로

#define DEBUG 1  // 0으로 변경하면 디버그 출력 비활성화

#if DEBUG
    #define DEBUG_PRINT(x)    Serial.print(x)
    #define DEBUG_PRINTLN(x)  Serial.println(x)
    #define DEBUG_PRINTF(...)  { char buf[128]; sprintf(buf, __VA_ARGS__); Serial.print(buf); }
#else
    #define DEBUG_PRINT(x)
    #define DEBUG_PRINTLN(x)
    #define DEBUG_PRINTF(...)
#endif

const int LED_PIN = 13;
const int BUTTON_PIN = 2;

void setup() {
    Serial.begin(9600);
    pinMode(LED_PIN, OUTPUT);
    pinMode(BUTTON_PIN, INPUT_PULLUP);

    DEBUG_PRINTLN("=== Debug Demo ===");
    DEBUG_PRINTF("LED pin: %d\n", LED_PIN);
    DEBUG_PRINTF("Button pin: %d\n", BUTTON_PIN);
}

void loop() {
    static bool lastState = HIGH;
    bool currentState = digitalRead(BUTTON_PIN);

    if (currentState != lastState) {
        DEBUG_PRINTF("Button state changed: %d -> %d\n", lastState, currentState);

        if (currentState == LOW) {
            DEBUG_PRINTLN("Button pressed!");
            digitalWrite(LED_PIN, !digitalRead(LED_PIN));
        }

        lastState = currentState;
    }

    delay(10);
}
```

### 변수 모니터링

```cpp
// serial_monitor.ino
// 실시간 변수 모니터링

const int SENSOR_PIN = A0;
const int LED_PIN = 13;

unsigned long lastPrint = 0;
const unsigned long printInterval = 500;

int sensorValue = 0;
int ledState = LOW;
unsigned long uptime = 0;
int loopCount = 0;

void setup() {
    Serial.begin(115200);  // 빠른 속도 사용
    pinMode(LED_PIN, OUTPUT);

    // CSV 헤더 출력
    Serial.println("time_ms,sensor,led,loop_count");
}

void loop() {
    loopCount++;

    // 센서 읽기
    sensorValue = analogRead(SENSOR_PIN);

    // LED 제어 (센서값에 따라)
    ledState = (sensorValue > 512) ? HIGH : LOW;
    digitalWrite(LED_PIN, ledState);

    // 주기적으로 데이터 출력
    if (millis() - lastPrint >= printInterval) {
        lastPrint = millis();

        // CSV 형식 출력
        Serial.print(millis());
        Serial.print(",");
        Serial.print(sensorValue);
        Serial.print(",");
        Serial.print(ledState);
        Serial.print(",");
        Serial.println(loopCount);

        loopCount = 0;
    }
}

// Serial Plotter에서 그래프로 확인 가능
// Tools → Serial Plotter
```

### 상태 머신 디버깅

```cpp
// state_debug.ino
// 상태 머신 디버깅

enum State { IDLE, RUNNING, PAUSED, ERROR };
State currentState = IDLE;
State lastState = IDLE;

const char* stateNames[] = {"IDLE", "RUNNING", "PAUSED", "ERROR"};

void printState() {
    if (currentState != lastState) {
        Serial.print("[STATE] ");
        Serial.print(stateNames[lastState]);
        Serial.print(" -> ");
        Serial.println(stateNames[currentState]);
        lastState = currentState;
    }
}

void setup() {
    Serial.begin(9600);
    pinMode(2, INPUT_PULLUP);  // Start
    pinMode(3, INPUT_PULLUP);  // Pause
    pinMode(13, OUTPUT);

    Serial.println("State Machine Debug");
    Serial.println("BTN1: Start/Resume, BTN2: Pause");
}

void loop() {
    bool btn1 = !digitalRead(2);
    bool btn2 = !digitalRead(3);

    switch (currentState) {
        case IDLE:
            if (btn1) currentState = RUNNING;
            break;

        case RUNNING:
            digitalWrite(13, (millis() / 500) % 2);  // LED 깜빡임
            if (btn2) currentState = PAUSED;
            break;

        case PAUSED:
            digitalWrite(13, LOW);
            if (btn1) currentState = RUNNING;
            break;

        case ERROR:
            // 에러 처리
            break;
    }

    printState();
    delay(50);
}
```

---

## 8. 실습 프로젝트: 터미널 인터페이스

```cpp
// terminal_interface.ino
// 완성된 터미널 인터페이스

#define VERSION "1.0.0"
#define MAX_CMD_LEN 64
#define MAX_ARGS 8

// 핀 설정
const int LED_PINS[] = {9, 10, 11, 12, 13};
const int NUM_LEDS = 5;
const int BUTTON_PIN = 2;

// 변수
char cmdBuffer[MAX_CMD_LEN];
int cmdIndex = 0;
bool echoEnabled = true;
unsigned long startTime;

void setup() {
    Serial.begin(9600);

    for (int i = 0; i < NUM_LEDS; i++) {
        pinMode(LED_PINS[i], OUTPUT);
    }
    pinMode(BUTTON_PIN, INPUT_PULLUP);

    startTime = millis();
    printWelcome();
    printPrompt();
}

void printWelcome() {
    Serial.println();
    Serial.println("========================================");
    Serial.println("     Arduino Terminal Interface");
    Serial.print("           Version ");
    Serial.println(VERSION);
    Serial.println("========================================");
    Serial.println("Type 'help' for available commands");
    Serial.println();
}

void printPrompt() {
    Serial.print("> ");
}

void printHelp() {
    Serial.println("\nAvailable commands:");
    Serial.println("  help              - Show this help");
    Serial.println("  led <n> <on/off>  - Control LED 0-4");
    Serial.println("  led all <on/off>  - Control all LEDs");
    Serial.println("  pattern <0-31>    - Set LED pattern");
    Serial.println("  status            - Show system status");
    Serial.println("  button            - Read button state");
    Serial.println("  uptime            - Show uptime");
    Serial.println("  echo <on/off>     - Toggle echo");
    Serial.println("  clear             - Clear screen");
    Serial.println("  reboot            - Soft reboot");
    Serial.println();
}

void cmdLed(int argc, char* argv[]) {
    if (argc < 3) {
        Serial.println("Usage: led <n|all> <on|off>");
        return;
    }

    bool state = (strcmp(argv[2], "on") == 0 || strcmp(argv[2], "1") == 0);

    if (strcmp(argv[1], "all") == 0) {
        for (int i = 0; i < NUM_LEDS; i++) {
            digitalWrite(LED_PINS[i], state);
        }
        Serial.print("All LEDs ");
        Serial.println(state ? "ON" : "OFF");
    } else {
        int led = atoi(argv[1]);
        if (led >= 0 && led < NUM_LEDS) {
            digitalWrite(LED_PINS[led], state);
            Serial.print("LED ");
            Serial.print(led);
            Serial.print(" ");
            Serial.println(state ? "ON" : "OFF");
        } else {
            Serial.println("Invalid LED number (0-4)");
        }
    }
}

void cmdPattern(int argc, char* argv[]) {
    if (argc < 2) {
        Serial.println("Usage: pattern <0-31>");
        return;
    }

    int pattern = atoi(argv[1]) & 0x1F;
    for (int i = 0; i < NUM_LEDS; i++) {
        digitalWrite(LED_PINS[i], (pattern >> i) & 1);
    }

    Serial.print("Pattern: 0b");
    for (int i = 4; i >= 0; i--) {
        Serial.print((pattern >> i) & 1);
    }
    Serial.print(" (");
    Serial.print(pattern);
    Serial.println(")");
}

void cmdStatus() {
    Serial.println("\n--- System Status ---");

    Serial.print("LEDs: ");
    for (int i = 0; i < NUM_LEDS; i++) {
        Serial.print(digitalRead(LED_PINS[i]) ? "1" : "0");
    }
    Serial.println();

    Serial.print("Button: ");
    Serial.println(digitalRead(BUTTON_PIN) ? "Released" : "Pressed");

    Serial.print("Uptime: ");
    printUptime();

    Serial.print("Free RAM: ");
    Serial.print(freeMemory());
    Serial.println(" bytes");

    Serial.println("---------------------\n");
}

void printUptime() {
    unsigned long secs = (millis() - startTime) / 1000;
    int hours = secs / 3600;
    int mins = (secs % 3600) / 60;
    int sec = secs % 60;

    char buf[20];
    sprintf(buf, "%02d:%02d:%02d", hours, mins, sec);
    Serial.println(buf);
}

int freeMemory() {
    extern int __heap_start, *__brkval;
    int v;
    return (int)&v - (__brkval == 0 ? (int)&__heap_start : (int)__brkval);
}

void processCommand(char* cmd) {
    // 빈 명령 무시
    if (strlen(cmd) == 0) return;

    // 토큰 분리
    char* argv[MAX_ARGS];
    int argc = 0;

    char* token = strtok(cmd, " ");
    while (token && argc < MAX_ARGS) {
        argv[argc++] = token;
        token = strtok(NULL, " ");
    }

    // 소문자로 변환 (명령어만)
    for (int i = 0; argv[0][i]; i++) {
        argv[0][i] = tolower(argv[0][i]);
    }

    // 명령어 처리
    if (strcmp(argv[0], "help") == 0) {
        printHelp();
    }
    else if (strcmp(argv[0], "led") == 0) {
        cmdLed(argc, argv);
    }
    else if (strcmp(argv[0], "pattern") == 0) {
        cmdPattern(argc, argv);
    }
    else if (strcmp(argv[0], "status") == 0) {
        cmdStatus();
    }
    else if (strcmp(argv[0], "button") == 0) {
        Serial.print("Button: ");
        Serial.println(digitalRead(BUTTON_PIN) ? "Released" : "Pressed");
    }
    else if (strcmp(argv[0], "uptime") == 0) {
        Serial.print("Uptime: ");
        printUptime();
    }
    else if (strcmp(argv[0], "echo") == 0) {
        if (argc > 1) {
            echoEnabled = (strcmp(argv[1], "on") == 0);
        }
        Serial.print("Echo: ");
        Serial.println(echoEnabled ? "ON" : "OFF");
    }
    else if (strcmp(argv[0], "clear") == 0) {
        Serial.print("\033[2J\033[H");  // ANSI clear
    }
    else if (strcmp(argv[0], "reboot") == 0) {
        Serial.println("Rebooting...");
        delay(100);
        asm volatile ("jmp 0");  // 소프트 리셋
    }
    else {
        Serial.print("Unknown command: ");
        Serial.println(argv[0]);
        Serial.println("Type 'help' for available commands");
    }
}

void loop() {
    while (Serial.available()) {
        char c = Serial.read();

        // Enter 키
        if (c == '\r' || c == '\n') {
            if (echoEnabled) Serial.println();

            cmdBuffer[cmdIndex] = '\0';
            processCommand(cmdBuffer);
            cmdIndex = 0;

            printPrompt();
        }
        // Backspace
        else if (c == '\b' || c == 127) {
            if (cmdIndex > 0) {
                cmdIndex--;
                if (echoEnabled) {
                    Serial.print("\b \b");  // 지우기
                }
            }
        }
        // 일반 문자
        else if (cmdIndex < MAX_CMD_LEN - 1 && c >= 32) {
            cmdBuffer[cmdIndex++] = c;
            if (echoEnabled) Serial.print(c);
        }
    }
}
```

---

## 연습 문제

### 연습 1: 온도 로거
시리얼로 "LOG START"를 입력하면 1초마다 가상의 온도 데이터를 출력하고, "LOG STOP"으로 중지하세요.

### 연습 2: LED 시퀀서
시리얼로 LED 패턴 시퀀스를 입력받아 순차 실행하세요.
예: "SEQ 1,3,5,15,0" → 각 패턴 500ms씩 실행

### 연습 3: 계산기 확장
괄호와 여러 연산자를 지원하는 계산기로 확장하세요.

### 연습 4: 이진 통신
PC에서 바이트 시퀀스를 보내면 해석하여 LED를 제어하는 바이너리 프로토콜을 구현하세요.

---

## 핵심 개념 정리

| 함수 | 설명 |
|------|------|
| `Serial.begin()` | 시리얼 통신 초기화 |
| `Serial.print()` | 데이터 출력 |
| `Serial.available()` | 수신 대기 바이트 수 |
| `Serial.read()` | 1바이트 읽기 |
| `Serial.readStringUntil()` | 구분자까지 읽기 |

| 개념 | 설명 |
|------|------|
| Baud Rate | 초당 전송 비트 수 |
| UART | 비동기 시리얼 통신 |
| TX/RX | 송신/수신 핀 |
| 버퍼 | 수신 데이터 임시 저장소 |
| 프로토콜 | 통신 규약 |

---

## 임베디드 C 기초 완료!

4개의 임베디드 기초 문서를 완료했습니다:

1. ✅ 임베디드 기초 - 개념, Arduino 환경
2. ✅ 비트 연산 심화 - 마스킹, 레지스터, volatile
3. ✅ GPIO 제어 - LED, 버튼, 디바운싱
4. ✅ 시리얼 통신 - UART, 파싱, 디버깅

### 다음 학습 추천

기초를 마쳤다면 다음 주제로 확장할 수 있습니다:

- **타이머와 PWM**: LED 밝기 조절, 서보 모터
- **인터럽트**: 외부/타이머 인터럽트, ISR
- **ADC와 센서**: 아날로그 입력, 온도/조도 센서
- **I2C/SPI 통신**: 센서/디스플레이 연결
- **RTOS**: FreeRTOS 기초

### 실습 플랫폼

- **Wokwi**: https://wokwi.com (무료 시뮬레이터)
- **TinkerCAD**: https://tinkercad.com/circuits
- **Arduino 공식**: https://www.arduino.cc
