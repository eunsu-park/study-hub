/*
 * Exercises for Lesson 13: Exceptions and File I/O
 * Topic: CPP
 * Compile: g++ -std=c++17 -Wall -Wextra -o ex13 13_exceptions_and_file_io.cpp
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <ctime>
#include <iomanip>
#include <stdexcept>
#include <memory>
using namespace std;

// === Exercise 1: Log File Class ===
// Problem: Write a Logger class that records messages with date/time.
//          Supports log levels and writes to both console and file.

class Logger {
public:
    enum class Level { DEBUG, INFO, WARNING, ERR };

private:
    ofstream file_;
    Level minLevel_;
    bool consoleOutput_;

    // Convert level to string label
    static string levelToString(Level level) {
        switch (level) {
            case Level::DEBUG:   return "DEBUG";
            case Level::INFO:    return "INFO";
            case Level::WARNING: return "WARNING";
            case Level::ERR:     return "ERROR";
            default:             return "UNKNOWN";
        }
    }

    // Get current timestamp as a formatted string
    static string timestamp() {
        time_t now = time(nullptr);
        tm* local = localtime(&now);
        ostringstream oss;
        oss << put_time(local, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

public:
    Logger(const string& filename, Level minLevel = Level::DEBUG,
           bool consoleOutput = true)
        : minLevel_(minLevel), consoleOutput_(consoleOutput)
    {
        file_.open(filename, ios::app);
        if (!file_.is_open()) {
            throw runtime_error("Failed to open log file: " + filename);
        }
    }

    ~Logger() {
        if (file_.is_open()) {
            file_.close();
        }
    }

    // Non-copyable (owns a file resource)
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    void log(Level level, const string& message) {
        if (level < minLevel_) return;

        string entry = "[" + timestamp() + "] ["
                      + levelToString(level) + "] " + message;

        // Write to file
        file_ << entry << endl;

        // Optionally write to console
        if (consoleOutput_) {
            cout << "  " << entry << endl;
        }
    }

    // Convenience methods
    void debug(const string& msg)   { log(Level::DEBUG, msg); }
    void info(const string& msg)    { log(Level::INFO, msg); }
    void warning(const string& msg) { log(Level::WARNING, msg); }
    void error(const string& msg)   { log(Level::ERR, msg); }
};

void exercise_1() {
    cout << "=== Exercise 1: Log File Class ===" << endl;

    try {
        Logger logger("/tmp/cpp_exercise_log.txt", Logger::Level::DEBUG);

        logger.debug("Application starting");
        logger.info("Configuration loaded successfully");
        logger.warning("Memory usage at 85%");
        logger.error("Failed to connect to database");
        logger.info("Retrying connection...");
        logger.info("Connection established");

        cout << "  Log written to /tmp/cpp_exercise_log.txt" << endl;
    } catch (const exception& e) {
        cerr << "  Logger error: " << e.what() << endl;
    }
}

// === Exercise 2: Exception Hierarchy ===
// Problem: Design a database-related exception class hierarchy with
//          ConnectionError, QueryError, AuthenticationError, etc.

// Base database exception
class DatabaseError : public runtime_error {
protected:
    int errorCode_;
public:
    DatabaseError(const string& msg, int code = 0)
        : runtime_error(msg), errorCode_(code) {}

    int errorCode() const { return errorCode_; }

    // Virtual method for structured error reporting
    virtual string details() const {
        return string("DatabaseError[") + to_string(errorCode_) + "]: " + what();
    }
};

class ConnectionError : public DatabaseError {
    string host_;
    int port_;
public:
    ConnectionError(const string& host, int port, const string& reason)
        : DatabaseError("Connection to " + host + ":" + to_string(port)
                       + " failed: " + reason, 1001),
          host_(host), port_(port) {}

    string details() const override {
        return "ConnectionError -> host=" + host_ + ", port="
               + to_string(port_) + " | " + what();
    }
};

class AuthenticationError : public DatabaseError {
    string username_;
public:
    AuthenticationError(const string& username, const string& reason)
        : DatabaseError("Authentication failed for '" + username + "': " + reason, 1002),
          username_(username) {}

    string details() const override {
        return "AuthError -> user=" + username_ + " | " + what();
    }
};

class QueryError : public DatabaseError {
    string query_;
public:
    QueryError(const string& query, const string& reason)
        : DatabaseError("Query failed: " + reason, 1003),
          query_(query) {}

    string details() const override {
        return "QueryError -> query=\"" + query_ + "\" | " + what();
    }
};

class TransactionError : public DatabaseError {
public:
    explicit TransactionError(const string& reason)
        : DatabaseError("Transaction error: " + reason, 1004) {}
};

// Simulate database operations that throw various exceptions
void connectToDatabase(const string& host, int port) {
    throw ConnectionError(host, port, "Connection refused");
}

void authenticate(const string& user) {
    throw AuthenticationError(user, "Invalid credentials");
}

void executeQuery(const string& sql) {
    throw QueryError(sql, "Syntax error near 'SELCT'");
}

void exercise_2() {
    cout << "\n=== Exercise 2: Exception Hierarchy ===" << endl;

    // Test each exception type with polymorphic catch
    auto testOp = [](const string& label, auto operation) {
        try {
            operation();
        } catch (const ConnectionError& e) {
            cout << "  Caught ConnectionError: " << e.details() << endl;
        } catch (const AuthenticationError& e) {
            cout << "  Caught AuthError: " << e.details() << endl;
        } catch (const QueryError& e) {
            cout << "  Caught QueryError: " << e.details() << endl;
        } catch (const DatabaseError& e) {
            // Fallback: catches any DatabaseError not caught above
            cout << "  Caught DatabaseError: " << e.details() << endl;
        }
    };

    testOp("connect", []{ connectToDatabase("db.example.com", 5432); });
    testOp("auth",    []{ authenticate("admin"); });
    testOp("query",   []{ executeQuery("SELCT * FROM users"); });

    // Demonstrate catching at the base level
    cout << "\n  Catching at base DatabaseError level:" << endl;
    try {
        throw TransactionError("Deadlock detected");
    } catch (const DatabaseError& e) {
        cout << "  " << e.details() << endl;
        cout << "  Error code: " << e.errorCode() << endl;
    }
}

// === Exercise 3: Simple JSON Parser ===
// Problem: Parse simple key-value JSON like {"name": "Alice", "age": 25}.
//          Supports string and numeric values.

class SimpleJsonParser {
public:
    using JsonObject = map<string, string>;

    // Parse a simple flat JSON object: {"key": "value", "key2": 123}
    static JsonObject parse(const string& json) {
        JsonObject result;
        size_t pos = 0;

        skipWhitespace(json, pos);
        expect(json, pos, '{');

        // Handle empty object
        skipWhitespace(json, pos);
        if (pos < json.size() && json[pos] == '}') {
            pos++;
            return result;
        }

        while (pos < json.size()) {
            skipWhitespace(json, pos);

            // Parse key (must be a string)
            string key = parseString(json, pos);

            skipWhitespace(json, pos);
            expect(json, pos, ':');
            skipWhitespace(json, pos);

            // Parse value (string or number)
            string value;
            if (json[pos] == '"') {
                value = parseString(json, pos);
            } else {
                value = parseNumber(json, pos);
            }

            result[key] = value;

            skipWhitespace(json, pos);
            if (pos < json.size() && json[pos] == ',') {
                pos++;  // Skip comma, continue
            } else {
                break;
            }
        }

        skipWhitespace(json, pos);
        expect(json, pos, '}');

        return result;
    }

    // Convert JsonObject back to string
    static string stringify(const JsonObject& obj) {
        ostringstream oss;
        oss << "{";
        bool first = true;
        for (const auto& [key, value] : obj) {
            if (!first) oss << ", ";
            first = false;
            oss << "\"" << key << "\": ";

            // Check if value looks like a number
            bool isNumber = !value.empty() &&
                (isdigit(value[0]) || value[0] == '-');
            if (isNumber) {
                oss << value;
            } else {
                oss << "\"" << value << "\"";
            }
        }
        oss << "}";
        return oss.str();
    }

private:
    static void skipWhitespace(const string& s, size_t& pos) {
        while (pos < s.size() && isspace(s[pos])) pos++;
    }

    static void expect(const string& s, size_t& pos, char ch) {
        if (pos >= s.size() || s[pos] != ch) {
            throw runtime_error(
                string("Expected '") + ch + "' at position " + to_string(pos));
        }
        pos++;
    }

    static string parseString(const string& s, size_t& pos) {
        expect(s, pos, '"');
        string result;
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\' && pos + 1 < s.size()) {
                pos++;  // Skip escape character
                switch (s[pos]) {
                    case '"':  result += '"';  break;
                    case '\\': result += '\\'; break;
                    case 'n':  result += '\n'; break;
                    case 't':  result += '\t'; break;
                    default:   result += s[pos]; break;
                }
            } else {
                result += s[pos];
            }
            pos++;
        }
        expect(s, pos, '"');
        return result;
    }

    static string parseNumber(const string& s, size_t& pos) {
        size_t start = pos;
        if (pos < s.size() && s[pos] == '-') pos++;
        while (pos < s.size() && (isdigit(s[pos]) || s[pos] == '.')) {
            pos++;
        }
        if (pos == start) {
            throw runtime_error("Expected number at position " + to_string(pos));
        }
        return s.substr(start, pos - start);
    }
};

void exercise_3() {
    cout << "\n=== Exercise 3: Simple JSON Parser ===" << endl;

    // Test 1: Basic key-value pairs
    string json1 = R"({"name": "Alice", "age": 25, "city": "Seoul"})";
    cout << "  Input:  " << json1 << endl;

    try {
        auto obj1 = SimpleJsonParser::parse(json1);
        cout << "  Parsed:" << endl;
        for (const auto& [key, value] : obj1) {
            cout << "    " << key << " = " << value << endl;
        }
        cout << "  Stringify: " << SimpleJsonParser::stringify(obj1) << endl;
    } catch (const exception& e) {
        cerr << "  Parse error: " << e.what() << endl;
    }

    // Test 2: Numbers and negative values
    string json2 = R"({"x": 10, "y": -20.5, "label": "point"})";
    cout << "\n  Input:  " << json2 << endl;

    try {
        auto obj2 = SimpleJsonParser::parse(json2);
        for (const auto& [key, value] : obj2) {
            cout << "    " << key << " = " << value << endl;
        }
    } catch (const exception& e) {
        cerr << "  Parse error: " << e.what() << endl;
    }

    // Test 3: Empty object
    string json3 = "{}";
    try {
        auto obj3 = SimpleJsonParser::parse(json3);
        cout << "\n  Empty object parsed: " << obj3.size() << " entries" << endl;
    } catch (const exception& e) {
        cerr << "  Parse error: " << e.what() << endl;
    }

    // Test 4: Malformed JSON (should throw)
    string json4 = R"({"broken": )";
    try {
        SimpleJsonParser::parse(json4);
        cout << "  Should not reach here" << endl;
    } catch (const exception& e) {
        cout << "\n  Expected parse error: " << e.what() << endl;
    }
}

int main() {
    exercise_1();
    exercise_2();
    exercise_3();
    cout << "\nAll exercises completed!" << endl;
    return 0;
}
