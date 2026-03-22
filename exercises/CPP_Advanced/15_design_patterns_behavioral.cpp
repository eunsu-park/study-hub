// Exercise 15: Behavioral Design Patterns
// Practice Observer, Strategy, Command, State, and Visitor patterns.
// Compile: g++ -std=c++20 -Wall -Wextra -o ex15 15_design_patterns_behavioral.cpp && ./ex15

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <cassert>
#include <sstream>

// TODO 1: Implement the Observer pattern for a stock price tracker.
// - StockExchange: subject that holds stock prices and notifies observers
// - Observer interface with update(symbol, price)
// - PriceLogger: logs all price changes
// - ThresholdAlert: alerts when price crosses a threshold

// class Observer { ... };
// class StockExchange { ... };
// class PriceLogger : public Observer { ... };
// class ThresholdAlert : public Observer { ... };

// TODO 2: Implement the Strategy pattern for text formatting.
// - Formatter interface with format(text) -> string
// - UpperCaseFormatter, MarkdownFormatter, HTMLFormatter
// - TextEditor class that uses a Formatter strategy

// class Formatter { ... };
// class TextEditor { ... };

// TODO 3: Implement the Command pattern for an undo/redo system.
// - Command interface with execute() and undo()
// - InsertCommand, DeleteCommand for a text buffer
// - CommandHistory with undo() and redo()

// class Command { ... };
// class TextBuffer { ... };
// class CommandHistory { ... };

// TODO 4: Implement the State pattern for a vending machine.
// States: Idle, CoinInserted, Dispensing, SoldOut
// Actions: insert_coin(), select_item(), dispense()
// Each state handles actions differently.

// class VendingState { ... };
// class VendingMachine { ... };

// TODO 5: Implement the Visitor pattern for an AST (abstract syntax tree).
// Nodes: NumberNode, BinaryOpNode (add, mul), UnaryOpNode (negate)
// Visitors: Evaluator (computes result), Printer (prints expression)

// class ASTNode { ... };
// class Visitor { ... };

int main() {
    std::cout << "=== Exercise 15: Behavioral Design Patterns ===\n\n";

    // Test 1: Observer
    // StockExchange exchange;
    // auto logger = std::make_shared<PriceLogger>();
    // auto alert = std::make_shared<ThresholdAlert>("AAPL", 150.0);
    // exchange.subscribe(logger);
    // exchange.subscribe(alert);
    // exchange.set_price("AAPL", 145.0);
    // exchange.set_price("AAPL", 155.0);  // should trigger alert
    // exchange.set_price("GOOG", 2800.0);
    // std::cout << "Test 1: check Observer output above\n";

    // Test 2: Strategy
    // TextEditor editor;
    // editor.set_formatter(std::make_unique<UpperCaseFormatter>());
    // assert(editor.format("hello world") == "HELLO WORLD");
    // editor.set_formatter(std::make_unique<HTMLFormatter>());
    // assert(editor.format("hello") == "<p>hello</p>");
    // std::cout << "Test 2 passed: Strategy\n";

    // Test 3: Command (undo/redo)
    // TextBuffer buf;
    // CommandHistory history;
    // history.execute(std::make_unique<InsertCommand>(buf, 0, "Hello"));
    // history.execute(std::make_unique<InsertCommand>(buf, 5, " World"));
    // assert(buf.text() == "Hello World");
    // history.undo();
    // assert(buf.text() == "Hello");
    // history.redo();
    // assert(buf.text() == "Hello World");
    // std::cout << "Test 3 passed: Command undo/redo\n";

    // Test 4: State
    // VendingMachine vm(3);  // 3 items
    // vm.insert_coin();
    // vm.select_item();      // dispenses
    // vm.select_item();      // no coin inserted
    // std::cout << "Test 4: check State output above\n";

    // Test 5: Visitor
    // auto expr = std::make_unique<BinaryOpNode>(
    //     '+',
    //     std::make_unique<NumberNode>(3),
    //     std::make_unique<BinaryOpNode>('*',
    //         std::make_unique<NumberNode>(4),
    //         std::make_unique<NumberNode>(5)));
    // Evaluator eval;
    // double result = eval.visit(*expr);
    // assert(result == 23.0);  // 3 + (4 * 5)
    // Printer printer;
    // std::string printed = printer.visit(*expr);
    // std::cout << "Expression: " << printed << " = " << result << '\n';
    // std::cout << "Test 5 passed: Visitor\n";

    std::cout << "Uncomment tests as you implement each pattern.\n";
    return 0;
}
