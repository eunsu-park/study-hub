/*
 * Exercises for Lesson 18: Design Patterns
 * Topic: CPP
 * Compile: g++ -std=c++17 -Wall -Wextra -o ex18 18_design_patterns.cpp
 */
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <stack>
#include <algorithm>
using namespace std;

// === Exercise 1: Logging System (Singleton + Strategy) ===
// Problem: Singleton Logger with pluggable output strategies (Console, File,
//          Null) and log levels (DEBUG, INFO, WARNING, ERROR).

enum class LogLevel { DEBUG, INFO, WARNING, ERR };

string logLevelStr(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERR:     return "ERROR";
    }
    return "UNKNOWN";
}

// Strategy interface for log output
class ILogStrategy {
public:
    virtual void write(const string& message) = 0;
    virtual ~ILogStrategy() = default;
};

class ConsoleStrategy : public ILogStrategy {
public:
    void write(const string& message) override {
        cout << "    [Console] " << message << endl;
    }
};

class StringBufferStrategy : public ILogStrategy {
    // Writes to an internal buffer (useful for testing or in-memory logs)
    ostringstream buffer_;
public:
    void write(const string& message) override {
        buffer_ << message << "\n";
    }
    string contents() const { return buffer_.str(); }
};

class NullStrategy : public ILogStrategy {
public:
    void write(const string&) override {
        // Intentionally discards all messages
    }
};

// Singleton Logger
class SingletonLogger {
    unique_ptr<ILogStrategy> strategy_;
    LogLevel minLevel_ = LogLevel::DEBUG;

    // Private constructor for singleton
    SingletonLogger() : strategy_(make_unique<ConsoleStrategy>()) {}

public:
    // Delete copy/move to enforce singleton
    SingletonLogger(const SingletonLogger&) = delete;
    SingletonLogger& operator=(const SingletonLogger&) = delete;

    static SingletonLogger& getInstance() {
        static SingletonLogger instance;
        return instance;
    }

    void setStrategy(unique_ptr<ILogStrategy> strategy) {
        strategy_ = std::move(strategy);
    }

    void setMinLevel(LogLevel level) {
        minLevel_ = level;
    }

    void log(LogLevel level, const string& message) {
        if (level < minLevel_) return;

        string formatted = "[" + logLevelStr(level) + "] " + message;
        strategy_->write(formatted);
    }
};

void exercise_1() {
    cout << "=== Exercise 1: Logging System (Singleton + Strategy) ===" << endl;

    auto& logger = SingletonLogger::getInstance();

    // Use console strategy
    logger.setStrategy(make_unique<ConsoleStrategy>());
    logger.setMinLevel(LogLevel::DEBUG);

    logger.log(LogLevel::DEBUG, "Initializing application");
    logger.log(LogLevel::INFO, "Server started on port 8080");
    logger.log(LogLevel::WARNING, "Disk usage at 90%");
    logger.log(LogLevel::ERR, "Failed to connect to cache");

    // Switch to higher minimum level
    cout << "\n    (Setting min level to WARNING)" << endl;
    logger.setMinLevel(LogLevel::WARNING);
    logger.log(LogLevel::DEBUG, "This should NOT appear");
    logger.log(LogLevel::INFO, "This should NOT appear either");
    logger.log(LogLevel::WARNING, "This WARNING appears");
    logger.log(LogLevel::ERR, "This ERROR appears");

    // Switch to string buffer strategy
    auto bufferStrategy = make_unique<StringBufferStrategy>();
    auto* bufferPtr = bufferStrategy.get();  // Keep reference before moving
    logger.setStrategy(std::move(bufferStrategy));
    logger.setMinLevel(LogLevel::DEBUG);

    logger.log(LogLevel::INFO, "Buffered message 1");
    logger.log(LogLevel::INFO, "Buffered message 2");

    cout << "\n    Buffered contents:" << endl;
    cout << "    " << bufferPtr->contents();

    // Reset to console for subsequent exercises
    logger.setStrategy(make_unique<ConsoleStrategy>());
}

// === Exercise 2: UI Components (Composite + Decorator) ===
// Problem: UI component hierarchy with composite containers and decorators
//          for borders and scrollbars.

class UIComponent {
public:
    virtual void render(int indent = 0) const = 0;
    virtual ~UIComponent() = default;

protected:
    static string indentStr(int indent) {
        return string(indent * 2, ' ');
    }
};

// Leaf components
class Button : public UIComponent {
    string label_;
public:
    explicit Button(const string& label) : label_(label) {}
    void render(int indent = 0) const override {
        cout << indentStr(indent) << "[Button: " << label_ << "]" << endl;
    }
};

class TextBox : public UIComponent {
    string placeholder_;
public:
    explicit TextBox(const string& placeholder) : placeholder_(placeholder) {}
    void render(int indent = 0) const override {
        cout << indentStr(indent) << "[TextBox: " << placeholder_ << "]" << endl;
    }
};

// Composite: can contain child components
class Panel : public UIComponent {
    string name_;
    vector<shared_ptr<UIComponent>> children_;
public:
    explicit Panel(const string& name) : name_(name) {}

    void add(shared_ptr<UIComponent> child) {
        children_.push_back(std::move(child));
    }

    void render(int indent = 0) const override {
        cout << indentStr(indent) << "[Panel: " << name_ << "]" << endl;
        for (const auto& child : children_) {
            child->render(indent + 1);
        }
    }
};

// Decorator base
class UIDecorator : public UIComponent {
protected:
    shared_ptr<UIComponent> wrapped_;
public:
    explicit UIDecorator(shared_ptr<UIComponent> component)
        : wrapped_(std::move(component)) {}
};

class BorderDecorator : public UIDecorator {
    string style_;
public:
    BorderDecorator(shared_ptr<UIComponent> component, const string& style = "solid")
        : UIDecorator(std::move(component)), style_(style) {}

    void render(int indent = 0) const override {
        cout << indentStr(indent) << "╔══ Border(" << style_ << ") ══╗" << endl;
        wrapped_->render(indent + 1);
        cout << indentStr(indent) << "╚════════════════════╝" << endl;
    }
};

class ScrollDecorator : public UIDecorator {
public:
    explicit ScrollDecorator(shared_ptr<UIComponent> component)
        : UIDecorator(std::move(component)) {}

    void render(int indent = 0) const override {
        cout << indentStr(indent) << "┌── Scrollable ──┐" << endl;
        wrapped_->render(indent + 1);
        cout << indentStr(indent) << "└── [scrollbar] ─┘" << endl;
    }
};

void exercise_2() {
    cout << "\n=== Exercise 2: UI Components (Composite + Decorator) ===" << endl;

    // Build a UI tree
    auto mainPanel = make_shared<Panel>("MainWindow");

    auto toolbar = make_shared<Panel>("Toolbar");
    toolbar->add(make_shared<Button>("Save"));
    toolbar->add(make_shared<Button>("Load"));
    toolbar->add(make_shared<Button>("Exit"));

    auto form = make_shared<Panel>("Form");
    form->add(make_shared<TextBox>("Enter name..."));
    form->add(make_shared<TextBox>("Enter email..."));
    form->add(make_shared<Button>("Submit"));

    mainPanel->add(toolbar);
    mainPanel->add(form);

    cout << "  Raw UI tree:" << endl;
    mainPanel->render(2);

    // Decorate: add border to form, scrollbar to main panel
    auto borderedForm = make_shared<BorderDecorator>(form, "dashed");
    auto scrollableMain = make_shared<ScrollDecorator>(mainPanel);

    cout << "\n  With decorators:" << endl;
    auto decoratedWindow = make_shared<Panel>("DecoratedApp");
    decoratedWindow->add(toolbar);
    decoratedWindow->add(borderedForm);

    auto finalUI = make_shared<ScrollDecorator>(decoratedWindow);
    finalUI->render(2);
}

// === Exercise 3: Document Editor (Command + Memento) ===
// Problem: Text editor with insert/delete commands, unlimited undo/redo,
//          and snapshot save/restore.

// Memento: captures document state
class DocumentMemento {
    friend class Document;
    string state_;
    explicit DocumentMemento(const string& state) : state_(state) {}
public:
    string getDescription() const {
        return state_.substr(0, 30) + (state_.size() > 30 ? "..." : "");
    }
};

// Document (Originator)
class Document {
    string content_;
public:
    const string& content() const { return content_; }

    void insert(size_t pos, const string& text) {
        if (pos > content_.size()) pos = content_.size();
        content_.insert(pos, text);
    }

    void erase(size_t pos, size_t len) {
        if (pos < content_.size()) {
            content_.erase(pos, len);
        }
    }

    DocumentMemento createMemento() const {
        return DocumentMemento(content_);
    }

    void restore(const DocumentMemento& memento) {
        content_ = memento.state_;
    }
};

// Command interface
class ICommand {
public:
    virtual void execute() = 0;
    virtual void undo() = 0;
    virtual string description() const = 0;
    virtual ~ICommand() = default;
};

class InsertCommand : public ICommand {
    Document& doc_;
    size_t pos_;
    string text_;
public:
    InsertCommand(Document& doc, size_t pos, const string& text)
        : doc_(doc), pos_(pos), text_(text) {}

    void execute() override { doc_.insert(pos_, text_); }
    void undo() override { doc_.erase(pos_, text_.size()); }
    string description() const override {
        return "Insert \"" + text_ + "\" at " + to_string(pos_);
    }
};

class DeleteCommand : public ICommand {
    Document& doc_;
    size_t pos_;
    size_t len_;
    string deleted_;  // Saved for undo
public:
    DeleteCommand(Document& doc, size_t pos, size_t len)
        : doc_(doc), pos_(pos), len_(len) {}

    void execute() override {
        // Save deleted text for undo
        deleted_ = doc_.content().substr(pos_, len_);
        doc_.erase(pos_, len_);
    }

    void undo() override {
        doc_.insert(pos_, deleted_);
    }

    string description() const override {
        return "Delete " + to_string(len_) + " chars at " + to_string(pos_);
    }
};

// Editor with undo/redo and snapshots
class Editor {
    Document doc_;
    stack<unique_ptr<ICommand>> undoStack_;
    stack<unique_ptr<ICommand>> redoStack_;
    vector<DocumentMemento> snapshots_;

public:
    void execute(unique_ptr<ICommand> cmd) {
        cmd->execute();
        undoStack_.push(std::move(cmd));
        // Clear redo stack on new command (standard undo/redo behavior)
        while (!redoStack_.empty()) redoStack_.pop();
    }

    void undo() {
        if (undoStack_.empty()) {
            cout << "    Nothing to undo" << endl;
            return;
        }
        auto& cmd = undoStack_.top();
        cout << "    Undo: " << cmd->description() << endl;
        cmd->undo();
        redoStack_.push(std::move(undoStack_.top()));
        undoStack_.pop();
    }

    void redo() {
        if (redoStack_.empty()) {
            cout << "    Nothing to redo" << endl;
            return;
        }
        auto& cmd = redoStack_.top();
        cout << "    Redo: " << cmd->description() << endl;
        cmd->execute();
        undoStack_.push(std::move(redoStack_.top()));
        redoStack_.pop();
    }

    void saveSnapshot() {
        snapshots_.push_back(doc_.createMemento());
        cout << "    Snapshot saved (#" << snapshots_.size() << ")" << endl;
    }

    void restoreSnapshot(size_t index) {
        if (index < snapshots_.size()) {
            doc_.restore(snapshots_[index]);
            cout << "    Restored snapshot #" << (index + 1) << endl;
        }
    }

    const string& content() const { return doc_.content(); }

    // Convenience methods
    void insert(size_t pos, const string& text) {
        execute(make_unique<InsertCommand>(doc_, pos, text));
    }

    void del(size_t pos, size_t len) {
        execute(make_unique<DeleteCommand>(doc_, pos, len));
    }
};

void exercise_3() {
    cout << "\n=== Exercise 3: Document Editor (Command + Memento) ===" << endl;

    Editor editor;

    // Build up a document
    editor.insert(0, "Hello");
    cout << "    Content: \"" << editor.content() << "\"" << endl;

    editor.insert(5, " World");
    cout << "    Content: \"" << editor.content() << "\"" << endl;

    editor.insert(5, " Beautiful");
    cout << "    Content: \"" << editor.content() << "\"" << endl;

    // Save a snapshot
    editor.saveSnapshot();

    // Delete "Beautiful "
    editor.del(5, 10);
    cout << "    Content: \"" << editor.content() << "\"" << endl;

    // Undo the delete
    cout << endl;
    editor.undo();
    cout << "    Content: \"" << editor.content() << "\"" << endl;

    // Undo the "Beautiful" insert
    editor.undo();
    cout << "    Content: \"" << editor.content() << "\"" << endl;

    // Redo
    editor.redo();
    cout << "    Content: \"" << editor.content() << "\"" << endl;

    // More edits
    editor.insert(editor.content().size(), "!");
    cout << "    Content: \"" << editor.content() << "\"" << endl;

    // Restore snapshot
    cout << endl;
    editor.restoreSnapshot(0);
    cout << "    Content: \"" << editor.content() << "\"" << endl;
}

// === Exercise 4: Plugin System (Factory + Type Erasure) ===
// Problem: Register/unregister plugins at runtime, execute through a
//          common interface using type erasure.

// Type-erased Plugin wrapper
class Plugin {
    // Type erasure: store any callable object that matches execute()
    struct Concept {
        virtual void execute() = 0;
        virtual string name() const = 0;
        virtual ~Concept() = default;
    };

    template<typename T>
    struct Model : Concept {
        T plugin_;
        explicit Model(T plugin) : plugin_(std::move(plugin)) {}
        void execute() override { plugin_.execute(); }
        string name() const override { return plugin_.name(); }
    };

    unique_ptr<Concept> impl_;

public:
    template<typename T>
    Plugin(T plugin) : impl_(make_unique<Model<T>>(std::move(plugin))) {}

    void execute() { impl_->execute(); }
    string name() const { return impl_->name(); }
};

// Plugin manager with registration
class PluginManager {
    unordered_map<string, Plugin> plugins_;

public:
    template<typename T>
    void registerPlugin(const string& id, T plugin) {
        plugins_.emplace(id, Plugin(std::move(plugin)));
        cout << "    Registered: " << id << endl;
    }

    void unregister(const string& id) {
        plugins_.erase(id);
        cout << "    Unregistered: " << id << endl;
    }

    void execute(const string& id) {
        auto it = plugins_.find(id);
        if (it != plugins_.end()) {
            cout << "    Executing " << it->second.name() << ": ";
            it->second.execute();
        } else {
            cout << "    Plugin not found: " << id << endl;
        }
    }

    void executeAll() {
        for (auto& [id, plugin] : plugins_) {
            cout << "    Executing " << plugin.name() << ": ";
            plugin.execute();
        }
    }
};

// Example plugins (different types, unified through type erasure)
struct GreeterPlugin {
    void execute() { cout << "Hello from GreeterPlugin!" << endl; }
    string name() const { return "Greeter"; }
};

struct CounterPlugin {
    int count = 0;
    void execute() { cout << "Count: " << ++count << endl; }
    string name() const { return "Counter"; }
};

struct TransformPlugin {
    string input;
    void execute() {
        string upper = input;
        transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
        cout << "\"" << input << "\" -> \"" << upper << "\"" << endl;
    }
    string name() const { return "Transform"; }
};

void exercise_4() {
    cout << "\n=== Exercise 4: Plugin System (Factory + Type Erasure) ===" << endl;

    PluginManager manager;

    // Register plugins of different types
    manager.registerPlugin("greeter", GreeterPlugin{});
    manager.registerPlugin("counter", CounterPlugin{});
    manager.registerPlugin("transform", TransformPlugin{"hello world"});

    // Execute individual plugins
    cout << endl;
    manager.execute("greeter");
    manager.execute("counter");
    manager.execute("counter");
    manager.execute("transform");

    // Execute all
    cout << "\n    --- Execute all ---" << endl;
    manager.executeAll();

    // Unregister and try again
    cout << endl;
    manager.unregister("greeter");
    manager.execute("greeter");  // Not found
}

int main() {
    exercise_1();
    exercise_2();
    exercise_3();
    exercise_4();
    cout << "\nAll exercises completed!" << endl;
    return 0;
}
