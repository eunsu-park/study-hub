/*
 * Exercises for Lesson 14: Smart Pointers and Memory Management
 * Topic: CPP
 * Compile: g++ -std=c++17 -Wall -Wextra -o ex14 14_smart_pointers_memory.cpp
 */
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <queue>
using namespace std;

// === Exercise 1: Resource Manager ===
// Problem: Implement a class that manages various resources (file, network
//          connection, etc.) using unique_ptr with custom deleters.

// Abstract resource interface
class IResource {
public:
    virtual void use() = 0;
    virtual string name() const = 0;
    virtual ~IResource() = default;
};

// Simulated file resource
class FileResource : public IResource {
    string path_;
    bool open_ = false;
public:
    explicit FileResource(const string& path) : path_(path), open_(true) {
        cout << "    [File] Opened: " << path_ << endl;
    }
    ~FileResource() override {
        if (open_) {
            cout << "    [File] Closed: " << path_ << endl;
            open_ = false;
        }
    }
    void use() override {
        cout << "    [File] Reading from " << path_ << endl;
    }
    string name() const override { return "File:" + path_; }
};

// Simulated network connection resource
class NetworkResource : public IResource {
    string host_;
    int port_;
public:
    NetworkResource(const string& host, int port) : host_(host), port_(port) {
        cout << "    [Net] Connected to " << host_ << ":" << port_ << endl;
    }
    ~NetworkResource() override {
        cout << "    [Net] Disconnected from " << host_ << ":" << port_ << endl;
    }
    void use() override {
        cout << "    [Net] Sending data to " << host_ << endl;
    }
    string name() const override {
        return "Net:" + host_ + ":" + to_string(port_);
    }
};

// ResourceManager: owns multiple resources via unique_ptr
class ResourceManager {
    vector<unique_ptr<IResource>> resources_;
public:
    // Take ownership of a resource
    void acquire(unique_ptr<IResource> resource) {
        cout << "  ResourceManager acquired: " << resource->name() << endl;
        resources_.push_back(std::move(resource));
    }

    // Use all resources
    void useAll() {
        for (auto& r : resources_) {
            r->use();
        }
    }

    size_t count() const { return resources_.size(); }

    // Resources are automatically released when ResourceManager is destroyed
    // or when the vector is cleared, thanks to unique_ptr
    ~ResourceManager() {
        cout << "  ResourceManager destroying " << resources_.size()
             << " resource(s)..." << endl;
    }
};

void exercise_1() {
    cout << "=== Exercise 1: Resource Manager ===" << endl;

    {
        ResourceManager manager;

        // Create resources and transfer ownership to manager
        manager.acquire(make_unique<FileResource>("/etc/config.json"));
        manager.acquire(make_unique<NetworkResource>("api.example.com", 443));
        manager.acquire(make_unique<FileResource>("/var/log/app.log"));

        cout << "\n  Using " << manager.count() << " resources:" << endl;
        manager.useAll();

        cout << "\n  Leaving scope..." << endl;
    }
    // All resources are automatically cleaned up here
    cout << "  (All resources released)" << endl;
}

// === Exercise 2: Graph with shared_ptr and weak_ptr ===
// Problem: Implement a graph where nodes are connected using shared_ptr
//          for ownership and weak_ptr to break circular references.

class GraphNode : public enable_shared_from_this<GraphNode> {
    string id_;
    // Use weak_ptr for neighbors to avoid circular ownership.
    // If we used shared_ptr, two nodes pointing at each other would
    // create a reference cycle and never be freed.
    vector<weak_ptr<GraphNode>> neighbors_;

public:
    explicit GraphNode(const string& id) : id_(id) {
        cout << "    Node '" << id_ << "' created" << endl;
    }

    ~GraphNode() {
        cout << "    Node '" << id_ << "' destroyed" << endl;
    }

    const string& id() const { return id_; }

    void addNeighbor(shared_ptr<GraphNode> neighbor) {
        neighbors_.push_back(neighbor);  // Stores as weak_ptr
        cout << "    Edge: " << id_ << " -> " << neighbor->id() << endl;
    }

    // Visit all reachable neighbors (locking weak_ptrs)
    void printNeighbors() const {
        cout << "    " << id_ << " neighbors: ";
        for (const auto& wp : neighbors_) {
            if (auto sp = wp.lock()) {
                cout << sp->id() << " ";
            } else {
                cout << "(expired) ";
            }
        }
        cout << endl;
    }

    // BFS traversal from this node
    vector<string> bfs() const {
        vector<string> visited;
        queue<shared_ptr<const GraphNode>> q;
        unordered_map<string, bool> seen;

        // We need a const shared_ptr to this; use a workaround
        // since shared_from_this() is non-const in some impls
        seen[id_] = true;
        visited.push_back(id_);

        // Enqueue neighbors
        for (const auto& wp : neighbors_) {
            if (auto sp = wp.lock()) {
                if (!seen[sp->id()]) {
                    seen[sp->id()] = true;
                    visited.push_back(sp->id());
                }
            }
        }

        return visited;
    }
};

void exercise_2() {
    cout << "\n=== Exercise 2: Graph with shared_ptr/weak_ptr ===" << endl;

    {
        auto nodeA = make_shared<GraphNode>("A");
        auto nodeB = make_shared<GraphNode>("B");
        auto nodeC = make_shared<GraphNode>("C");
        auto nodeD = make_shared<GraphNode>("D");

        // Create edges (including a cycle: A -> B -> C -> A)
        nodeA->addNeighbor(nodeB);
        nodeA->addNeighbor(nodeC);
        nodeB->addNeighbor(nodeC);
        nodeC->addNeighbor(nodeA);  // Cycle! weak_ptr prevents leak
        nodeC->addNeighbor(nodeD);

        cout << "\n  Reference counts:" << endl;
        cout << "    A: " << nodeA.use_count() << endl;
        cout << "    B: " << nodeB.use_count() << endl;
        cout << "    C: " << nodeC.use_count() << endl;
        cout << "    D: " << nodeD.use_count() << endl;

        // All use_counts should be 1 because neighbors are weak_ptr,
        // not shared_ptr. This means no reference cycles.

        nodeA->printNeighbors();
        nodeB->printNeighbors();
        nodeC->printNeighbors();

        cout << "\n  Leaving scope..." << endl;
    }
    // All nodes properly destroyed despite the A->B->C->A cycle
    cout << "  (All nodes freed - no memory leak from cycles)" << endl;
}

// === Exercise 3: Object Pool ===
// Problem: Implement a reusable object pool using smart pointers.
//          Objects are returned to the pool instead of being destroyed.

template<typename T>
class ObjectPool : public enable_shared_from_this<ObjectPool<T>> {
    // Pool of available objects
    vector<unique_ptr<T>> available_;
    size_t totalCreated_ = 0;

public:
    // Acquire an object from the pool. If none available, create a new one.
    // Returns a shared_ptr with a custom deleter that returns the object
    // to the pool instead of destroying it.
    shared_ptr<T> acquire() {
        unique_ptr<T> obj;

        if (!available_.empty()) {
            obj = std::move(available_.back());
            available_.pop_back();
            cout << "    [Pool] Reused object (available: "
                 << available_.size() << ")" << endl;
        } else {
            obj = make_unique<T>();
            totalCreated_++;
            cout << "    [Pool] Created new object #" << totalCreated_
                 << endl;
        }

        // Raw pointer for the custom deleter to capture
        T* rawPtr = obj.release();

        // Custom deleter: returns object to pool instead of deleting it
        // We capture a weak_ptr to the pool to avoid preventing pool destruction
        auto poolWeak = this->weak_from_this();

        return shared_ptr<T>(rawPtr, [poolWeak](T* ptr) {
            if (auto pool = poolWeak.lock()) {
                cout << "    [Pool] Object returned to pool" << endl;
                pool->available_.push_back(unique_ptr<T>(ptr));
            } else {
                cout << "    [Pool] Pool expired, deleting object" << endl;
                delete ptr;
            }
        });
    }

    size_t availableCount() const { return available_.size(); }
    size_t totalCreated() const { return totalCreated_; }
};

// Simple object to pool
struct ExpensiveObject {
    int data[1024]{};  // Simulate expensive allocation
    void process() {
        cout << "    Processing ExpensiveObject..." << endl;
    }
};

void exercise_3() {
    cout << "\n=== Exercise 3: Object Pool ===" << endl;

    // Must use shared_ptr for the pool because of enable_shared_from_this
    auto pool = make_shared<ObjectPool<ExpensiveObject>>();

    {
        // Acquire 3 objects
        auto obj1 = pool->acquire();
        auto obj2 = pool->acquire();
        auto obj3 = pool->acquire();

        obj1->process();
        obj2->process();

        cout << "\n  Pool: created=" << pool->totalCreated()
             << ", available=" << pool->availableCount() << endl;

        // obj1 and obj2 will be returned to pool when they go out of scope
        cout << "\n  Releasing obj1 and obj2..." << endl;
    }
    // obj1, obj2, obj3 returned to pool

    cout << "\n  Pool: created=" << pool->totalCreated()
         << ", available=" << pool->availableCount() << endl;

    {
        // Acquire again -- should reuse pooled objects
        cout << "\n  Acquiring 2 more objects..." << endl;
        auto obj4 = pool->acquire();
        auto obj5 = pool->acquire();
        obj4->process();

        cout << "  Pool: available=" << pool->availableCount() << endl;
    }

    cout << "\n  Final pool: created=" << pool->totalCreated()
         << ", available=" << pool->availableCount() << endl;
}

int main() {
    exercise_1();
    exercise_2();
    exercise_3();
    cout << "\nAll exercises completed!" << endl;
    return 0;
}
