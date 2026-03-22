// classes_demo.cpp — Class definition, constructor, destructor, encapsulation
// Compile: g++ -std=c++20 -Wall -Wextra -o classes_demo classes_demo.cpp

#include <iostream>
#include <string>
#include <vector>

class BankAccount {
private:
    std::string owner_;
    double balance_;
    static int account_count_;

public:
    // Default constructor
    BankAccount() : owner_("Unknown"), balance_(0.0) {
        ++account_count_;
        std::cout << "  [Default ctor] Account created\n";
    }

    // Parameterized constructor
    BankAccount(const std::string& owner, double balance)
        : owner_(owner), balance_(balance) {
        ++account_count_;
        std::cout << "  [Param ctor] Account for " << owner_ << " created\n";
    }

    // Destructor
    ~BankAccount() {
        --account_count_;
        std::cout << "  [Dtor] Account for " << owner_ << " destroyed\n";
    }

    // Getters
    const std::string& owner() const { return owner_; }
    double balance() const { return balance_; }

    // Methods
    void deposit(double amount) {
        if (amount > 0) {
            balance_ += amount;
            std::cout << owner_ << ": deposited $" << amount << '\n';
        }
    }

    bool withdraw(double amount) {
        if (amount > 0 && amount <= balance_) {
            balance_ -= amount;
            std::cout << owner_ << ": withdrew $" << amount << '\n';
            return true;
        }
        std::cout << owner_ << ": insufficient funds\n";
        return false;
    }

    void display() const {
        std::cout << "Account{owner=" << owner_
                  << ", balance=$" << balance_ << "}\n";
    }

    // Static member function
    static int total_accounts() { return account_count_; }
};

// Static member initialization
int BankAccount::account_count_ = 0;

// --- Struct (public by default) ---
struct Point {
    double x = 0.0;
    double y = 0.0;

    double distance_to(const Point& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        return std::sqrt(dx * dx + dy * dy);
    }
};

int main() {
    std::cout << "=== Constructor / Destructor ===\n";
    {
        BankAccount a1;
        BankAccount a2("Alice", 1000.0);
        std::cout << "Active accounts: " << BankAccount::total_accounts() << '\n';
    }  // a1 and a2 destroyed here
    std::cout << "After scope: " << BankAccount::total_accounts() << " accounts\n";

    std::cout << "\n=== Methods & Encapsulation ===\n";
    BankAccount acct("Bob", 500.0);
    acct.display();
    acct.deposit(200.0);
    acct.withdraw(150.0);
    acct.withdraw(1000.0);  // insufficient
    acct.display();

    std::cout << "\n=== Struct (Point) ===\n";
    Point p1{3.0, 4.0};
    Point p2{0.0, 0.0};
    std::cout << "Distance from (" << p1.x << "," << p1.y << ") to origin: "
              << p1.distance_to(p2) << '\n';

    std::cout << "\n=== Objects in Container ===\n";
    std::vector<BankAccount> accounts;
    accounts.emplace_back("Charlie", 300.0);
    accounts.emplace_back("Diana", 750.0);
    for (const auto& a : accounts) {
        a.display();
    }

    std::cout << "\n--- End of main ---\n";
    return 0;
}
