/*
 * Stack and Queue
 * Stack, Queue, Deque, Monotonic Stack/Queue
 *
 * Applications of linear data structures.
 */

#include <iostream>
#include <vector>
#include <stack>
#include <queue>
#include <deque>
#include <string>
#include <unordered_map>

using namespace std;

// =============================================================================
// 1. Basic Stack Applications
// =============================================================================

// Valid parentheses check
bool isValidParentheses(const string& s) {
    stack<char> st;
    unordered_map<char, char> pairs = {{')', '('}, {']', '['}, {'}', '{'}};

    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') {
            st.push(c);
        } else {
            if (st.empty() || st.top() != pairs[c])
                return false;
            st.pop();
        }
    }

    return st.empty();
}

// Reverse Polish Notation evaluation
int evalRPN(const vector<string>& tokens) {
    stack<int> st;

    for (const string& token : tokens) {
        if (token == "+" || token == "-" || token == "*" || token == "/") {
            int b = st.top(); st.pop();
            int a = st.top(); st.pop();

            if (token == "+") st.push(a + b);
            else if (token == "-") st.push(a - b);
            else if (token == "*") st.push(a * b);
            else st.push(a / b);
        } else {
            st.push(stoi(token));
        }
    }

    return st.top();
}

// Infix to Postfix conversion
string infixToPostfix(const string& infix) {
    stack<char> st;
    string postfix;
    unordered_map<char, int> precedence = {{'+', 1}, {'-', 1}, {'*', 2}, {'/', 2}};

    for (char c : infix) {
        if (isalnum(c)) {
            postfix += c;
        } else if (c == '(') {
            st.push(c);
        } else if (c == ')') {
            while (!st.empty() && st.top() != '(') {
                postfix += st.top();
                st.pop();
            }
            st.pop();  // Remove '('
        } else {
            while (!st.empty() && st.top() != '(' &&
                   precedence[st.top()] >= precedence[c]) {
                postfix += st.top();
                st.pop();
            }
            st.push(c);
        }
    }

    while (!st.empty()) {
        postfix += st.top();
        st.pop();
    }

    return postfix;
}

// =============================================================================
// 2. Monotonic Stack
// =============================================================================

// Next Greater Element
vector<int> nextGreaterElement(const vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, -1);
    stack<int> st;  // Store indices

    for (int i = 0; i < n; i++) {
        while (!st.empty() && nums[st.top()] < nums[i]) {
            result[st.top()] = nums[i];
            st.pop();
        }
        st.push(i);
    }

    return result;
}

// Largest Rectangle in Histogram
int largestRectangleArea(const vector<int>& heights) {
    stack<int> st;
    int maxArea = 0;
    int n = heights.size();

    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];

        while (!st.empty() && heights[st.top()] > h) {
            int height = heights[st.top()];
            st.pop();
            int width = st.empty() ? i : i - st.top() - 1;
            maxArea = max(maxArea, height * width);
        }
        st.push(i);
    }

    return maxArea;
}

// Daily Temperatures
vector<int> dailyTemperatures(const vector<int>& temperatures) {
    int n = temperatures.size();
    vector<int> result(n, 0);
    stack<int> st;

    for (int i = 0; i < n; i++) {
        while (!st.empty() && temperatures[st.top()] < temperatures[i]) {
            result[st.top()] = i - st.top();
            st.pop();
        }
        st.push(i);
    }

    return result;
}

// =============================================================================
// 3. Queue Applications
// =============================================================================

// Queue for BFS (simple example)
vector<int> bfsOrder(const vector<vector<int>>& graph, int start) {
    vector<int> result;
    vector<bool> visited(graph.size(), false);
    queue<int> q;

    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);

        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }

    return result;
}

// =============================================================================
// 4. Deque Applications
// =============================================================================

// Sliding Window Maximum
vector<int> maxSlidingWindow(const vector<int>& nums, int k) {
    deque<int> dq;  // Store indices
    vector<int> result;

    for (int i = 0; i < (int)nums.size(); i++) {
        // Remove elements outside the window
        while (!dq.empty() && dq.front() <= i - k) {
            dq.pop_front();
        }

        // Remove elements smaller than current
        while (!dq.empty() && nums[dq.back()] < nums[i]) {
            dq.pop_back();
        }

        dq.push_back(i);

        if (i >= k - 1) {
            result.push_back(nums[dq.front()]);
        }
    }

    return result;
}

// =============================================================================
// 5. Queue Implementation Using Two Stacks
// =============================================================================

class MyQueue {
private:
    stack<int> input, output;

    void transfer() {
        if (output.empty()) {
            while (!input.empty()) {
                output.push(input.top());
                input.pop();
            }
        }
    }

public:
    void push(int x) {
        input.push(x);
    }

    int pop() {
        transfer();
        int val = output.top();
        output.pop();
        return val;
    }

    int peek() {
        transfer();
        return output.top();
    }

    bool empty() {
        return input.empty() && output.empty();
    }
};

// =============================================================================
// 6. Min Stack
// =============================================================================

class MinStack {
private:
    stack<int> st;
    stack<int> minSt;

public:
    void push(int val) {
        st.push(val);
        if (minSt.empty() || val <= minSt.top()) {
            minSt.push(val);
        }
    }

    void pop() {
        if (st.top() == minSt.top()) {
            minSt.pop();
        }
        st.pop();
    }

    int top() {
        return st.top();
    }

    int getMin() {
        return minSt.top();
    }
};

// =============================================================================
// Test
// =============================================================================

void printVector(const vector<int>& v) {
    cout << "[";
    for (size_t i = 0; i < v.size(); i++) {
        cout << v[i];
        if (i < v.size() - 1) cout << ", ";
    }
    cout << "]";
}

int main() {
    cout << "============================================================" << endl;
    cout << "Stack and Queue Examples" << endl;
    cout << "============================================================" << endl;

    // 1. Parentheses validation
    cout << "\n[1] Valid Parentheses" << endl;
    cout << "    \"()[]{}\" : " << (isValidParentheses("()[]{}") ? "valid" : "invalid") << endl;
    cout << "    \"([)]\"   : " << (isValidParentheses("([)]") ? "valid" : "invalid") << endl;

    // 2. Reverse Polish Notation
    cout << "\n[2] Reverse Polish Notation" << endl;
    vector<string> rpn = {"2", "1", "+", "3", "*"};
    cout << "    [\"2\",\"1\",\"+\",\"3\",\"*\"] = " << evalRPN(rpn) << endl;

    cout << "    \"a+b*c\" -> \"" << infixToPostfix("a+b*c") << "\"" << endl;

    // 3. Monotonic Stack
    cout << "\n[3] Monotonic Stack" << endl;
    vector<int> nums = {2, 1, 2, 4, 3};
    cout << "    Array: [2,1,2,4,3]" << endl;
    cout << "    Next greater element: ";
    printVector(nextGreaterElement(nums));
    cout << endl;

    vector<int> heights = {2, 1, 5, 6, 2, 3};
    cout << "    Histogram [2,1,5,6,2,3] largest rectangle: " << largestRectangleArea(heights) << endl;

    vector<int> temps = {73, 74, 75, 71, 69, 72, 76, 73};
    cout << "    Daily temperatures [73,74,75,71,69,72,76,73]: ";
    printVector(dailyTemperatures(temps));
    cout << endl;

    // 4. Sliding Window Maximum
    cout << "\n[4] Sliding Window Maximum" << endl;
    vector<int> window = {1, 3, -1, -3, 5, 3, 6, 7};
    cout << "    Array: [1,3,-1,-3,5,3,6,7], k=3" << endl;
    cout << "    Maximum values: ";
    printVector(maxSlidingWindow(window, 3));
    cout << endl;

    // 5. Queue using two stacks
    cout << "\n[5] Queue Using Two Stacks" << endl;
    MyQueue q;
    q.push(1);
    q.push(2);
    cout << "    push(1), push(2)" << endl;
    cout << "    peek(): " << q.peek() << endl;
    cout << "    pop(): " << q.pop() << endl;
    cout << "    empty(): " << (q.empty() ? "true" : "false") << endl;

    // 6. Min Stack
    cout << "\n[6] Min Stack" << endl;
    MinStack ms;
    ms.push(-2);
    ms.push(0);
    ms.push(-3);
    cout << "    push(-2), push(0), push(-3)" << endl;
    cout << "    getMin(): " << ms.getMin() << endl;
    ms.pop();
    cout << "    After pop() getMin(): " << ms.getMin() << endl;

    // 7. Complexity Summary
    cout << "\n[7] Complexity Summary" << endl;
    cout << "    | Operation    | Stack | Queue | Deque |" << endl;
    cout << "    |--------------|-------|-------|-------|" << endl;
    cout << "    | push/enqueue | O(1)  | O(1)  | O(1)  |" << endl;
    cout << "    | pop/dequeue  | O(1)  | O(1)  | O(1)  |" << endl;
    cout << "    | peek/front   | O(1)  | O(1)  | O(1)  |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
