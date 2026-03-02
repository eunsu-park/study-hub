/*
 * Topological Sort
 * Kahn's Algorithm, DFS-based, Cycle Detection
 *
 * Finds a linear ordering of vertices in a DAG (Directed Acyclic Graph).
 */

#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <algorithm>

using namespace std;

// =============================================================================
// 1. Kahn's Algorithm (BFS-based)
// =============================================================================

vector<int> topologicalSortKahn(int n, const vector<vector<int>>& adj) {
    vector<int> indegree(n, 0);

    // Compute in-degrees
    for (int u = 0; u < n; u++) {
        for (int v : adj[u]) {
            indegree[v]++;
        }
    }

    // Add vertices with in-degree 0 to queue
    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (indegree[i] == 0) {
            q.push(i);
        }
    }

    vector<int> result;
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);

        for (int neighbor : adj[node]) {
            indegree[neighbor]--;
            if (indegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    // If cycle exists, not all vertices can be visited
    if ((int)result.size() != n) {
        return {};  // Cycle exists
    }

    return result;
}

// =============================================================================
// 2. DFS-based Topological Sort
// =============================================================================

void topoDFS(int node, const vector<vector<int>>& adj,
             vector<bool>& visited, stack<int>& st) {
    visited[node] = true;

    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) {
            topoDFS(neighbor, adj, visited, st);
        }
    }

    st.push(node);
}

vector<int> topologicalSortDFS(int n, const vector<vector<int>>& adj) {
    vector<bool> visited(n, false);
    stack<int> st;

    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            topoDFS(i, adj, visited, st);
        }
    }

    vector<int> result;
    while (!st.empty()) {
        result.push_back(st.top());
        st.pop();
    }

    return result;
}

// =============================================================================
// 3. Cycle Detection (DFS)
// =============================================================================

bool hasCycleDFS(int node, const vector<vector<int>>& adj,
                 vector<int>& color) {
    color[node] = 1;  // In progress (gray)

    for (int neighbor : adj[node]) {
        if (color[neighbor] == 1) {
            return true;  // Back edge found (cycle)
        }
        if (color[neighbor] == 0 && hasCycleDFS(neighbor, adj, color)) {
            return true;
        }
    }

    color[node] = 2;  // Completed (black)
    return false;
}

bool hasCycle(int n, const vector<vector<int>>& adj) {
    vector<int> color(n, 0);  // 0: unvisited, 1: in progress, 2: completed

    for (int i = 0; i < n; i++) {
        if (color[i] == 0 && hasCycleDFS(i, adj, color)) {
            return true;
        }
    }

    return false;
}

// =============================================================================
// 4. Find All Topological Sort Orders
// =============================================================================

void allTopoSorts(vector<vector<int>>& adj, vector<int>& indegree,
                  vector<int>& result, vector<bool>& visited,
                  vector<vector<int>>& allResults, int n) {
    bool found = false;

    for (int i = 0; i < n; i++) {
        if (indegree[i] == 0 && !visited[i]) {
            // Select this vertex
            visited[i] = true;
            result.push_back(i);

            for (int neighbor : adj[i]) {
                indegree[neighbor]--;
            }

            allTopoSorts(adj, indegree, result, visited, allResults, n);

            // Backtrack
            visited[i] = false;
            result.pop_back();
            for (int neighbor : adj[i]) {
                indegree[neighbor]++;
            }

            found = true;
        }
    }

    if (!found && (int)result.size() == n) {
        allResults.push_back(result);
    }
}

vector<vector<int>> findAllTopologicalSorts(int n, const vector<vector<int>>& adj) {
    vector<int> indegree(n, 0);
    for (int u = 0; u < n; u++) {
        for (int v : adj[u]) {
            indegree[v]++;
        }
    }

    vector<int> result;
    vector<bool> visited(n, false);
    vector<vector<int>> allResults;
    vector<vector<int>> adjCopy = adj;

    allTopoSorts(adjCopy, indegree, result, visited, allResults, n);

    return allResults;
}

// =============================================================================
// 5. Course Schedule
// =============================================================================

bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
    vector<vector<int>> adj(numCourses);
    vector<int> indegree(numCourses, 0);

    for (auto& [course, prereq] : prerequisites) {
        adj[prereq].push_back(course);
        indegree[course]++;
    }

    queue<int> q;
    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) {
            q.push(i);
        }
    }

    int count = 0;
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        count++;

        for (int neighbor : adj[node]) {
            indegree[neighbor]--;
            if (indegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    return count == numCourses;
}

vector<int> findOrder(int numCourses, vector<pair<int, int>>& prerequisites) {
    vector<vector<int>> adj(numCourses);
    vector<int> indegree(numCourses, 0);

    for (auto& [course, prereq] : prerequisites) {
        adj[prereq].push_back(course);
        indegree[course]++;
    }

    queue<int> q;
    for (int i = 0; i < numCourses; i++) {
        if (indegree[i] == 0) {
            q.push(i);
        }
    }

    vector<int> result;
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        result.push_back(node);

        for (int neighbor : adj[node]) {
            indegree[neighbor]--;
            if (indegree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    if ((int)result.size() != numCourses) {
        return {};
    }

    return result;
}

// =============================================================================
// 6. Alien Dictionary
// =============================================================================

string alienOrder(vector<string>& words) {
    // Build graph
    unordered_map<char, unordered_set<char>> adj;
    unordered_map<char, int> indegree;

    // Initialize all characters
    for (const string& word : words) {
        for (char c : word) {
            if (indegree.find(c) == indegree.end()) {
                indegree[c] = 0;
            }
        }
    }

    // Compare adjacent words to determine order
    for (int i = 0; i < (int)words.size() - 1; i++) {
        string& w1 = words[i];
        string& w2 = words[i + 1];

        // Invalid order check (prefix cannot come after)
        if (w1.length() > w2.length() &&
            w1.substr(0, w2.length()) == w2) {
            return "";
        }

        for (int j = 0; j < (int)min(w1.length(), w2.length()); j++) {
            if (w1[j] != w2[j]) {
                if (adj[w1[j]].find(w2[j]) == adj[w1[j]].end()) {
                    adj[w1[j]].insert(w2[j]);
                    indegree[w2[j]]++;
                }
                break;
            }
        }
    }

    // Topological sort
    queue<char> q;
    for (auto& [c, deg] : indegree) {
        if (deg == 0) {
            q.push(c);
        }
    }

    string result;
    while (!q.empty()) {
        char c = q.front();
        q.pop();
        result += c;

        for (char next : adj[c]) {
            indegree[next]--;
            if (indegree[next] == 0) {
                q.push(next);
            }
        }
    }

    if (result.length() != indegree.size()) {
        return "";  // Cycle exists
    }

    return result;
}

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
    cout << "Topological Sort Examples" << endl;
    cout << "============================================================" << endl;

    // Test graph
    //   5 -> 0 <- 4
    //   |         |
    //   v         v
    //   2 -> 3 -> 1
    int n = 6;
    vector<vector<int>> adj(n);
    adj[5] = {0, 2};
    adj[4] = {0, 1};
    adj[2] = {3};
    adj[3] = {1};

    // 1. Kahn's Algorithm
    cout << "\n[1] Kahn's Algorithm (BFS)" << endl;
    auto result1 = topologicalSortKahn(n, adj);
    cout << "    Topological order: ";
    printVector(result1);
    cout << endl;

    // 2. DFS-based
    cout << "\n[2] DFS-based Topological Sort" << endl;
    auto result2 = topologicalSortDFS(n, adj);
    cout << "    Topological order: ";
    printVector(result2);
    cout << endl;

    // 3. Cycle Detection
    cout << "\n[3] Cycle Detection" << endl;
    cout << "    Current graph: " << (hasCycle(n, adj) ? "has cycle" : "DAG") << endl;

    vector<vector<int>> cycleAdj(3);
    cycleAdj[0] = {1};
    cycleAdj[1] = {2};
    cycleAdj[2] = {0};
    cout << "    Cycle graph: " << (hasCycle(3, cycleAdj) ? "has cycle" : "DAG") << endl;

    // 4. All Topological Sorts
    cout << "\n[4] All Topological Sort Orders" << endl;
    vector<vector<int>> smallAdj(4);
    smallAdj[0] = {1, 2};
    smallAdj[1] = {3};
    smallAdj[2] = {3};
    auto allSorts = findAllTopologicalSorts(4, smallAdj);
    cout << "    Graph: 0->1->3, 0->2->3" << endl;
    cout << "    Possible orders: " << allSorts.size() << endl;
    for (auto& order : allSorts) {
        cout << "      ";
        printVector(order);
        cout << endl;
    }

    // 5. Course Schedule
    cout << "\n[5] Course Schedule" << endl;
    vector<pair<int, int>> prereqs = {{1, 0}, {2, 0}, {3, 1}, {3, 2}};
    cout << "    Courses: 4, Prerequisites: (1,0), (2,0), (3,1), (3,2)" << endl;
    cout << "    Can finish: " << (canFinish(4, prereqs) ? "yes" : "no") << endl;
    auto order = findOrder(4, prereqs);
    cout << "    Course order: ";
    printVector(order);
    cout << endl;

    // 6. Complexity Summary
    cout << "\n[6] Complexity Summary" << endl;
    cout << "    | Algorithm      | Time       | Space      |" << endl;
    cout << "    |----------------|------------|------------|" << endl;
    cout << "    | Kahn (BFS)     | O(V + E)   | O(V)       |" << endl;
    cout << "    | DFS-based      | O(V + E)   | O(V)       |" << endl;
    cout << "    | All orders     | O(V! x V)  | O(V)       |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
