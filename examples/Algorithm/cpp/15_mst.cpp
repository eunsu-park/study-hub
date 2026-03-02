/*
 * Minimum Spanning Tree
 * Kruskal, Prim, Union-Find
 *
 * Connects all vertices of a graph with minimum total cost.
 */

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <numeric>

using namespace std;

// =============================================================================
// 1. Union-Find (Disjoint Set Union)
// =============================================================================

class UnionFind {
private:
    vector<int> parent, rank_;

public:
    UnionFind(int n) : parent(n), rank_(n, 0) {
        iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    }

    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;

        // Union by rank
        if (rank_[px] < rank_[py]) swap(px, py);
        parent[py] = px;
        if (rank_[px] == rank_[py]) rank_[px]++;

        return true;
    }

    bool connected(int x, int y) {
        return find(x) == find(y);
    }
};

// =============================================================================
// 2. Kruskal's Algorithm
// =============================================================================

struct Edge {
    int u, v, weight;
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

pair<int, vector<Edge>> kruskal(int n, vector<Edge>& edges) {
    sort(edges.begin(), edges.end());

    UnionFind uf(n);
    vector<Edge> mst;
    int totalWeight = 0;

    for (const auto& e : edges) {
        if (uf.unite(e.u, e.v)) {
            mst.push_back(e);
            totalWeight += e.weight;

            if ((int)mst.size() == n - 1) break;
        }
    }

    return {totalWeight, mst};
}

// =============================================================================
// 3. Prim's Algorithm
// =============================================================================

pair<int, vector<pair<int, int>>> prim(int n, const vector<vector<pair<int, int>>>& adj) {
    vector<bool> visited(n, false);
    vector<pair<int, int>> mst;  // {u, v} edges
    int totalWeight = 0;

    // {weight, vertex, parent}
    priority_queue<tuple<int, int, int>, vector<tuple<int, int, int>>, greater<>> pq;
    pq.push({0, 0, -1});

    while (!pq.empty() && (int)mst.size() < n) {
        auto [w, u, parent] = pq.top();
        pq.pop();

        if (visited[u]) continue;
        visited[u] = true;
        totalWeight += w;

        if (parent != -1) {
            mst.push_back({parent, u});
        }

        for (auto [v, weight] : adj[u]) {
            if (!visited[v]) {
                pq.push({weight, v, u});
            }
        }
    }

    return {totalWeight, mst};
}

// =============================================================================
// 4. Second Minimum Spanning Tree
// =============================================================================

int secondMST(int n, vector<Edge>& edges) {
    // First compute MST
    auto [mstWeight, mst] = kruskal(n, edges);

    int secondBest = INT_MAX;

    // Remove each MST edge and recompute MST
    for (int i = 0; i < (int)mst.size(); i++) {
        vector<Edge> filtered;
        for (const auto& e : edges) {
            if (!(e.u == mst[i].u && e.v == mst[i].v && e.weight == mst[i].weight)) {
                filtered.push_back(e);
            }
        }

        auto [newWeight, newMst] = kruskal(n, filtered);

        if ((int)newMst.size() == n - 1) {
            secondBest = min(secondBest, newWeight);
        }
    }

    return secondBest;
}

// =============================================================================
// 5. Maximum Spanning Tree
// =============================================================================

pair<int, vector<Edge>> maxSpanningTree(int n, vector<Edge>& edges) {
    // Negate weights and apply Kruskal
    for (auto& e : edges) {
        e.weight = -e.weight;
    }

    auto [weight, mst] = kruskal(n, edges);

    // Restore original weights
    for (auto& e : edges) {
        e.weight = -e.weight;
    }
    for (auto& e : mst) {
        e.weight = -e.weight;
    }

    return {-weight, mst};
}

// =============================================================================
// 6. Kruskal Application: Minimum Connection Cost
// =============================================================================

int minCostToConnect(int n, vector<vector<int>>& connections) {
    vector<Edge> edges;
    for (const auto& conn : connections) {
        edges.push_back({conn[0] - 1, conn[1] - 1, conn[2]});
    }

    auto [cost, mst] = kruskal(n, edges);

    if ((int)mst.size() != n - 1) {
        return -1;  // Cannot connect
    }

    return cost;
}

// =============================================================================
// 7. Union-Find Application: Friend Network
// =============================================================================

class FriendNetwork {
private:
    unordered_map<string, string> parent;
    unordered_map<string, int> size_;

    string find(const string& x) {
        if (parent.find(x) == parent.end()) {
            parent[x] = x;
            size_[x] = 1;
        }
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

public:
    int unite(const string& a, const string& b) {
        string pa = find(a), pb = find(b);

        if (pa == pb) {
            return size_[pa];
        }

        if (size_[pa] < size_[pb]) swap(pa, pb);
        parent[pb] = pa;
        size_[pa] += size_[pb];

        return size_[pa];
    }
};

// =============================================================================
// 8. Union-Find Application: Find Redundant Connection
// =============================================================================

vector<int> findRedundantConnection(vector<vector<int>>& edges) {
    int n = edges.size();
    UnionFind uf(n + 1);

    for (const auto& e : edges) {
        if (!uf.unite(e[0], e[1])) {
            return e;  // Redundant connection
        }
    }

    return {};
}

// =============================================================================
// Test
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "Minimum Spanning Tree Examples" << endl;
    cout << "============================================================" << endl;

    // Test graph
    //     1 --(4)-- 2
    //    /|        /|
    //  (1)|      (3)|
    //  /  |      /  |
    // 0   (2)   /   (5)
    //  \  |  /      |
    //  (3)| /       |
    //    \|/        |
    //     3 --(6)-- 4

    int n = 5;
    vector<Edge> edges = {
        {0, 1, 1}, {0, 3, 3}, {1, 2, 4}, {1, 3, 2},
        {2, 3, 3}, {2, 4, 5}, {3, 4, 6}
    };

    // 1. Kruskal
    cout << "\n[1] Kruskal's Algorithm" << endl;
    auto [kWeight, kMst] = kruskal(n, edges);
    cout << "    MST weight: " << kWeight << endl;
    cout << "    MST edges: ";
    for (const auto& e : kMst) {
        cout << "(" << e.u << "-" << e.v << ":" << e.weight << ") ";
    }
    cout << endl;

    // 2. Prim
    cout << "\n[2] Prim's Algorithm" << endl;
    vector<vector<pair<int, int>>> adj(n);
    for (const auto& e : edges) {
        adj[e.u].push_back({e.v, e.weight});
        adj[e.v].push_back({e.u, e.weight});
    }
    auto [pWeight, pMst] = prim(n, adj);
    cout << "    MST weight: " << pWeight << endl;
    cout << "    MST edges: ";
    for (const auto& [u, v] : pMst) {
        cout << "(" << u << "-" << v << ") ";
    }
    cout << endl;

    // 3. Union-Find
    cout << "\n[3] Union-Find" << endl;
    UnionFind uf(5);
    uf.unite(0, 1);
    uf.unite(2, 3);
    uf.unite(1, 2);
    cout << "    After union: 0-1, 2-3, 1-2" << endl;
    cout << "    0 and 3 connected: " << (uf.connected(0, 3) ? "yes" : "no") << endl;
    cout << "    0 and 4 connected: " << (uf.connected(0, 4) ? "yes" : "no") << endl;

    // 4. Second MST
    cout << "\n[4] Second Minimum Spanning Tree" << endl;
    vector<Edge> edges2 = edges;  // Copy
    int secondWeight = secondMST(n, edges2);
    cout << "    Second MST weight: " << secondWeight << endl;

    // 5. Maximum Spanning Tree
    cout << "\n[5] Maximum Spanning Tree" << endl;
    vector<Edge> edges3 = edges;  // Copy
    auto [maxWeight, maxMst] = maxSpanningTree(n, edges3);
    cout << "    Max ST weight: " << maxWeight << endl;

    // 6. Redundant Connection
    cout << "\n[6] Find Redundant Connection" << endl;
    vector<vector<int>> redEdges = {{1, 2}, {1, 3}, {2, 3}};
    auto redundant = findRedundantConnection(redEdges);
    cout << "    Edges: (1,2), (1,3), (2,3)" << endl;
    cout << "    Redundant: (" << redundant[0] << ", " << redundant[1] << ")" << endl;

    // 7. Complexity Summary
    cout << "\n[7] Complexity Summary" << endl;
    cout << "    | Algorithm   | Time             | Notes              |" << endl;
    cout << "    |-------------|------------------|--------------------|" << endl;
    cout << "    | Kruskal     | O(E log E)       | Edge-based, sparse |" << endl;
    cout << "    | Prim        | O((V+E) log V)   | Vertex-based, dense|" << endl;
    cout << "    | Union-Find  | O(a(n)) ~ O(1)   | Nearly constant    |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
