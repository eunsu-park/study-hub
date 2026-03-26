/*
 * LCA (Lowest Common Ancestor)
 * Binary Lifting, Euler Tour, Tree Path Queries
 *
 * Finds the lowest common ancestor of two nodes in a tree.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAXN 100005
#define LOG 17

/* =============================================================================
 * 1. Tree Structure
 * ============================================================================= */

typedef struct AdjNode {
    int vertex;
    int weight;
    struct AdjNode* next;
} AdjNode;

AdjNode* adj[MAXN];
int depth[MAXN];
int parent[MAXN][LOG];  /* Binary Lifting table */
int dist[MAXN];         /* Distance from root */
int n;

void add_edge(int u, int v, int w) {
    AdjNode* node = malloc(sizeof(AdjNode));
    node->vertex = v;
    node->weight = w;
    node->next = adj[u];
    adj[u] = node;

    node = malloc(sizeof(AdjNode));
    node->vertex = u;
    node->weight = w;
    node->next = adj[v];
    adj[v] = node;
}

/* =============================================================================
 * 2. Binary Lifting Preprocessing
 * ============================================================================= */

void dfs_preprocess(int v, int p, int d, int distance) {
    depth[v] = d;
    parent[v][0] = p;
    dist[v] = distance;

    for (int i = 1; i < LOG; i++) {
        if (parent[v][i-1] != -1)
            parent[v][i] = parent[parent[v][i-1]][i-1];
        else
            parent[v][i] = -1;
    }

    AdjNode* node = adj[v];
    while (node) {
        if (node->vertex != p) {
            dfs_preprocess(node->vertex, v, d + 1, distance + node->weight);
        }
        node = node->next;
    }
}

void preprocess(int root) {
    memset(parent, -1, sizeof(parent));
    dfs_preprocess(root, -1, 0, 0);
}

/* =============================================================================
 * 3. LCA Query
 * ============================================================================= */

int lca(int u, int v) {
    /* Match depths */
    if (depth[u] < depth[v]) {
        int temp = u; u = v; v = temp;
    }

    int diff = depth[u] - depth[v];
    for (int i = 0; i < LOG; i++) {
        if ((diff >> i) & 1) {
            u = parent[u][i];
        }
    }

    if (u == v) return u;

    /* Climb up simultaneously */
    for (int i = LOG - 1; i >= 0; i--) {
        if (parent[u][i] != parent[v][i]) {
            u = parent[u][i];
            v = parent[v][i];
        }
    }

    return parent[u][0];
}

/* =============================================================================
 * 4. K-th Ancestor
 * ============================================================================= */

int kth_ancestor(int v, int k) {
    for (int i = 0; i < LOG && v != -1; i++) {
        if ((k >> i) & 1) {
            v = parent[v][i];
        }
    }
    return v;
}

/* =============================================================================
 * 5. Distance Between Two Nodes
 * ============================================================================= */

int distance_between(int u, int v) {
    int ancestor = lca(u, v);
    return dist[u] + dist[v] - 2 * dist[ancestor];
}

/* =============================================================================
 * 6. K-th Node on Path
 * ============================================================================= */

int kth_node_on_path(int u, int v, int k) {
    int ancestor = lca(u, v);
    int dist_u_lca = depth[u] - depth[ancestor];
    int dist_v_lca = depth[v] - depth[ancestor];

    if (k <= dist_u_lca) {
        return kth_ancestor(u, k);
    } else {
        return kth_ancestor(v, dist_u_lca + dist_v_lca - k);
    }
}

/* =============================================================================
 * 7. Euler Tour + RMQ Method
 * ============================================================================= */

int euler_tour[2 * MAXN];
int first_occurrence[MAXN];
int euler_depth[2 * MAXN];
int euler_idx;

/* Sparse Table for RMQ */
int sparse_table[2 * MAXN][LOG];
int log_table[2 * MAXN];

void euler_dfs(int v, int p, int d) {
    first_occurrence[v] = euler_idx;
    euler_tour[euler_idx] = v;
    euler_depth[euler_idx] = d;
    euler_idx++;

    AdjNode* node = adj[v];
    while (node) {
        if (node->vertex != p) {
            euler_dfs(node->vertex, v, d + 1);
            euler_tour[euler_idx] = v;
            euler_depth[euler_idx] = d;
            euler_idx++;
        }
        node = node->next;
    }
}

void build_sparse_table(int len) {
    /* Preprocess log table */
    log_table[1] = 0;
    for (int i = 2; i <= len; i++) {
        log_table[i] = log_table[i / 2] + 1;
    }

    /* Initialize Sparse Table */
    for (int i = 0; i < len; i++) {
        sparse_table[i][0] = i;
    }

    for (int j = 1; (1 << j) <= len; j++) {
        for (int i = 0; i + (1 << j) - 1 < len; i++) {
            int left = sparse_table[i][j-1];
            int right = sparse_table[i + (1 << (j-1))][j-1];
            sparse_table[i][j] = (euler_depth[left] < euler_depth[right]) ? left : right;
        }
    }
}

int rmq_query(int l, int r) {
    int k = log_table[r - l + 1];
    int left = sparse_table[l][k];
    int right = sparse_table[r - (1 << k) + 1][k];
    return (euler_depth[left] < euler_depth[right]) ? left : right;
}

int lca_rmq(int u, int v) {
    int l = first_occurrence[u];
    int r = first_occurrence[v];
    if (l > r) {
        int temp = l; l = r; r = temp;
    }
    return euler_tour[rmq_query(l, r)];
}

void preprocess_euler(int root) {
    euler_idx = 0;
    euler_dfs(root, -1, 0);
    build_sparse_table(euler_idx);
}

/* =============================================================================
 * Test
 * ============================================================================= */

void cleanup(void) {
    for (int i = 0; i < n; i++) {
        AdjNode* node = adj[i];
        while (node) {
            AdjNode* temp = node;
            node = node->next;
            free(temp);
        }
        adj[i] = NULL;
    }
}

int main(void) {
    printf("============================================================\n");
    printf("LCA (Lowest Common Ancestor) Examples\n");
    printf("============================================================\n");

    /*
     * Tree structure:
     *        0
     *       /|\
     *      1 2 3
     *     /|   |
     *    4 5   6
     *   /
     *  7
     */
    n = 8;
    add_edge(0, 1, 1);
    add_edge(0, 2, 2);
    add_edge(0, 3, 3);
    add_edge(1, 4, 1);
    add_edge(1, 5, 1);
    add_edge(3, 6, 2);
    add_edge(4, 7, 1);

    /* 1. Binary Lifting Preprocessing */
    printf("\n[1] Binary Lifting LCA\n");
    preprocess(0);

    printf("    Tree: 0-1-4-7, 0-1-5, 0-2, 0-3-6\n");
    printf("    LCA(4, 5) = %d\n", lca(4, 5));
    printf("    LCA(4, 6) = %d\n", lca(4, 6));
    printf("    LCA(7, 2) = %d\n", lca(7, 2));
    printf("    LCA(5, 7) = %d\n", lca(5, 7));

    /* 2. K-th Ancestor */
    printf("\n[2] K-th Ancestor\n");
    printf("    1st ancestor of 7: %d\n", kth_ancestor(7, 1));
    printf("    2nd ancestor of 7: %d\n", kth_ancestor(7, 2));
    printf("    3rd ancestor of 7: %d\n", kth_ancestor(7, 3));

    /* 3. Distance */
    printf("\n[3] Distance Between Two Nodes\n");
    printf("    dist(4, 5) = %d\n", distance_between(4, 5));
    printf("    dist(7, 6) = %d\n", distance_between(7, 6));
    printf("    dist(7, 2) = %d\n", distance_between(7, 2));

    /* 4. K-th Node on Path */
    printf("\n[4] K-th Node on Path\n");
    printf("    0th node on path 7->6: %d\n", kth_node_on_path(7, 6, 0));
    printf("    2nd node on path 7->6: %d\n", kth_node_on_path(7, 6, 2));
    printf("    4th node on path 7->6: %d\n", kth_node_on_path(7, 6, 4));

    /* 5. Euler Tour + RMQ */
    printf("\n[5] Euler Tour + RMQ LCA\n");
    preprocess_euler(0);
    printf("    LCA_RMQ(4, 5) = %d\n", lca_rmq(4, 5));
    printf("    LCA_RMQ(7, 6) = %d\n", lca_rmq(7, 6));

    /* 6. Complexity Comparison */
    printf("\n[6] LCA Algorithm Comparison\n");
    printf("    | Method           | Preprocess | Query     | Space      |\n");
    printf("    |------------------|-----------|-----------|------------|\n");
    printf("    | Binary Lifting   | O(n log n)| O(log n)  | O(n log n) |\n");
    printf("    | Euler + RMQ      | O(n log n)| O(1)      | O(n log n) |\n");
    printf("    | Tarjan (offline) | O(n + q)  | O(a(n))   | O(n)       |\n");

    cleanup();

    printf("\n============================================================\n");

    return 0;
}
