/*
 * Minimum Spanning Tree (MST)
 * Kruskal, Prim, Union-Find
 *
 * A minimum-weight tree that connects all vertices of a graph.
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>

#define INF INT_MAX

/* =============================================================================
 * 1. Union-Find (Disjoint Set Union)
 * ============================================================================= */

typedef struct {
    int* parent;
    int* rank;
    int size;
} UnionFind;

UnionFind* uf_create(int n) {
    UnionFind* uf = malloc(sizeof(UnionFind));
    uf->parent = malloc(n * sizeof(int));
    uf->rank = calloc(n, sizeof(int));
    uf->size = n;

    for (int i = 0; i < n; i++)
        uf->parent[i] = i;

    return uf;
}

void uf_free(UnionFind* uf) {
    free(uf->parent);
    free(uf->rank);
    free(uf);
}

int uf_find(UnionFind* uf, int x) {
    if (uf->parent[x] != x)
        uf->parent[x] = uf_find(uf, uf->parent[x]);  /* Path compression */
    return uf->parent[x];
}

bool uf_union(UnionFind* uf, int x, int y) {
    int px = uf_find(uf, x);
    int py = uf_find(uf, y);

    if (px == py) return false;

    /* Union by rank */
    if (uf->rank[px] < uf->rank[py]) {
        uf->parent[px] = py;
    } else if (uf->rank[px] > uf->rank[py]) {
        uf->parent[py] = px;
    } else {
        uf->parent[py] = px;
        uf->rank[px]++;
    }

    return true;
}

bool uf_connected(UnionFind* uf, int x, int y) {
    return uf_find(uf, x) == uf_find(uf, y);
}

/* =============================================================================
 * 2. Edge Structure
 * ============================================================================= */

typedef struct {
    int src;
    int dest;
    int weight;
} Edge;

int compare_edges(const void* a, const void* b) {
    return ((Edge*)a)->weight - ((Edge*)b)->weight;
}

/* =============================================================================
 * 3. Kruskal's Algorithm
 * ============================================================================= */

typedef struct {
    Edge* edges;
    int num_edges;
    int total_weight;
} MST;

MST kruskal(int vertices, Edge edges[], int num_edges) {
    /* Sort edges by weight */
    qsort(edges, num_edges, sizeof(Edge), compare_edges);

    UnionFind* uf = uf_create(vertices);
    MST mst;
    mst.edges = malloc((vertices - 1) * sizeof(Edge));
    mst.num_edges = 0;
    mst.total_weight = 0;

    for (int i = 0; i < num_edges && mst.num_edges < vertices - 1; i++) {
        int u = edges[i].src;
        int v = edges[i].dest;

        if (uf_union(uf, u, v)) {
            mst.edges[mst.num_edges++] = edges[i];
            mst.total_weight += edges[i].weight;
        }
    }

    uf_free(uf);
    return mst;
}

/* =============================================================================
 * 4. Prim's Algorithm (Array-based)
 * ============================================================================= */

MST prim_array(int** graph, int vertices) {
    int* key = malloc(vertices * sizeof(int));      /* Minimum weight */
    int* parent = malloc(vertices * sizeof(int));   /* Parent in MST */
    bool* in_mst = calloc(vertices, sizeof(bool));

    for (int i = 0; i < vertices; i++) {
        key[i] = INF;
        parent[i] = -1;
    }

    key[0] = 0;

    for (int count = 0; count < vertices - 1; count++) {
        /* Find vertex with minimum key */
        int min_key = INF, u = -1;
        for (int v = 0; v < vertices; v++) {
            if (!in_mst[v] && key[v] < min_key) {
                min_key = key[v];
                u = v;
            }
        }

        if (u == -1) break;
        in_mst[u] = true;

        /* Update adjacent vertices */
        for (int v = 0; v < vertices; v++) {
            if (graph[u][v] && !in_mst[v] && graph[u][v] < key[v]) {
                parent[v] = u;
                key[v] = graph[u][v];
            }
        }
    }

    /* Build MST */
    MST mst;
    mst.edges = malloc((vertices - 1) * sizeof(Edge));
    mst.num_edges = 0;
    mst.total_weight = 0;

    for (int i = 1; i < vertices; i++) {
        if (parent[i] != -1) {
            mst.edges[mst.num_edges].src = parent[i];
            mst.edges[mst.num_edges].dest = i;
            mst.edges[mst.num_edges].weight = graph[parent[i]][i];
            mst.total_weight += graph[parent[i]][i];
            mst.num_edges++;
        }
    }

    free(key);
    free(parent);
    free(in_mst);
    return mst;
}

/* =============================================================================
 * 5. Prim's Algorithm (Priority Queue)
 * ============================================================================= */

typedef struct {
    int vertex;
    int key;
} PQNode;

typedef struct {
    PQNode* data;
    int size;
    int capacity;
} PriorityQueue;

PriorityQueue* pq_create(int capacity) {
    PriorityQueue* pq = malloc(sizeof(PriorityQueue));
    pq->data = malloc(capacity * sizeof(PQNode));
    pq->size = 0;
    pq->capacity = capacity;
    return pq;
}

void pq_push(PriorityQueue* pq, int vertex, int key) {
    int i = pq->size++;
    pq->data[i].vertex = vertex;
    pq->data[i].key = key;

    while (i > 0) {
        int parent = (i - 1) / 2;
        if (pq->data[parent].key <= pq->data[i].key) break;
        PQNode temp = pq->data[parent];
        pq->data[parent] = pq->data[i];
        pq->data[i] = temp;
        i = parent;
    }
}

PQNode pq_pop(PriorityQueue* pq) {
    PQNode min = pq->data[0];
    pq->data[0] = pq->data[--pq->size];

    int i = 0;
    while (2 * i + 1 < pq->size) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (pq->data[left].key < pq->data[smallest].key)
            smallest = left;
        if (right < pq->size && pq->data[right].key < pq->data[smallest].key)
            smallest = right;

        if (smallest == i) break;

        PQNode temp = pq->data[i];
        pq->data[i] = pq->data[smallest];
        pq->data[smallest] = temp;
        i = smallest;
    }

    return min;
}

/* Adjacency list structure */
typedef struct AdjNode {
    int dest;
    int weight;
    struct AdjNode* next;
} AdjNode;

MST prim_heap(AdjNode** adj, int vertices) {
    int* key = malloc(vertices * sizeof(int));
    int* parent = malloc(vertices * sizeof(int));
    bool* in_mst = calloc(vertices, sizeof(bool));

    for (int i = 0; i < vertices; i++) {
        key[i] = INF;
        parent[i] = -1;
    }

    PriorityQueue* pq = pq_create(vertices * vertices);
    key[0] = 0;
    pq_push(pq, 0, 0);

    while (pq->size > 0) {
        PQNode node = pq_pop(pq);
        int u = node.vertex;

        if (in_mst[u]) continue;
        in_mst[u] = true;

        AdjNode* neighbor = adj[u];
        while (neighbor) {
            int v = neighbor->dest;
            int w = neighbor->weight;

            if (!in_mst[v] && w < key[v]) {
                key[v] = w;
                parent[v] = u;
                pq_push(pq, v, w);
            }
            neighbor = neighbor->next;
        }
    }

    MST mst;
    mst.edges = malloc((vertices - 1) * sizeof(Edge));
    mst.num_edges = 0;
    mst.total_weight = 0;

    for (int i = 1; i < vertices; i++) {
        if (parent[i] != -1) {
            mst.edges[mst.num_edges].src = parent[i];
            mst.edges[mst.num_edges].dest = i;
            mst.edges[mst.num_edges].weight = key[i];
            mst.total_weight += key[i];
            mst.num_edges++;
        }
    }

    free(key);
    free(parent);
    free(in_mst);
    free(pq->data);
    free(pq);
    return mst;
}

/* =============================================================================
 * 6. Union-Find Application: Connected Components
 * ============================================================================= */

int count_components(int n, int edges[][2], int num_edges) {
    UnionFind* uf = uf_create(n);

    for (int i = 0; i < num_edges; i++) {
        uf_union(uf, edges[i][0], edges[i][1]);
    }

    int count = 0;
    for (int i = 0; i < n; i++) {
        if (uf->parent[i] == i) count++;
    }

    uf_free(uf);
    return count;
}

/* =============================================================================
 * Test
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("Minimum Spanning Tree (MST) Examples\n");
    printf("============================================================\n");

    /* 1. Union-Find */
    printf("\n[1] Union-Find Basics\n");
    UnionFind* uf = uf_create(6);
    printf("    union(0, 1): %s\n", uf_union(uf, 0, 1) ? "merged" : "already connected");
    printf("    union(2, 3): %s\n", uf_union(uf, 2, 3) ? "merged" : "already connected");
    printf("    union(1, 2): %s\n", uf_union(uf, 1, 2) ? "merged" : "already connected");
    printf("    connected(0, 3): %s\n", uf_connected(uf, 0, 3) ? "yes" : "no");
    printf("    connected(0, 5): %s\n", uf_connected(uf, 0, 5) ? "yes" : "no");
    uf_free(uf);

    /* 2. Kruskal */
    printf("\n[2] Kruskal's Algorithm\n");
    Edge edges[] = {
        {0, 1, 4}, {0, 7, 8}, {1, 2, 8}, {1, 7, 11},
        {2, 3, 7}, {2, 8, 2}, {2, 5, 4}, {3, 4, 9},
        {3, 5, 14}, {4, 5, 10}, {5, 6, 2}, {6, 7, 1},
        {6, 8, 6}, {7, 8, 7}
    };

    printf("    Edges: 14, Vertices: 9\n");
    MST mst1 = kruskal(9, edges, 14);
    printf("    MST edges:\n");
    for (int i = 0; i < mst1.num_edges; i++) {
        printf("      %d - %d (weight %d)\n",
               mst1.edges[i].src, mst1.edges[i].dest, mst1.edges[i].weight);
    }
    printf("    Total weight: %d\n", mst1.total_weight);
    free(mst1.edges);

    /* 3. Prim (Array) */
    printf("\n[3] Prim's Algorithm (Array)\n");
    int** graph = malloc(5 * sizeof(int*));
    for (int i = 0; i < 5; i++) {
        graph[i] = calloc(5, sizeof(int));
    }
    graph[0][1] = graph[1][0] = 2;
    graph[0][3] = graph[3][0] = 6;
    graph[1][2] = graph[2][1] = 3;
    graph[1][3] = graph[3][1] = 8;
    graph[1][4] = graph[4][1] = 5;
    graph[2][4] = graph[4][2] = 7;
    graph[3][4] = graph[4][3] = 9;

    MST mst2 = prim_array(graph, 5);
    printf("    MST edges:\n");
    for (int i = 0; i < mst2.num_edges; i++) {
        printf("      %d - %d (weight %d)\n",
               mst2.edges[i].src, mst2.edges[i].dest, mst2.edges[i].weight);
    }
    printf("    Total weight: %d\n", mst2.total_weight);
    free(mst2.edges);

    for (int i = 0; i < 5; i++) free(graph[i]);
    free(graph);

    /* 4. Connected Components Count */
    printf("\n[4] Connected Components Count\n");
    int comp_edges[][2] = {{0, 1}, {1, 2}, {3, 4}};
    printf("    Vertices: 0-4, Edges: (0,1), (1,2), (3,4)\n");
    printf("    Connected components: %d\n", count_components(5, comp_edges, 3));

    /* 5. Algorithm Comparison */
    printf("\n[5] Algorithm Comparison\n");
    printf("    | Algorithm      | Time          | Best for       |\n");
    printf("    |----------------|---------------|----------------|\n");
    printf("    | Kruskal        | O(E log E)    | Sparse graph   |\n");
    printf("    | Prim (Array)   | O(V^2)        | Dense graph    |\n");
    printf("    | Prim (Heap)    | O(E log V)    | Sparse graph   |\n");

    /* 6. Union-Find Complexity */
    printf("\n[6] Union-Find Complexity\n");
    printf("    | Operation | Time            |\n");
    printf("    |-----------|----------------|\n");
    printf("    | find      | O(a(n)) ~ O(1) |\n");
    printf("    | union     | O(a(n)) ~ O(1) |\n");
    printf("    | a(n): inverse Ackermann function (grows extremely slowly)\n");

    printf("\n============================================================\n");

    return 0;
}
