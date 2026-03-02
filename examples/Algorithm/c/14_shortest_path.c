/*
 * Shortest Path
 * Dijkstra, Bellman-Ford, Floyd-Warshall
 *
 * Algorithms for finding shortest paths in graphs.
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>

#define INF INT_MAX

/* =============================================================================
 * 1. Graph Structure
 * ============================================================================= */

typedef struct Edge {
    int dest;
    int weight;
    struct Edge* next;
} Edge;

typedef struct {
    Edge** adj;
    int vertices;
} Graph;

Graph* graph_create(int vertices) {
    Graph* g = malloc(sizeof(Graph));
    g->vertices = vertices;
    g->adj = calloc(vertices, sizeof(Edge*));
    return g;
}

void graph_add_edge(Graph* g, int src, int dest, int weight) {
    Edge* edge = malloc(sizeof(Edge));
    edge->dest = dest;
    edge->weight = weight;
    edge->next = g->adj[src];
    g->adj[src] = edge;
}

void graph_free(Graph* g) {
    for (int i = 0; i < g->vertices; i++) {
        Edge* e = g->adj[i];
        while (e) {
            Edge* temp = e;
            e = e->next;
            free(temp);
        }
    }
    free(g->adj);
    free(g);
}

/* =============================================================================
 * 2. Dijkstra's Algorithm (Array-based)
 * ============================================================================= */

int* dijkstra_array(Graph* g, int src) {
    int* dist = malloc(g->vertices * sizeof(int));
    bool* visited = calloc(g->vertices, sizeof(bool));

    for (int i = 0; i < g->vertices; i++)
        dist[i] = INF;
    dist[src] = 0;

    for (int count = 0; count < g->vertices - 1; count++) {
        /* Find vertex with minimum distance */
        int min_dist = INF, u = -1;
        for (int v = 0; v < g->vertices; v++) {
            if (!visited[v] && dist[v] < min_dist) {
                min_dist = dist[v];
                u = v;
            }
        }

        if (u == -1) break;
        visited[u] = true;

        /* Update adjacent vertices */
        Edge* e = g->adj[u];
        while (e) {
            if (!visited[e->dest] && dist[u] != INF &&
                dist[u] + e->weight < dist[e->dest]) {
                dist[e->dest] = dist[u] + e->weight;
            }
            e = e->next;
        }
    }

    free(visited);
    return dist;
}

/* =============================================================================
 * 3. Dijkstra's Algorithm (Priority Queue)
 * ============================================================================= */

typedef struct {
    int vertex;
    int dist;
} HeapNode;

typedef struct {
    HeapNode* data;
    int size;
    int capacity;
} MinHeap;

MinHeap* heap_create(int capacity) {
    MinHeap* h = malloc(sizeof(MinHeap));
    h->data = malloc(capacity * sizeof(HeapNode));
    h->size = 0;
    h->capacity = capacity;
    return h;
}

void heap_push(MinHeap* h, int vertex, int dist) {
    int i = h->size++;
    h->data[i].vertex = vertex;
    h->data[i].dist = dist;

    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->data[parent].dist <= h->data[i].dist) break;
        HeapNode temp = h->data[parent];
        h->data[parent] = h->data[i];
        h->data[i] = temp;
        i = parent;
    }
}

HeapNode heap_pop(MinHeap* h) {
    HeapNode min = h->data[0];
    h->data[0] = h->data[--h->size];

    int i = 0;
    while (2 * i + 1 < h->size) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (h->data[left].dist < h->data[smallest].dist)
            smallest = left;
        if (right < h->size && h->data[right].dist < h->data[smallest].dist)
            smallest = right;

        if (smallest == i) break;

        HeapNode temp = h->data[i];
        h->data[i] = h->data[smallest];
        h->data[smallest] = temp;
        i = smallest;
    }

    return min;
}

int* dijkstra_heap(Graph* g, int src) {
    int* dist = malloc(g->vertices * sizeof(int));
    for (int i = 0; i < g->vertices; i++)
        dist[i] = INF;
    dist[src] = 0;

    MinHeap* pq = heap_create(g->vertices * g->vertices);
    heap_push(pq, src, 0);

    while (pq->size > 0) {
        HeapNode node = heap_pop(pq);
        int u = node.vertex;
        int d = node.dist;

        if (d > dist[u]) continue;

        Edge* e = g->adj[u];
        while (e) {
            if (dist[u] + e->weight < dist[e->dest]) {
                dist[e->dest] = dist[u] + e->weight;
                heap_push(pq, e->dest, dist[e->dest]);
            }
            e = e->next;
        }
    }

    free(pq->data);
    free(pq);
    return dist;
}

/* =============================================================================
 * 4. Bellman-Ford Algorithm
 * ============================================================================= */

typedef struct {
    int src;
    int dest;
    int weight;
} EdgeList;

int* bellman_ford(int vertices, EdgeList edges[], int num_edges, int src, bool* has_negative_cycle) {
    int* dist = malloc(vertices * sizeof(int));
    for (int i = 0; i < vertices; i++)
        dist[i] = INF;
    dist[src] = 0;

    /* Iterate V-1 times */
    for (int i = 0; i < vertices - 1; i++) {
        for (int j = 0; j < num_edges; j++) {
            int u = edges[j].src;
            int v = edges[j].dest;
            int w = edges[j].weight;

            if (dist[u] != INF && dist[u] + w < dist[v])
                dist[v] = dist[u] + w;
        }
    }

    /* Check for negative cycles */
    *has_negative_cycle = false;
    for (int j = 0; j < num_edges; j++) {
        int u = edges[j].src;
        int v = edges[j].dest;
        int w = edges[j].weight;

        if (dist[u] != INF && dist[u] + w < dist[v]) {
            *has_negative_cycle = true;
            break;
        }
    }

    return dist;
}

/* =============================================================================
 * 5. Floyd-Warshall Algorithm
 * ============================================================================= */

int** floyd_warshall(int** graph, int vertices) {
    int** dist = malloc(vertices * sizeof(int*));
    for (int i = 0; i < vertices; i++) {
        dist[i] = malloc(vertices * sizeof(int));
        for (int j = 0; j < vertices; j++)
            dist[i][j] = graph[i][j];
    }

    for (int k = 0; k < vertices; k++) {
        for (int i = 0; i < vertices; i++) {
            for (int j = 0; j < vertices; j++) {
                if (dist[i][k] != INF && dist[k][j] != INF &&
                    dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    return dist;
}

/* =============================================================================
 * 6. Path Reconstruction
 * ============================================================================= */

int* dijkstra_with_path(Graph* g, int src, int** parent) {
    int* dist = malloc(g->vertices * sizeof(int));
    *parent = malloc(g->vertices * sizeof(int));
    bool* visited = calloc(g->vertices, sizeof(bool));

    for (int i = 0; i < g->vertices; i++) {
        dist[i] = INF;
        (*parent)[i] = -1;
    }
    dist[src] = 0;

    for (int count = 0; count < g->vertices - 1; count++) {
        int min_dist = INF, u = -1;
        for (int v = 0; v < g->vertices; v++) {
            if (!visited[v] && dist[v] < min_dist) {
                min_dist = dist[v];
                u = v;
            }
        }

        if (u == -1) break;
        visited[u] = true;

        Edge* e = g->adj[u];
        while (e) {
            if (!visited[e->dest] && dist[u] != INF &&
                dist[u] + e->weight < dist[e->dest]) {
                dist[e->dest] = dist[u] + e->weight;
                (*parent)[e->dest] = u;
            }
            e = e->next;
        }
    }

    free(visited);
    return dist;
}

void print_path(int parent[], int dest) {
    if (parent[dest] == -1) {
        printf("%d", dest);
        return;
    }
    print_path(parent, parent[dest]);
    printf(" -> %d", dest);
}

/* =============================================================================
 * Test
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("Shortest Path Examples\n");
    printf("============================================================\n");

    /* 1. Dijkstra (Array) */
    printf("\n[1] Dijkstra's Algorithm (Array)\n");
    Graph* g1 = graph_create(5);
    graph_add_edge(g1, 0, 1, 4);
    graph_add_edge(g1, 0, 2, 1);
    graph_add_edge(g1, 2, 1, 2);
    graph_add_edge(g1, 1, 3, 1);
    graph_add_edge(g1, 2, 3, 5);
    graph_add_edge(g1, 3, 4, 3);

    int* dist1 = dijkstra_array(g1, 0);
    printf("    Distances from 0 to each vertex:\n");
    for (int i = 0; i < 5; i++)
        printf("      0 -> %d: %d\n", i, dist1[i]);
    free(dist1);

    /* 2. Dijkstra (Heap) */
    printf("\n[2] Dijkstra's Algorithm (Heap)\n");
    int* dist2 = dijkstra_heap(g1, 0);
    printf("    Distances from 0 to each vertex:\n");
    for (int i = 0; i < 5; i++)
        printf("      0 -> %d: %d\n", i, dist2[i]);
    free(dist2);

    /* 3. Path Reconstruction */
    printf("\n[3] Path Reconstruction\n");
    int* parent;
    int* dist3 = dijkstra_with_path(g1, 0, &parent);
    for (int i = 1; i < 5; i++) {
        printf("    0 -> %d (distance %d): ", i, dist3[i]);
        print_path(parent, i);
        printf("\n");
    }
    free(dist3);
    free(parent);
    graph_free(g1);

    /* 4. Bellman-Ford */
    printf("\n[4] Bellman-Ford Algorithm\n");
    EdgeList edges[] = {
        {0, 1, 4}, {0, 2, 1}, {2, 1, 2},
        {1, 3, 1}, {2, 3, 5}, {3, 4, 3}
    };
    bool has_neg_cycle;
    int* dist4 = bellman_ford(5, edges, 6, 0, &has_neg_cycle);
    printf("    Negative cycle: %s\n", has_neg_cycle ? "exists" : "none");
    printf("    Distances: ");
    for (int i = 0; i < 5; i++)
        printf("%d ", dist4[i]);
    printf("\n");
    free(dist4);

    /* Negative cycle test */
    printf("\n    Negative edge test:\n");
    EdgeList edges_neg[] = {
        {0, 1, 1}, {1, 2, -1}, {2, 0, -1}
    };
    int* dist_neg = bellman_ford(3, edges_neg, 3, 0, &has_neg_cycle);
    printf("    Negative cycle: %s\n", has_neg_cycle ? "exists" : "none");
    free(dist_neg);

    /* 5. Floyd-Warshall */
    printf("\n[5] Floyd-Warshall Algorithm\n");
    int** matrix = malloc(4 * sizeof(int*));
    for (int i = 0; i < 4; i++) {
        matrix[i] = malloc(4 * sizeof(int));
        for (int j = 0; j < 4; j++)
            matrix[i][j] = (i == j) ? 0 : INF;
    }
    matrix[0][1] = 3;
    matrix[0][3] = 7;
    matrix[1][0] = 8;
    matrix[1][2] = 2;
    matrix[2][0] = 5;
    matrix[2][3] = 1;
    matrix[3][0] = 2;

    int** dist5 = floyd_warshall(matrix, 4);
    printf("    All-pairs shortest distances:\n");
    printf("       ");
    for (int i = 0; i < 4; i++) printf("%4d ", i);
    printf("\n");
    for (int i = 0; i < 4; i++) {
        printf("    %d: ", i);
        for (int j = 0; j < 4; j++) {
            if (dist5[i][j] == INF)
                printf(" INF ");
            else
                printf("%4d ", dist5[i][j]);
        }
        printf("\n");
    }

    for (int i = 0; i < 4; i++) {
        free(matrix[i]);
        free(dist5[i]);
    }
    free(matrix);
    free(dist5);

    /* 6. Algorithm Comparison */
    printf("\n[6] Algorithm Comparison\n");
    printf("    | Algorithm         | Time          | Features              |\n");
    printf("    |-------------------|---------------|-----------------------|\n");
    printf("    | Dijkstra (Array)  | O(V^2)        | Positive weights only |\n");
    printf("    | Dijkstra (Heap)   | O(E log V)    | Best for sparse graph |\n");
    printf("    | Bellman-Ford      | O(V * E)      | Allows neg. weights   |\n");
    printf("    | Floyd-Warshall    | O(V^3)        | All-pairs shortest    |\n");

    printf("\n============================================================\n");

    return 0;
}
