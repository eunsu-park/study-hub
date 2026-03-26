/*
 * Graph Basics
 * DFS, BFS, Graph Representation, Connected Components
 *
 * Graph data structures and fundamental traversal algorithms.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define MAX_VERTICES 100

/* =============================================================================
 * 1. Graph Representation - Adjacency List
 * ============================================================================= */

typedef struct AdjNode {
    int vertex;
    int weight;
    struct AdjNode* next;
} AdjNode;

typedef struct {
    AdjNode** adj;
    int vertices;
    bool directed;
} Graph;

Graph* graph_create(int vertices, bool directed) {
    Graph* g = malloc(sizeof(Graph));
    g->vertices = vertices;
    g->directed = directed;
    g->adj = calloc(vertices, sizeof(AdjNode*));
    return g;
}

void graph_add_edge(Graph* g, int src, int dest, int weight) {
    AdjNode* node = malloc(sizeof(AdjNode));
    node->vertex = dest;
    node->weight = weight;
    node->next = g->adj[src];
    g->adj[src] = node;

    if (!g->directed) {
        node = malloc(sizeof(AdjNode));
        node->vertex = src;
        node->weight = weight;
        node->next = g->adj[dest];
        g->adj[dest] = node;
    }
}

void graph_free(Graph* g) {
    for (int i = 0; i < g->vertices; i++) {
        AdjNode* node = g->adj[i];
        while (node) {
            AdjNode* temp = node;
            node = node->next;
            free(temp);
        }
    }
    free(g->adj);
    free(g);
}

void graph_print(Graph* g) {
    for (int i = 0; i < g->vertices; i++) {
        printf("    %d: ", i);
        AdjNode* node = g->adj[i];
        while (node) {
            printf("%d(%d) ", node->vertex, node->weight);
            node = node->next;
        }
        printf("\n");
    }
}

/* =============================================================================
 * 2. Adjacency Matrix
 * ============================================================================= */

typedef struct {
    int** matrix;
    int vertices;
} AdjMatrix;

AdjMatrix* matrix_create(int vertices) {
    AdjMatrix* m = malloc(sizeof(AdjMatrix));
    m->vertices = vertices;
    m->matrix = malloc(vertices * sizeof(int*));
    for (int i = 0; i < vertices; i++) {
        m->matrix[i] = calloc(vertices, sizeof(int));
    }
    return m;
}

void matrix_add_edge(AdjMatrix* m, int src, int dest, int weight, bool directed) {
    m->matrix[src][dest] = weight;
    if (!directed)
        m->matrix[dest][src] = weight;
}

void matrix_free(AdjMatrix* m) {
    for (int i = 0; i < m->vertices; i++)
        free(m->matrix[i]);
    free(m->matrix);
    free(m);
}

/* =============================================================================
 * 3. DFS (Depth-First Search)
 * ============================================================================= */

void dfs_recursive(Graph* g, int v, bool visited[]) {
    visited[v] = true;
    printf("%d ", v);

    AdjNode* node = g->adj[v];
    while (node) {
        if (!visited[node->vertex])
            dfs_recursive(g, node->vertex, visited);
        node = node->next;
    }
}

void dfs(Graph* g, int start) {
    bool* visited = calloc(g->vertices, sizeof(bool));
    dfs_recursive(g, start, visited);
    free(visited);
}

/* Stack-based DFS */
void dfs_iterative(Graph* g, int start) {
    bool* visited = calloc(g->vertices, sizeof(bool));
    int* stack = malloc(g->vertices * sizeof(int));
    int top = -1;

    stack[++top] = start;

    while (top >= 0) {
        int v = stack[top--];

        if (!visited[v]) {
            visited[v] = true;
            printf("%d ", v);

            AdjNode* node = g->adj[v];
            while (node) {
                if (!visited[node->vertex])
                    stack[++top] = node->vertex;
                node = node->next;
            }
        }
    }

    free(visited);
    free(stack);
}

/* =============================================================================
 * 4. BFS (Breadth-First Search)
 * ============================================================================= */

void bfs(Graph* g, int start) {
    bool* visited = calloc(g->vertices, sizeof(bool));
    int* queue = malloc(g->vertices * sizeof(int));
    int front = 0, rear = 0;

    visited[start] = true;
    queue[rear++] = start;

    while (front < rear) {
        int v = queue[front++];
        printf("%d ", v);

        AdjNode* node = g->adj[v];
        while (node) {
            if (!visited[node->vertex]) {
                visited[node->vertex] = true;
                queue[rear++] = node->vertex;
            }
            node = node->next;
        }
    }

    free(visited);
    free(queue);
}

/* BFS Shortest Distances */
int* bfs_distances(Graph* g, int start) {
    int* dist = malloc(g->vertices * sizeof(int));
    for (int i = 0; i < g->vertices; i++)
        dist[i] = -1;

    int* queue = malloc(g->vertices * sizeof(int));
    int front = 0, rear = 0;

    dist[start] = 0;
    queue[rear++] = start;

    while (front < rear) {
        int v = queue[front++];

        AdjNode* node = g->adj[v];
        while (node) {
            if (dist[node->vertex] == -1) {
                dist[node->vertex] = dist[v] + 1;
                queue[rear++] = node->vertex;
            }
            node = node->next;
        }
    }

    free(queue);
    return dist;
}

/* =============================================================================
 * 5. Connected Components
 * ============================================================================= */

int count_connected_components(Graph* g) {
    bool* visited = calloc(g->vertices, sizeof(bool));
    int count = 0;

    for (int i = 0; i < g->vertices; i++) {
        if (!visited[i]) {
            dfs_recursive(g, i, visited);
            count++;
        }
    }

    free(visited);
    return count;
}

/* =============================================================================
 * 6. Cycle Detection
 * ============================================================================= */

/* Cycle detection in undirected graph */
bool has_cycle_undirected_util(Graph* g, int v, bool visited[], int parent) {
    visited[v] = true;

    AdjNode* node = g->adj[v];
    while (node) {
        if (!visited[node->vertex]) {
            if (has_cycle_undirected_util(g, node->vertex, visited, v))
                return true;
        } else if (node->vertex != parent) {
            return true;
        }
        node = node->next;
    }

    return false;
}

bool has_cycle_undirected(Graph* g) {
    bool* visited = calloc(g->vertices, sizeof(bool));

    for (int i = 0; i < g->vertices; i++) {
        if (!visited[i]) {
            if (has_cycle_undirected_util(g, i, visited, -1)) {
                free(visited);
                return true;
            }
        }
    }

    free(visited);
    return false;
}

/* Cycle detection in directed graph (3-color algorithm) */
bool has_cycle_directed_util(Graph* g, int v, int color[]) {
    color[v] = 1;  /* Gray: processing */

    AdjNode* node = g->adj[v];
    while (node) {
        if (color[node->vertex] == 1)
            return true;  /* Gray node found = cycle */
        if (color[node->vertex] == 0) {
            if (has_cycle_directed_util(g, node->vertex, color))
                return true;
        }
        node = node->next;
    }

    color[v] = 2;  /* Black: completed */
    return false;
}

bool has_cycle_directed(Graph* g) {
    int* color = calloc(g->vertices, sizeof(int));

    for (int i = 0; i < g->vertices; i++) {
        if (color[i] == 0) {
            if (has_cycle_directed_util(g, i, color)) {
                free(color);
                return true;
            }
        }
    }

    free(color);
    return false;
}

/* =============================================================================
 * 7. Bipartite Graph Check
 * ============================================================================= */

bool is_bipartite(Graph* g) {
    int* color = malloc(g->vertices * sizeof(int));
    for (int i = 0; i < g->vertices; i++)
        color[i] = -1;

    int* queue = malloc(g->vertices * sizeof(int));

    for (int start = 0; start < g->vertices; start++) {
        if (color[start] != -1) continue;

        int front = 0, rear = 0;
        queue[rear++] = start;
        color[start] = 0;

        while (front < rear) {
            int v = queue[front++];

            AdjNode* node = g->adj[v];
            while (node) {
                if (color[node->vertex] == -1) {
                    color[node->vertex] = 1 - color[v];
                    queue[rear++] = node->vertex;
                } else if (color[node->vertex] == color[v]) {
                    free(color);
                    free(queue);
                    return false;
                }
                node = node->next;
            }
        }
    }

    free(color);
    free(queue);
    return true;
}

/* =============================================================================
 * Test
 * ============================================================================= */

int main(void) {
    printf("============================================================\n");
    printf("Graph Basics Examples\n");
    printf("============================================================\n");

    /* 1. Graph Creation */
    printf("\n[1] Graph Creation (Adjacency List)\n");
    Graph* g = graph_create(6, false);
    graph_add_edge(g, 0, 1, 1);
    graph_add_edge(g, 0, 2, 1);
    graph_add_edge(g, 1, 2, 1);
    graph_add_edge(g, 1, 3, 1);
    graph_add_edge(g, 2, 4, 1);
    graph_add_edge(g, 3, 4, 1);
    graph_add_edge(g, 4, 5, 1);

    printf("    Graph structure:\n");
    graph_print(g);

    /* 2. DFS */
    printf("\n[2] DFS (starting from 0)\n");
    printf("    Recursive: ");
    dfs(g, 0);
    printf("\n");
    printf("    Iterative: ");
    dfs_iterative(g, 0);
    printf("\n");

    /* 3. BFS */
    printf("\n[3] BFS (starting from 0)\n");
    printf("    Order: ");
    bfs(g, 0);
    printf("\n");

    int* distances = bfs_distances(g, 0);
    printf("    Distances: ");
    for (int i = 0; i < 6; i++)
        printf("%d->%d ", i, distances[i]);
    printf("\n");
    free(distances);

    /* 4. Connected Components */
    printf("\n[4] Connected Components\n");
    printf("    Current graph: %d component(s)\n", count_connected_components(g));

    Graph* g2 = graph_create(6, false);
    graph_add_edge(g2, 0, 1, 1);
    graph_add_edge(g2, 2, 3, 1);
    graph_add_edge(g2, 4, 5, 1);
    printf("    Disconnected graph: %d component(s)\n", count_connected_components(g2));
    graph_free(g2);

    /* 5. Cycle Detection */
    printf("\n[5] Cycle Detection\n");
    printf("    Undirected graph cycle: %s\n",
           has_cycle_undirected(g) ? "exists" : "none");

    Graph* dag = graph_create(4, true);
    graph_add_edge(dag, 0, 1, 1);
    graph_add_edge(dag, 1, 2, 1);
    graph_add_edge(dag, 2, 3, 1);
    printf("    Directed graph (DAG) cycle: %s\n",
           has_cycle_directed(dag) ? "exists" : "none");

    graph_add_edge(dag, 3, 1, 1);  /* Add cycle */
    printf("    Directed graph (with cycle) cycle: %s\n",
           has_cycle_directed(dag) ? "exists" : "none");
    graph_free(dag);

    /* 6. Bipartite Graph */
    printf("\n[6] Bipartite Graph Check\n");
    Graph* bipartite = graph_create(4, false);
    graph_add_edge(bipartite, 0, 1, 1);
    graph_add_edge(bipartite, 0, 3, 1);
    graph_add_edge(bipartite, 1, 2, 1);
    graph_add_edge(bipartite, 2, 3, 1);
    printf("    Square graph: %s\n",
           is_bipartite(bipartite) ? "bipartite" : "not bipartite");

    Graph* non_bipartite = graph_create(3, false);
    graph_add_edge(non_bipartite, 0, 1, 1);
    graph_add_edge(non_bipartite, 1, 2, 1);
    graph_add_edge(non_bipartite, 2, 0, 1);
    printf("    Triangle graph: %s\n",
           is_bipartite(non_bipartite) ? "bipartite" : "not bipartite");
    graph_free(bipartite);
    graph_free(non_bipartite);

    graph_free(g);

    /* 7. Complexity Summary */
    printf("\n[7] Graph Algorithm Complexity\n");
    printf("    | Algorithm          | Time        | Space    |\n");
    printf("    |--------------------|-------------|----------|\n");
    printf("    | DFS                | O(V + E)    | O(V)     |\n");
    printf("    | BFS                | O(V + E)    | O(V)     |\n");
    printf("    | Connected Comp.    | O(V + E)    | O(V)     |\n");
    printf("    | Cycle Detection    | O(V + E)    | O(V)     |\n");
    printf("    | Bipartite Check    | O(V + E)    | O(V)     |\n");

    printf("\n============================================================\n");

    return 0;
}
