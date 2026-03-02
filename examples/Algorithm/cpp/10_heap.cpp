/*
 * Heap
 * Min/Max Heap, Priority Queue, Heap Sort
 *
 * An efficient priority-based data structure.
 */

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <functional>

using namespace std;

// =============================================================================
// 1. Min Heap Implementation
// =============================================================================

class MinHeap {
private:
    vector<int> heap;

    void heapifyUp(int idx) {
        while (idx > 0) {
            int parent = (idx - 1) / 2;
            if (heap[parent] <= heap[idx]) break;
            swap(heap[parent], heap[idx]);
            idx = parent;
        }
    }

    void heapifyDown(int idx) {
        int size = heap.size();
        while (true) {
            int smallest = idx;
            int left = 2 * idx + 1;
            int right = 2 * idx + 2;

            if (left < size && heap[left] < heap[smallest])
                smallest = left;
            if (right < size && heap[right] < heap[smallest])
                smallest = right;

            if (smallest == idx) break;
            swap(heap[idx], heap[smallest]);
            idx = smallest;
        }
    }

public:
    void push(int val) {
        heap.push_back(val);
        heapifyUp(heap.size() - 1);
    }

    int pop() {
        if (heap.empty()) throw runtime_error("Heap is empty");
        int minVal = heap[0];
        heap[0] = heap.back();
        heap.pop_back();
        if (!heap.empty()) heapifyDown(0);
        return minVal;
    }

    int top() const {
        if (heap.empty()) throw runtime_error("Heap is empty");
        return heap[0];
    }

    bool empty() const { return heap.empty(); }
    size_t size() const { return heap.size(); }
};

// =============================================================================
// 2. Heap Sort
// =============================================================================

void heapify(vector<int>& arr, int n, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;
    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heapSort(vector<int>& arr) {
    int n = arr.size();

    // Build heap
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }

    // Extract
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}

// =============================================================================
// 3. Kth Element
// =============================================================================

// Kth largest element in array
int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> minHeap;

    for (int num : nums) {
        minHeap.push(num);
        if ((int)minHeap.size() > k) {
            minHeap.pop();
        }
    }

    return minHeap.top();
}

// Kth smallest element in array
int findKthSmallest(vector<int>& nums, int k) {
    priority_queue<int> maxHeap;

    for (int num : nums) {
        maxHeap.push(num);
        if ((int)maxHeap.size() > k) {
            maxHeap.pop();
        }
    }

    return maxHeap.top();
}

// =============================================================================
// 4. Median Stream
// =============================================================================

class MedianFinder {
private:
    priority_queue<int> maxHeap;  // Smaller half
    priority_queue<int, vector<int>, greater<int>> minHeap;  // Larger half

public:
    void addNum(int num) {
        maxHeap.push(num);
        minHeap.push(maxHeap.top());
        maxHeap.pop();

        if (minHeap.size() > maxHeap.size()) {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
    }

    double findMedian() {
        if (maxHeap.size() > minHeap.size()) {
            return maxHeap.top();
        }
        return (maxHeap.top() + minHeap.top()) / 2.0;
    }
};

// =============================================================================
// 5. Merge K Sorted Lists
// =============================================================================

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

ListNode* mergeKLists(vector<ListNode*>& lists) {
    auto cmp = [](ListNode* a, ListNode* b) { return a->val > b->val; };
    priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);

    for (ListNode* list : lists) {
        if (list) pq.push(list);
    }

    ListNode dummy(0);
    ListNode* curr = &dummy;

    while (!pq.empty()) {
        ListNode* node = pq.top();
        pq.pop();
        curr->next = node;
        curr = curr->next;
        if (node->next) pq.push(node->next);
    }

    return dummy.next;
}

// =============================================================================
// 6. K Closest Points
// =============================================================================

vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
    auto dist = [](const vector<int>& p) {
        return p[0] * p[0] + p[1] * p[1];
    };

    auto cmp = [&dist](const vector<int>& a, const vector<int>& b) {
        return dist(a) < dist(b);
    };

    priority_queue<vector<int>, vector<vector<int>>, decltype(cmp)> maxHeap(cmp);

    for (auto& point : points) {
        maxHeap.push(point);
        if ((int)maxHeap.size() > k) {
            maxHeap.pop();
        }
    }

    vector<vector<int>> result;
    while (!maxHeap.empty()) {
        result.push_back(maxHeap.top());
        maxHeap.pop();
    }

    return result;
}

// =============================================================================
// 7. Top K Frequent Elements
// =============================================================================

vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> freq;
    for (int num : nums) {
        freq[num]++;
    }

    auto cmp = [&freq](int a, int b) { return freq[a] > freq[b]; };
    priority_queue<int, vector<int>, decltype(cmp)> minHeap(cmp);

    for (auto& [num, count] : freq) {
        minHeap.push(num);
        if ((int)minHeap.size() > k) {
            minHeap.pop();
        }
    }

    vector<int> result;
    while (!minHeap.empty()) {
        result.push_back(minHeap.top());
        minHeap.pop();
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
    cout << "Heap Examples" << endl;
    cout << "============================================================" << endl;

    // 1. Min Heap
    cout << "\n[1] Min Heap" << endl;
    MinHeap minHeap;
    minHeap.push(5);
    minHeap.push(3);
    minHeap.push(7);
    minHeap.push(1);
    cout << "    Inserted: 5, 3, 7, 1" << endl;
    cout << "    Minimum: " << minHeap.top() << endl;
    cout << "    Pop order: ";
    while (!minHeap.empty()) {
        cout << minHeap.pop() << " ";
    }
    cout << endl;

    // 2. STL priority_queue
    cout << "\n[2] STL priority_queue" << endl;
    priority_queue<int> maxPQ;  // Max heap
    priority_queue<int, vector<int>, greater<int>> minPQ;  // Min heap

    for (int x : {3, 1, 4, 1, 5, 9}) {
        maxPQ.push(x);
        minPQ.push(x);
    }

    cout << "    Max heap top: " << maxPQ.top() << endl;
    cout << "    Min heap top: " << minPQ.top() << endl;

    // 3. Heap Sort
    cout << "\n[3] Heap Sort" << endl;
    vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
    cout << "    Before sorting: ";
    printVector(arr);
    cout << endl;
    heapSort(arr);
    cout << "    After sorting: ";
    printVector(arr);
    cout << endl;

    // 4. Kth Element
    cout << "\n[4] Kth Element" << endl;
    vector<int> nums = {3, 2, 1, 5, 6, 4};
    cout << "    Array: [3,2,1,5,6,4]" << endl;
    cout << "    2nd largest: " << findKthLargest(nums, 2) << endl;
    cout << "    3rd smallest: " << findKthSmallest(nums, 3) << endl;

    // 5. Median Stream
    cout << "\n[5] Median Stream" << endl;
    MedianFinder mf;
    mf.addNum(1);
    mf.addNum(2);
    cout << "    [1, 2] median: " << mf.findMedian() << endl;
    mf.addNum(3);
    cout << "    [1, 2, 3] median: " << mf.findMedian() << endl;

    // 6. Top K Frequency
    cout << "\n[6] Top K Frequent Elements" << endl;
    vector<int> freqNums = {1, 1, 1, 2, 2, 3};
    auto topK = topKFrequent(freqNums, 2);
    cout << "    [1,1,1,2,2,3], k=2: ";
    printVector(topK);
    cout << endl;

    // 7. Complexity Summary
    cout << "\n[7] Complexity Summary" << endl;
    cout << "    | Operation    | Time       |" << endl;
    cout << "    |--------------|------------|" << endl;
    cout << "    | Insert       | O(log n)   |" << endl;
    cout << "    | Delete (top) | O(log n)   |" << endl;
    cout << "    | Min/Max      | O(1)       |" << endl;
    cout << "    | Build heap   | O(n)       |" << endl;
    cout << "    | Heap sort    | O(n log n) |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
