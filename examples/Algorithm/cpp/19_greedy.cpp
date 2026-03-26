/*
 * Greedy Algorithm
 * Activity Selection, Huffman, Interval Scheduling, Fractional Knapsack
 *
 * Finds optimal solutions by making the best choice at each step.
 */

#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <string>
#include <unordered_map>

using namespace std;

// =============================================================================
// 1. Activity Selection Problem
// =============================================================================

struct Activity {
    int start, end, idx;
};

vector<int> activitySelection(vector<Activity>& activities) {
    // Sort by end time
    sort(activities.begin(), activities.end(),
         [](const Activity& a, const Activity& b) {
             return a.end < b.end;
         });

    vector<int> selected;
    int lastEnd = 0;

    for (const auto& act : activities) {
        if (act.start >= lastEnd) {
            selected.push_back(act.idx);
            lastEnd = act.end;
        }
    }

    return selected;
}

// =============================================================================
// 2. Meeting Room Assignment (Minimum Meeting Rooms)
// =============================================================================

int minMeetingRooms(vector<pair<int, int>>& intervals) {
    vector<pair<int, int>> events;

    for (const auto& [start, end] : intervals) {
        events.push_back({start, 1});   // Start
        events.push_back({end, -1});    // End
    }

    sort(events.begin(), events.end());

    int rooms = 0, maxRooms = 0;
    for (const auto& [time, type] : events) {
        rooms += type;
        maxRooms = max(maxRooms, rooms);
    }

    return maxRooms;
}

// =============================================================================
// 3. Fractional Knapsack
// =============================================================================

struct Item {
    int weight, value;
    double ratio() const { return (double)value / weight; }
};

double fractionalKnapsack(int W, vector<Item>& items) {
    // Sort by value/weight ratio
    sort(items.begin(), items.end(),
         [](const Item& a, const Item& b) {
             return a.ratio() > b.ratio();
         });

    double totalValue = 0;
    int remaining = W;

    for (const auto& item : items) {
        if (remaining >= item.weight) {
            totalValue += item.value;
            remaining -= item.weight;
        } else {
            totalValue += item.ratio() * remaining;
            break;
        }
    }

    return totalValue;
}

// =============================================================================
// 4. Huffman Coding
// =============================================================================

struct HuffmanNode {
    char ch;
    int freq;
    HuffmanNode *left, *right;

    HuffmanNode(char c, int f) : ch(c), freq(f), left(nullptr), right(nullptr) {}
};

struct Compare {
    bool operator()(HuffmanNode* a, HuffmanNode* b) {
        return a->freq > b->freq;
    }
};

void generateCodes(HuffmanNode* root, string code,
                   unordered_map<char, string>& codes) {
    if (!root) return;

    if (!root->left && !root->right) {
        codes[root->ch] = code.empty() ? "0" : code;
        return;
    }

    generateCodes(root->left, code + "0", codes);
    generateCodes(root->right, code + "1", codes);
}

unordered_map<char, string> huffmanCoding(const string& text) {
    // Frequency count
    unordered_map<char, int> freq;
    for (char c : text) freq[c]++;

    // Priority queue (min heap)
    priority_queue<HuffmanNode*, vector<HuffmanNode*>, Compare> pq;
    for (auto& [c, f] : freq) {
        pq.push(new HuffmanNode(c, f));
    }

    // Build tree
    while (pq.size() > 1) {
        HuffmanNode* left = pq.top(); pq.pop();
        HuffmanNode* right = pq.top(); pq.pop();

        HuffmanNode* parent = new HuffmanNode('\0', left->freq + right->freq);
        parent->left = left;
        parent->right = right;
        pq.push(parent);
    }

    // Generate codes
    unordered_map<char, string> codes;
    if (!pq.empty()) {
        generateCodes(pq.top(), "", codes);
    }

    return codes;
}

// =============================================================================
// 5. Job Scheduling with Deadlines
// =============================================================================

struct Job {
    int id, deadline, profit;
};

int jobScheduling(vector<Job>& jobs) {
    // Sort by profit in descending order
    sort(jobs.begin(), jobs.end(),
         [](const Job& a, const Job& b) {
             return a.profit > b.profit;
         });

    int maxDeadline = 0;
    for (const auto& job : jobs) {
        maxDeadline = max(maxDeadline, job.deadline);
    }

    vector<int> slots(maxDeadline + 1, -1);  // Time slot availability
    int totalProfit = 0;

    for (const auto& job : jobs) {
        // Find empty slot starting from deadline
        for (int t = job.deadline; t >= 1; t--) {
            if (slots[t] == -1) {
                slots[t] = job.id;
                totalProfit += job.profit;
                break;
            }
        }
    }

    return totalProfit;
}

// =============================================================================
// 6. Minimum Coin Change
// =============================================================================

int minCoins(vector<int>& coins, int amount) {
    // Use largest coins first (greedy approach - optimal only for certain denominations)
    sort(coins.rbegin(), coins.rend());

    int count = 0;
    for (int coin : coins) {
        count += amount / coin;
        amount %= coin;
    }

    return amount == 0 ? count : -1;
}

// =============================================================================
// 7. Interval Cover Problem
// =============================================================================

int intervalCover(vector<pair<int, int>>& intervals, int target) {
    // Sort by start point
    sort(intervals.begin(), intervals.end());

    int count = 0;
    int current = 0;
    int i = 0;
    int n = intervals.size();

    while (current < target) {
        int maxEnd = current;

        // Among intervals covering the current position, select the one reaching farthest
        while (i < n && intervals[i].first <= current) {
            maxEnd = max(maxEnd, intervals[i].second);
            i++;
        }

        if (maxEnd == current) {
            return -1;  // Cannot cover
        }

        current = maxEnd;
        count++;
    }

    return count;
}

// =============================================================================
// 8. Jump Game
// =============================================================================

bool canJump(const vector<int>& nums) {
    int maxReach = 0;

    for (int i = 0; i < (int)nums.size(); i++) {
        if (i > maxReach) return false;
        maxReach = max(maxReach, i + nums[i]);
    }

    return true;
}

int minJumps(const vector<int>& nums) {
    int n = nums.size();
    if (n <= 1) return 0;

    int jumps = 0;
    int currentEnd = 0;
    int farthest = 0;

    for (int i = 0; i < n - 1; i++) {
        farthest = max(farthest, i + nums[i]);

        if (i == currentEnd) {
            jumps++;
            currentEnd = farthest;

            if (currentEnd >= n - 1) break;
        }
    }

    return jumps;
}

// =============================================================================
// 9. Gas Station
// =============================================================================

int canCompleteCircuit(const vector<int>& gas, const vector<int>& cost) {
    int n = gas.size();
    int totalTank = 0;
    int currTank = 0;
    int startStation = 0;

    for (int i = 0; i < n; i++) {
        int diff = gas[i] - cost[i];
        totalTank += diff;
        currTank += diff;

        if (currTank < 0) {
            startStation = i + 1;
            currTank = 0;
        }
    }

    return totalTank >= 0 ? startStation : -1;
}

// =============================================================================
// 10. String Partitioning
// =============================================================================

vector<int> partitionLabels(const string& s) {
    // Last occurrence position of each character
    vector<int> last(26, 0);
    for (int i = 0; i < (int)s.length(); i++) {
        last[s[i] - 'a'] = i;
    }

    vector<int> result;
    int start = 0, end = 0;

    for (int i = 0; i < (int)s.length(); i++) {
        end = max(end, last[s[i] - 'a']);

        if (i == end) {
            result.push_back(end - start + 1);
            start = i + 1;
        }
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
    cout << "Greedy Algorithm Example" << endl;
    cout << "============================================================" << endl;

    // 1. Activity Selection
    cout << "\n[1] Activity Selection" << endl;
    vector<Activity> activities = {
        {1, 4, 0}, {3, 5, 1}, {0, 6, 2}, {5, 7, 3},
        {3, 9, 4}, {5, 9, 5}, {6, 10, 6}, {8, 11, 7}
    };
    auto selected = activitySelection(activities);
    cout << "    Selected activities: ";
    printVector(selected);
    cout << endl;

    // 2. Minimum Meeting Rooms
    cout << "\n[2] Minimum Meeting Rooms" << endl;
    vector<pair<int, int>> meetings = {{0, 30}, {5, 10}, {15, 20}};
    cout << "    Meetings: [(0,30), (5,10), (15,20)]" << endl;
    cout << "    Min rooms: " << minMeetingRooms(meetings) << endl;

    // 3. Fractional Knapsack
    cout << "\n[3] Fractional Knapsack" << endl;
    vector<Item> items = {{10, 60}, {20, 100}, {30, 120}};
    cout << "    Items: (weight, value) = (10,60), (20,100), (30,120)" << endl;
    cout << "    Capacity 50, Max value: " << fractionalKnapsack(50, items) << endl;

    // 4. Huffman Coding
    cout << "\n[4] Huffman Coding" << endl;
    string text = "aabbbcccc";
    auto codes = huffmanCoding(text);
    cout << "    Text: \"" << text << "\"" << endl;
    cout << "    Codes:" << endl;
    for (auto& [ch, code] : codes) {
        cout << "      '" << ch << "': " << code << endl;
    }

    // 5. Job Scheduling
    cout << "\n[5] Job Scheduling" << endl;
    vector<Job> jobs = {{1, 4, 20}, {2, 1, 10}, {3, 1, 40}, {4, 1, 30}};
    cout << "    Jobs: (id, deadline, profit)" << endl;
    cout << "    Max profit: " << jobScheduling(jobs) << endl;

    // 6. Jump Game
    cout << "\n[6] Jump Game" << endl;
    vector<int> nums1 = {2, 3, 1, 1, 4};
    vector<int> nums2 = {3, 2, 1, 0, 4};
    cout << "    [2,3,1,1,4] reachable: " << (canJump(nums1) ? "Yes" : "No") << endl;
    cout << "    [2,3,1,1,4] min jumps: " << minJumps(nums1) << endl;
    cout << "    [3,2,1,0,4] reachable: " << (canJump(nums2) ? "Yes" : "No") << endl;

    // 7. Gas Station
    cout << "\n[7] Gas Station" << endl;
    vector<int> gas = {1, 2, 3, 4, 5};
    vector<int> cost = {3, 4, 5, 1, 2};
    cout << "    gas: [1,2,3,4,5], cost: [3,4,5,1,2]" << endl;
    cout << "    Start position: " << canCompleteCircuit(gas, cost) << endl;

    // 8. String Partitioning
    cout << "\n[8] String Partitioning" << endl;
    string s = "ababcbacadefegdehijhklij";
    auto parts = partitionLabels(s);
    cout << "    String: \"" << s << "\"" << endl;
    cout << "    Partition sizes: ";
    printVector(parts);
    cout << endl;

    // 9. Greedy vs DP
    cout << "\n[9] Greedy vs DP" << endl;
    cout << "    | Criterion      | Greedy        | DP              |" << endl;
    cout << "    |----------------|---------------|-----------------|" << endl;
    cout << "    | Approach       | Local optimal | Global optimal  |" << endl;
    cout << "    | Decision change| X             | O               |" << endl;
    cout << "    | Time complexity| Usually low   | Usually high    |" << endl;
    cout << "    | Optimality     | Specific only | Always guaranteed|" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
