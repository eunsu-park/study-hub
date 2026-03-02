/*
 * Hash Table
 * unordered_map, unordered_set, Collision Resolution
 *
 * Provides O(1) average time for search/insert/delete.
 */

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <algorithm>

using namespace std;

// =============================================================================
// 1. Basic Hash Function
// =============================================================================

// String hash (polynomial hash)
size_t polynomialHash(const string& s, size_t base = 31, size_t mod = 1e9 + 9) {
    size_t hash = 0;
    size_t power = 1;
    for (char c : s) {
        hash = (hash + (c - 'a' + 1) * power) % mod;
        power = (power * base) % mod;
    }
    return hash;
}

// =============================================================================
// 2. Chaining Hash Table Implementation
// =============================================================================

template<typename K, typename V>
class ChainingHashTable {
private:
    vector<list<pair<K, V>>> table;
    size_t capacity;
    size_t count;

    size_t hash(const K& key) const {
        return std::hash<K>{}(key) % capacity;
    }

public:
    ChainingHashTable(size_t cap = 16) : capacity(cap), count(0) {
        table.resize(capacity);
    }

    void insert(const K& key, const V& value) {
        size_t idx = hash(key);
        for (auto& p : table[idx]) {
            if (p.first == key) {
                p.second = value;
                return;
            }
        }
        table[idx].emplace_back(key, value);
        count++;
    }

    V* get(const K& key) {
        size_t idx = hash(key);
        for (auto& p : table[idx]) {
            if (p.first == key) {
                return &p.second;
            }
        }
        return nullptr;
    }

    bool remove(const K& key) {
        size_t idx = hash(key);
        for (auto it = table[idx].begin(); it != table[idx].end(); ++it) {
            if (it->first == key) {
                table[idx].erase(it);
                count--;
                return true;
            }
        }
        return false;
    }

    size_t size() const { return count; }
};

// =============================================================================
// 3. Open Addressing (Linear Probing)
// =============================================================================

template<typename K, typename V>
class LinearProbingHashTable {
private:
    struct Entry {
        K key;
        V value;
        bool occupied = false;
        bool deleted = false;
    };

    vector<Entry> table;
    size_t capacity;
    size_t count;

    size_t hash(const K& key) const {
        return std::hash<K>{}(key) % capacity;
    }

public:
    LinearProbingHashTable(size_t cap = 16) : capacity(cap), count(0) {
        table.resize(capacity);
    }

    void insert(const K& key, const V& value) {
        size_t idx = hash(key);
        size_t start = idx;

        do {
            if (!table[idx].occupied || table[idx].deleted) {
                table[idx] = {key, value, true, false};
                count++;
                return;
            }
            if (table[idx].key == key) {
                table[idx].value = value;
                return;
            }
            idx = (idx + 1) % capacity;
        } while (idx != start);
    }

    V* get(const K& key) {
        size_t idx = hash(key);
        size_t start = idx;

        do {
            if (!table[idx].occupied && !table[idx].deleted) {
                return nullptr;
            }
            if (table[idx].occupied && !table[idx].deleted && table[idx].key == key) {
                return &table[idx].value;
            }
            idx = (idx + 1) % capacity;
        } while (idx != start);

        return nullptr;
    }

    size_t size() const { return count; }
};

// =============================================================================
// 4. LRU Cache (Least Recently Used)
// =============================================================================

class LRUCache {
private:
    int capacity;
    list<pair<int, int>> cache;  // {key, value}
    unordered_map<int, list<pair<int, int>>::iterator> map;

public:
    LRUCache(int cap) : capacity(cap) {}

    int get(int key) {
        if (map.find(key) == map.end()) {
            return -1;
        }
        // Move to front
        cache.splice(cache.begin(), cache, map[key]);
        return map[key]->second;
    }

    void put(int key, int value) {
        if (map.find(key) != map.end()) {
            // Update existing key
            map[key]->second = value;
            cache.splice(cache.begin(), cache, map[key]);
        } else {
            // Insert new key
            if ((int)cache.size() == capacity) {
                // Remove LRU entry
                int lruKey = cache.back().first;
                cache.pop_back();
                map.erase(lruKey);
            }
            cache.emplace_front(key, value);
            map[key] = cache.begin();
        }
    }
};

// =============================================================================
// 5. Hash Application Problems
// =============================================================================

// Two Sum
vector<int> twoSum(const vector<int>& nums, int target) {
    unordered_map<int, int> map;
    for (int i = 0; i < (int)nums.size(); i++) {
        int complement = target - nums[i];
        if (map.count(complement)) {
            return {map[complement], i};
        }
        map[nums[i]] = i;
    }
    return {};
}

// Group Anagrams
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> groups;

    for (const string& s : strs) {
        string key = s;
        sort(key.begin(), key.end());
        groups[key].push_back(s);
    }

    vector<vector<string>> result;
    for (auto& p : groups) {
        result.push_back(move(p.second));
    }
    return result;
}

// Longest Consecutive Sequence
int longestConsecutive(const vector<int>& nums) {
    unordered_set<int> numSet(nums.begin(), nums.end());
    int maxLen = 0;

    for (int num : numSet) {
        if (!numSet.count(num - 1)) {  // Starting point
            int currentNum = num;
            int currentLen = 1;

            while (numSet.count(currentNum + 1)) {
                currentNum++;
                currentLen++;
            }

            maxLen = max(maxLen, currentLen);
        }
    }

    return maxLen;
}

// Subarray Sum Equals k
int subarraySum(const vector<int>& nums, int k) {
    unordered_map<int, int> prefixCount;
    prefixCount[0] = 1;
    int sum = 0;
    int count = 0;

    for (int num : nums) {
        sum += num;
        if (prefixCount.count(sum - k)) {
            count += prefixCount[sum - k];
        }
        prefixCount[sum]++;
    }

    return count;
}

// =============================================================================
// Test
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "Hash Table Examples" << endl;
    cout << "============================================================" << endl;

    // 1. Basic hash function
    cout << "\n[1] Hash Function" << endl;
    cout << "    hash(\"hello\") = " << polynomialHash("hello") << endl;
    cout << "    hash(\"world\") = " << polynomialHash("world") << endl;

    // 2. Chaining hash table
    cout << "\n[2] Chaining Hash Table" << endl;
    ChainingHashTable<string, int> cht;
    cht.insert("apple", 100);
    cht.insert("banana", 200);
    cht.insert("cherry", 300);
    cout << "    apple: " << *cht.get("apple") << endl;
    cout << "    banana: " << *cht.get("banana") << endl;

    // 3. Linear probing hash table
    cout << "\n[3] Linear Probing Hash Table" << endl;
    LinearProbingHashTable<string, int> lpt;
    lpt.insert("one", 1);
    lpt.insert("two", 2);
    lpt.insert("three", 3);
    cout << "    one: " << *lpt.get("one") << endl;
    cout << "    size: " << lpt.size() << endl;

    // 4. LRU Cache
    cout << "\n[4] LRU Cache" << endl;
    LRUCache lru(2);
    lru.put(1, 1);
    lru.put(2, 2);
    cout << "    put(1,1), put(2,2)" << endl;
    cout << "    get(1): " << lru.get(1) << endl;
    lru.put(3, 3);  // Evicts key 2
    cout << "    put(3,3), get(2): " << lru.get(2) << endl;

    // 5. Two Sum
    cout << "\n[5] Two Sum" << endl;
    vector<int> nums = {2, 7, 11, 15};
    auto result = twoSum(nums, 9);
    cout << "    [2,7,11,15], target=9: [" << result[0] << "," << result[1] << "]" << endl;

    // 6. Longest Consecutive Sequence
    cout << "\n[6] Longest Consecutive Sequence" << endl;
    vector<int> consecutive = {100, 4, 200, 1, 3, 2};
    cout << "    [100,4,200,1,3,2]: " << longestConsecutive(consecutive) << endl;

    // 7. Complexity Summary
    cout << "\n[7] Complexity Summary" << endl;
    cout << "    | Operation  | Average | Worst |" << endl;
    cout << "    |------------|---------|-------|" << endl;
    cout << "    | Insert     | O(1)    | O(n)  |" << endl;
    cout << "    | Search     | O(1)    | O(n)  |" << endl;
    cout << "    | Delete     | O(1)    | O(n)  |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
