/*
 * String Algorithms
 * KMP, Rabin-Karp, Z-Algorithm, Suffix Array
 *
 * Algorithms for string searching and processing.
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>

using namespace std;

// =============================================================================
// 1. KMP Algorithm
// =============================================================================

// Failure function (partial match table)
vector<int> computeFailure(const string& pattern) {
    int m = pattern.length();
    vector<int> failure(m, 0);

    int k = 0;
    for (int i = 1; i < m; i++) {
        while (k > 0 && pattern[k] != pattern[i]) {
            k = failure[k - 1];
        }
        if (pattern[k] == pattern[i]) {
            k++;
        }
        failure[i] = k;
    }

    return failure;
}

// KMP search
vector<int> kmpSearch(const string& text, const string& pattern) {
    vector<int> result;
    if (pattern.empty()) return result;

    vector<int> failure = computeFailure(pattern);
    int n = text.length(), m = pattern.length();
    int j = 0;

    for (int i = 0; i < n; i++) {
        while (j > 0 && text[i] != pattern[j]) {
            j = failure[j - 1];
        }
        if (text[i] == pattern[j]) {
            j++;
        }
        if (j == m) {
            result.push_back(i - m + 1);
            j = failure[j - 1];
        }
    }

    return result;
}

// =============================================================================
// 2. Rabin-Karp Algorithm
// =============================================================================

class RabinKarp {
private:
    const long long BASE = 31;
    const long long MOD = 1e9 + 9;

public:
    vector<int> search(const string& text, const string& pattern) {
        vector<int> result;
        int n = text.length(), m = pattern.length();
        if (m > n) return result;

        // Pattern hash
        long long patternHash = 0;
        long long textHash = 0;
        long long power = 1;

        for (int i = 0; i < m; i++) {
            patternHash = (patternHash * BASE + pattern[i]) % MOD;
            textHash = (textHash * BASE + text[i]) % MOD;
            if (i < m - 1) {
                power = (power * BASE) % MOD;
            }
        }

        for (int i = 0; i <= n - m; i++) {
            if (patternHash == textHash) {
                // Verify hash collision
                if (text.substr(i, m) == pattern) {
                    result.push_back(i);
                }
            }

            if (i < n - m) {
                textHash = (textHash - text[i] * power % MOD + MOD) % MOD;
                textHash = (textHash * BASE + text[i + m]) % MOD;
            }
        }

        return result;
    }

    // Rolling hash for substring comparison
    long long hash(const string& s) {
        long long h = 0;
        for (char c : s) {
            h = (h * BASE + c) % MOD;
        }
        return h;
    }
};

// =============================================================================
// 3. Z-Algorithm
// =============================================================================

vector<int> zFunction(const string& s) {
    int n = s.length();
    vector<int> z(n, 0);
    int l = 0, r = 0;

    for (int i = 1; i < n; i++) {
        if (i < r) {
            z[i] = min(r - i, z[i - l]);
        }
        while (i + z[i] < n && s[z[i]] == s[i + z[i]]) {
            z[i]++;
        }
        if (i + z[i] > r) {
            l = i;
            r = i + z[i];
        }
    }

    return z;
}

vector<int> zSearch(const string& text, const string& pattern) {
    string combined = pattern + "$" + text;
    vector<int> z = zFunction(combined);
    vector<int> result;

    int m = pattern.length();
    for (int i = m + 1; i < (int)combined.length(); i++) {
        if (z[i] == m) {
            result.push_back(i - m - 1);
        }
    }

    return result;
}

// =============================================================================
// 4. Suffix Array
// =============================================================================

vector<int> buildSuffixArray(const string& s) {
    int n = s.length();
    vector<int> sa(n), rank_(n), tmp(n);

    for (int i = 0; i < n; i++) {
        sa[i] = i;
        rank_[i] = s[i];
    }

    for (int k = 1; k < n; k *= 2) {
        auto cmp = [&](int a, int b) {
            if (rank_[a] != rank_[b]) return rank_[a] < rank_[b];
            int ra = (a + k < n) ? rank_[a + k] : -1;
            int rb = (b + k < n) ? rank_[b + k] : -1;
            return ra < rb;
        };

        sort(sa.begin(), sa.end(), cmp);

        tmp[sa[0]] = 0;
        for (int i = 1; i < n; i++) {
            tmp[sa[i]] = tmp[sa[i-1]] + (cmp(sa[i-1], sa[i]) ? 1 : 0);
        }
        rank_ = tmp;
    }

    return sa;
}

// LCP Array (Kasai's Algorithm)
vector<int> buildLCPArray(const string& s, const vector<int>& sa) {
    int n = s.length();
    vector<int> rank_(n), lcp(n);

    for (int i = 0; i < n; i++) {
        rank_[sa[i]] = i;
    }

    int k = 0;
    for (int i = 0; i < n; i++) {
        if (rank_[i] == 0) {
            k = 0;
            continue;
        }

        int j = sa[rank_[i] - 1];
        while (i + k < n && j + k < n && s[i + k] == s[j + k]) {
            k++;
        }
        lcp[rank_[i]] = k;
        if (k > 0) k--;
    }

    return lcp;
}

// =============================================================================
// 5. Manacher's Algorithm (Longest Palindrome)
// =============================================================================

string manacher(const string& s) {
    // Insert # between characters
    string t = "#";
    for (char c : s) {
        t += c;
        t += '#';
    }

    int n = t.length();
    vector<int> p(n, 0);
    int center = 0, right = 0;

    for (int i = 0; i < n; i++) {
        if (i < right) {
            p[i] = min(right - i, p[2 * center - i]);
        }
        while (i - p[i] - 1 >= 0 && i + p[i] + 1 < n &&
               t[i - p[i] - 1] == t[i + p[i] + 1]) {
            p[i]++;
        }
        if (i + p[i] > right) {
            center = i;
            right = i + p[i];
        }
    }

    // Find the longest palindrome
    int maxLen = 0, maxCenter = 0;
    for (int i = 0; i < n; i++) {
        if (p[i] > maxLen) {
            maxLen = p[i];
            maxCenter = i;
        }
    }

    int start = (maxCenter - maxLen) / 2;
    return s.substr(start, maxLen);
}

// =============================================================================
// 6. Trie (Simple Version)
// =============================================================================

class SimpleTrie {
private:
    struct Node {
        unordered_map<char, Node*> children;
        bool isEnd = false;
    };
    Node* root;

public:
    SimpleTrie() : root(new Node()) {}

    void insert(const string& word) {
        Node* curr = root;
        for (char c : word) {
            if (!curr->children.count(c)) {
                curr->children[c] = new Node();
            }
            curr = curr->children[c];
        }
        curr->isEnd = true;
    }

    bool search(const string& word) {
        Node* curr = root;
        for (char c : word) {
            if (!curr->children.count(c)) return false;
            curr = curr->children[c];
        }
        return curr->isEnd;
    }

    bool startsWith(const string& prefix) {
        Node* curr = root;
        for (char c : prefix) {
            if (!curr->children.count(c)) return false;
            curr = curr->children[c];
        }
        return true;
    }
};

// =============================================================================
// 7. String Hashing
// =============================================================================

class StringHash {
private:
    const long long BASE = 31;
    const long long MOD = 1e9 + 9;
    vector<long long> hash_, power_;

public:
    StringHash(const string& s) {
        int n = s.length();
        hash_.resize(n + 1, 0);
        power_.resize(n + 1, 1);

        for (int i = 0; i < n; i++) {
            hash_[i + 1] = (hash_[i] * BASE + s[i]) % MOD;
            power_[i + 1] = (power_[i] * BASE) % MOD;
        }
    }

    // Hash of range [l, r)
    long long getHash(int l, int r) {
        return (hash_[r] - hash_[l] * power_[r - l] % MOD + MOD) % MOD;
    }
};

// =============================================================================
// 8. Aho-Corasick
// =============================================================================

class AhoCorasick {
private:
    static const int ALPHABET = 26;
    struct Node {
        int children[ALPHABET];
        int fail;
        vector<int> output;  // Matched pattern indices

        Node() : fail(0) {
            fill(children, children + ALPHABET, -1);
        }
    };

    vector<Node> trie;

    int charToIdx(char c) { return c - 'a'; }

public:
    AhoCorasick() {
        trie.emplace_back();  // Root
    }

    void addPattern(const string& pattern, int idx) {
        int curr = 0;
        for (char c : pattern) {
            int ci = charToIdx(c);
            if (trie[curr].children[ci] == -1) {
                trie[curr].children[ci] = trie.size();
                trie.emplace_back();
            }
            curr = trie[curr].children[ci];
        }
        trie[curr].output.push_back(idx);
    }

    void build() {
        queue<int> q;

        for (int i = 0; i < ALPHABET; i++) {
            if (trie[0].children[i] != -1) {
                trie[trie[0].children[i]].fail = 0;
                q.push(trie[0].children[i]);
            }
        }

        while (!q.empty()) {
            int curr = q.front();
            q.pop();

            for (int i = 0; i < ALPHABET; i++) {
                int next = trie[curr].children[i];
                if (next != -1) {
                    int fail = trie[curr].fail;
                    while (fail && trie[fail].children[i] == -1) {
                        fail = trie[fail].fail;
                    }
                    trie[next].fail = (trie[fail].children[i] != -1 && trie[fail].children[i] != next)
                                      ? trie[fail].children[i] : 0;

                    // Merge output
                    for (int idx : trie[trie[next].fail].output) {
                        trie[next].output.push_back(idx);
                    }
                    q.push(next);
                }
            }
        }
    }

    vector<pair<int, int>> search(const string& text) {
        vector<pair<int, int>> result;  // {position, pattern index}
        int curr = 0;

        for (int i = 0; i < (int)text.length(); i++) {
            int ci = charToIdx(text[i]);

            while (curr && trie[curr].children[ci] == -1) {
                curr = trie[curr].fail;
            }
            curr = (trie[curr].children[ci] != -1) ? trie[curr].children[ci] : 0;

            for (int idx : trie[curr].output) {
                result.push_back({i, idx});
            }
        }

        return result;
    }
};

// =============================================================================
// Test
// =============================================================================

#include <queue>

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
    cout << "String Algorithms Example" << endl;
    cout << "============================================================" << endl;

    string text = "ABABDABACDABABCABAB";
    string pattern = "ABAB";

    // 1. KMP
    cout << "\n[1] KMP Algorithm" << endl;
    cout << "    Text: \"" << text << "\"" << endl;
    cout << "    Pattern: \"" << pattern << "\"" << endl;
    auto kmpResult = kmpSearch(text, pattern);
    cout << "    Found at: ";
    printVector(kmpResult);
    cout << endl;

    auto failure = computeFailure(pattern);
    cout << "    Failure function: ";
    printVector(failure);
    cout << endl;

    // 2. Rabin-Karp
    cout << "\n[2] Rabin-Karp Algorithm" << endl;
    RabinKarp rk;
    auto rkResult = rk.search(text, pattern);
    cout << "    Found at: ";
    printVector(rkResult);
    cout << endl;

    // 3. Z-Algorithm
    cout << "\n[3] Z-Algorithm" << endl;
    auto zResult = zSearch(text, pattern);
    cout << "    Found at: ";
    printVector(zResult);
    cout << endl;

    string zStr = "aabxaab";
    auto z = zFunction(zStr);
    cout << "    Z(\"aabxaab\"): ";
    printVector(z);
    cout << endl;

    // 4. Suffix Array
    cout << "\n[4] Suffix Array" << endl;
    string s = "banana";
    auto sa = buildSuffixArray(s);
    cout << "    \"banana\" SA: ";
    printVector(sa);
    cout << endl;
    auto lcp = buildLCPArray(s, sa);
    cout << "    LCP: ";
    printVector(lcp);
    cout << endl;

    // 5. Manacher
    cout << "\n[5] Manacher's Algorithm" << endl;
    cout << "    \"babad\" longest palindrome: \"" << manacher("babad") << "\"" << endl;
    cout << "    \"cbbd\" longest palindrome: \"" << manacher("cbbd") << "\"" << endl;

    // 6. Trie
    cout << "\n[6] Trie" << endl;
    SimpleTrie trie;
    trie.insert("apple");
    trie.insert("app");
    cout << "    insert: apple, app" << endl;
    cout << "    search(apple): " << (trie.search("apple") ? "true" : "false") << endl;
    cout << "    search(app): " << (trie.search("app") ? "true" : "false") << endl;
    cout << "    startsWith(ap): " << (trie.startsWith("ap") ? "true" : "false") << endl;

    // 7. String Hashing
    cout << "\n[7] String Hashing" << endl;
    StringHash sh("abcabc");
    cout << "    \"abcabc\" hash" << endl;
    cout << "    hash[0,3) = hash[3,6): " <<
         (sh.getHash(0, 3) == sh.getHash(3, 6) ? "true" : "false") << endl;

    // 8. Complexity Summary
    cout << "\n[8] Complexity Summary" << endl;
    cout << "    | Algorithm      | Preprocess | Search      |" << endl;
    cout << "    |----------------|------------|-------------|" << endl;
    cout << "    | KMP            | O(m)       | O(n)        |" << endl;
    cout << "    | Rabin-Karp     | O(m)       | O(n) avg    |" << endl;
    cout << "    | Z-Algorithm    | O(n)       | O(n)        |" << endl;
    cout << "    | Suffix Array   | O(n log n) | O(m log n)  |" << endl;
    cout << "    | Manacher       | -          | O(n)        |" << endl;
    cout << "    | Aho-Corasick   | O(Sum(m))  | O(n + hits) |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
