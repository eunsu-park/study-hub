/*
 * Array and String
 * Two Pointers, Sliding Window, Prefix Sum
 *
 * Core techniques for array and string processing.
 */

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <climits>

using namespace std;

// =============================================================================
// 1. Two Pointers
// =============================================================================

// Two sum in a sorted array
pair<int, int> twoSumSorted(const vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;

    while (left < right) {
        int sum = arr[left] + arr[right];
        if (sum == target)
            return {left, right};
        else if (sum < target)
            left++;
        else
            right--;
    }

    return {-1, -1};
}

// Container With Most Water
int maxArea(const vector<int>& height) {
    int left = 0, right = height.size() - 1;
    int maxWater = 0;

    while (left < right) {
        int h = min(height[left], height[right]);
        int w = right - left;
        maxWater = max(maxWater, h * w);

        if (height[left] < height[right])
            left++;
        else
            right--;
    }

    return maxWater;
}

// Three Sum
vector<vector<int>> threeSum(vector<int>& nums) {
    vector<vector<int>> result;
    sort(nums.begin(), nums.end());
    int n = nums.size();

    for (int i = 0; i < n - 2; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;

        int left = i + 1, right = n - 1;
        while (left < right) {
            int sum = nums[i] + nums[left] + nums[right];
            if (sum == 0) {
                result.push_back({nums[i], nums[left], nums[right]});
                while (left < right && nums[left] == nums[left + 1]) left++;
                while (left < right && nums[right] == nums[right - 1]) right--;
                left++;
                right--;
            } else if (sum < 0) {
                left++;
            } else {
                right--;
            }
        }
    }

    return result;
}

// =============================================================================
// 2. Sliding Window
// =============================================================================

// Maximum sum of a fixed-size window
int maxSumWindow(const vector<int>& arr, int k) {
    int n = arr.size();
    if (n < k) return -1;

    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += arr[i];
    }

    int maxSum = windowSum;
    for (int i = k; i < n; i++) {
        windowSum += arr[i] - arr[i - k];
        maxSum = max(maxSum, windowSum);
    }

    return maxSum;
}

// Longest substring without repeating characters
int lengthOfLongestSubstring(const string& s) {
    unordered_set<char> charSet;
    int maxLen = 0;
    int left = 0;

    for (int right = 0; right < (int)s.length(); right++) {
        while (charSet.count(s[right])) {
            charSet.erase(s[left]);
            left++;
        }
        charSet.insert(s[right]);
        maxLen = max(maxLen, right - left + 1);
    }

    return maxLen;
}

// Minimum length subarray with sum >= target
int minSubArrayLen(int target, const vector<int>& nums) {
    int n = nums.size();
    int minLen = INT_MAX;
    int sum = 0;
    int left = 0;

    for (int right = 0; right < n; right++) {
        sum += nums[right];
        while (sum >= target) {
            minLen = min(minLen, right - left + 1);
            sum -= nums[left];
            left++;
        }
    }

    return minLen == INT_MAX ? 0 : minLen;
}

// =============================================================================
// 3. Prefix Sum
// =============================================================================

class PrefixSum {
private:
    vector<long long> prefix;

public:
    PrefixSum(const vector<int>& arr) {
        int n = arr.size();
        prefix.resize(n + 1, 0);
        for (int i = 0; i < n; i++) {
            prefix[i + 1] = prefix[i] + arr[i];
        }
    }

    // Range sum for [left, right]
    long long rangeSum(int left, int right) {
        return prefix[right + 1] - prefix[left];
    }
};

// Count subarrays with sum equal to k
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
// 4. Kadane's Algorithm
// =============================================================================

int maxSubArray(const vector<int>& nums) {
    int maxSum = nums[0];
    int currentSum = nums[0];

    for (size_t i = 1; i < nums.size(); i++) {
        currentSum = max(nums[i], currentSum + nums[i]);
        maxSum = max(maxSum, currentSum);
    }

    return maxSum;
}

// Also return indices of the maximum subarray
tuple<int, int, int> maxSubArrayWithIndices(const vector<int>& nums) {
    int maxSum = nums[0];
    int currentSum = nums[0];
    int start = 0, end = 0, tempStart = 0;

    for (size_t i = 1; i < nums.size(); i++) {
        if (nums[i] > currentSum + nums[i]) {
            currentSum = nums[i];
            tempStart = i;
        } else {
            currentSum += nums[i];
        }

        if (currentSum > maxSum) {
            maxSum = currentSum;
            start = tempStart;
            end = i;
        }
    }

    return {maxSum, start, end};
}

// =============================================================================
// 5. String Processing
// =============================================================================

// Palindrome check
bool isPalindrome(const string& s) {
    int left = 0, right = s.length() - 1;
    while (left < right) {
        if (s[left] != s[right]) return false;
        left++;
        right--;
    }
    return true;
}

// Anagram check
bool isAnagram(const string& s, const string& t) {
    if (s.length() != t.length()) return false;

    vector<int> count(26, 0);
    for (size_t i = 0; i < s.length(); i++) {
        count[s[i] - 'a']++;
        count[t[i] - 'a']--;
    }

    for (int c : count) {
        if (c != 0) return false;
    }
    return true;
}

// String compression
string compress(const string& s) {
    if (s.empty()) return s;

    string result;
    int count = 1;

    for (size_t i = 1; i <= s.length(); i++) {
        if (i < s.length() && s[i] == s[i - 1]) {
            count++;
        } else {
            result += s[i - 1];
            if (count > 1) {
                result += to_string(count);
            }
            count = 1;
        }
    }

    return result.length() < s.length() ? result : s;
}

// =============================================================================
// Test
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "Array and String Examples" << endl;
    cout << "============================================================" << endl;

    // 1. Two Pointers
    cout << "\n[1] Two Pointers" << endl;
    vector<int> sorted = {1, 2, 3, 4, 6};
    auto [idx1, idx2] = twoSumSorted(sorted, 6);
    cout << "    Array: [1,2,3,4,6], target: 6" << endl;
    cout << "    Indices: (" << idx1 << ", " << idx2 << ")" << endl;

    vector<int> heights = {1, 8, 6, 2, 5, 4, 8, 3, 7};
    cout << "    Container with most water: " << maxArea(heights) << endl;

    // 2. Sliding Window
    cout << "\n[2] Sliding Window" << endl;
    vector<int> arr = {1, 4, 2, 10, 23, 3, 1, 0, 20};
    cout << "    Max sum of window size 4: " << maxSumWindow(arr, 4) << endl;

    cout << "    \"abcabcbb\" longest substring: " << lengthOfLongestSubstring("abcabcbb") << endl;

    vector<int> nums = {2, 3, 1, 2, 4, 3};
    cout << "    Min length with sum >= 7: " << minSubArrayLen(7, nums) << endl;

    // 3. Prefix Sum
    cout << "\n[3] Prefix Sum" << endl;
    vector<int> prefixArr = {1, 2, 3, 4, 5};
    PrefixSum ps(prefixArr);
    cout << "    Array: [1,2,3,4,5]" << endl;
    cout << "    Range sum [1,3]: " << ps.rangeSum(1, 3) << endl;

    vector<int> subarr = {1, 1, 1};
    cout << "    Subarrays with sum 2: " << subarraySum(subarr, 2) << endl;

    // 4. Kadane's Algorithm
    cout << "\n[4] Kadane's Algorithm" << endl;
    vector<int> kadane = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    cout << "    Array: [-2,1,-3,4,-1,2,1,-5,4]" << endl;
    cout << "    Maximum subarray sum: " << maxSubArray(kadane) << endl;

    // 5. String Processing
    cout << "\n[5] String Processing" << endl;
    cout << "    \"racecar\" palindrome: " << (isPalindrome("racecar") ? "yes" : "no") << endl;
    cout << "    \"anagram\"/\"nagaram\" anagram: " << (isAnagram("anagram", "nagaram") ? "yes" : "no") << endl;
    cout << "    \"aabcccccaaa\" compressed: " << compress("aabcccccaaa") << endl;

    // 6. Complexity Summary
    cout << "\n[6] Complexity Summary" << endl;
    cout << "    | Algorithm       | Time       |" << endl;
    cout << "    |-----------------|------------|" << endl;
    cout << "    | Two Pointers    | O(n)       |" << endl;
    cout << "    | Sliding Window  | O(n)       |" << endl;
    cout << "    | Prefix Sum      | O(1) query |" << endl;
    cout << "    | Kadane's        | O(n)       |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
