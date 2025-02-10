#include <bits/stdc++.h>
using namespace std;

class Solution
{
public:
    string minWindow(string s, string t)
    {
        if (t.empty())
            return "";

        unordered_map<char, int> countT, window;
        for (char c : t)
            countT[c]++;

        int have = 0, need = countT.size();
        int left = 0, minLength = INT_MAX, startIdx = 0;

        for (int right = 0; right < s.size(); right++)
        {
            char c = s[right];
            window[c]++;

            if (countT.find(c) != countT.end() && window[c] == countT[c])
            {
                have++;
            }

            while (have == need)
            {
                if (right - left + 1 < minLength)
                {
                    minLength = right - left + 1;
                    startIdx = left;
                }

                window[s[left]]--;
                if (countT.find(s[left]) != countT.end() && window[s[left]] < countT[s[left]])
                {
                    have--;
                }
                left++;
            }
        }

        return minLength == INT_MAX ? "" : s.substr(startIdx, minLength);
    }
};

int main()
{
    Solution sol;
    string s = "ADOBECODEBANC", t = "ABC";
    cout << "Minimum window substring: " << sol.minWindow(s, t) << endl;
    return 0;
}
