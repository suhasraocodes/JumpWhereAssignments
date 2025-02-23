#include <bits/stdc++.h>
using namespace std;

bool ValidAnagram(string s1, string s2)
{

    unordered_map<char, int> mp;

    transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
    transform(s2.begin(), s2.end(), s2.begin(), ::tolower);

    if (s1.length() != s2.length())
        return false;

    for (int i = 0; i < s1.size(); i++)
    {
        mp[s1[i]]++;
    }
    for (int i = 0; i < s2.size(); i++)
    {
        mp[s2[i]]--;
    }
    for (auto itr : mp)
    {
        if (itr.second != 0)
        {
            return false;
        }
    }
    return true;
}

int main()
{
    string s1, s2;
    cin >> s1 >> s2;
    cout << ValidAnagram(s1, s2);
    return 0;
}
