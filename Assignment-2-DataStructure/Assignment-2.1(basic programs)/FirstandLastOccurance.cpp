#include <bits/stdc++.h>
using namespace std;

vector<int> Occurances(vector<int> n, int target)
{
    int flag = 0;
    vector<int> res;
    for (int i = 0; i < n.size() - 1; i++)
    {
        if (n[i] == target && flag == 0)
        {
            res.push_back(i);
            flag = 1;
        }
        if (n[i] == target && n[i + 1] != target)
        {
            res.push_back(i);
        }
    }

    if ((n[n.size() - 1] == target) && flag == 1)
    {
        res.push_back(n.size() - 1);
    }
    else if ((n[n.size() - 1] == target) && flag == 0)
    {
        res.push_back(n.size() - 1);
        res.push_back(n.size() - 1);
        flag = 1;
    }
    if (flag == 0)
    {
        res.push_back(-1);
        res.push_back(-1);
    }
    return res;
}

int main()
{
    vector<int> inp;
    int n, t;
    cin >> n;
    for (int i = 0; i < n; i++)
    {
        int x;
        cin >> x;
        inp.push_back(x);
    }
    cin >> t;
    inp = Occurances(inp, t);
    for (auto it : inp)
    {
        cout << it << " ";
    }
    return 0;
}