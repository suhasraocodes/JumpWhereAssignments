#include <bits/stdc++.h>
using namespace std;

int KthLargest(vector<int> &n, int k)
{
    priority_queue<int, vector<int>, greater<int>> minHeap;

    for (int num : n)
    {
        minHeap.push(num);
        if (minHeap.size() > k)
        {
            minHeap.pop();
        }
    }
    return minHeap.top();
}

int main()
{
    int n, k;
    cin >> n;
    vector<int> inp(n);
    for (int i = 0; i < n; i++)
    {
        cin >> inp[i];
    }
    cin >> k;

    cout << KthLargest(inp, k) << endl;
    return 0;
}
