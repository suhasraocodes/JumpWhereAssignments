#include <bits/stdc++.h>
using namespace std;

int canCompleteCircuit(vector<int> gas, vector<int> cost)
{
    const int gasSum = accumulate(gas.begin(), gas.end(), 0);
    const int costSum = accumulate(cost.begin(), cost.end(), 0);
    if (gasSum - costSum < 0)
        return -1;

    int ans = 0;
    int sum = 0;

    for (int i = 0; i < gas.size(); ++i)
    {
        sum += gas[i] - cost[i];

        if (sum < 0)
        {
            sum = 0;
            ans = i + 1;
        }

        return ans;
    }
};

int main()
{
    vector<int> gas = {1, 2, 3, 4, 5};
    vector<int> cost = {3, 4, 5, 1, 2};

    int result = canCompleteCircuit(gas, cost);

    cout << "Starting index: " << result << endl;

    return 0;
}
