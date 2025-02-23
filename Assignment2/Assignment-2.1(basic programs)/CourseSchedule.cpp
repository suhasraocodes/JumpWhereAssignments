#include <iostream>
#include <vector>
#include <queue>

using namespace std;

class Solution
{
public:
    bool canFinish(int numCourses, vector<vector<int>> &prerequisites)
    {
        vector<vector<int>> graph(numCourses);
        vector<int> inDegree(numCourses);

        for (auto &prerequisite : prerequisites)
        {
            int course = prerequisite[0];
            int prereq = prerequisite[1];
            graph[prereq].push_back(course);
            ++inDegree[course];
        }

        queue<int> noPrereqCourses;
        for (int i = 0; i < numCourses; ++i)
        {
            if (inDegree[i] == 0)
            {
                noPrereqCourses.push(i);
            }
        }

        int finishedCount = 0;
        while (!noPrereqCourses.empty())
        {
            int current = noPrereqCourses.front();
            noPrereqCourses.pop();
            ++finishedCount;

            for (int neighbor : graph[current])
            {
                if (--inDegree[neighbor] == 0)
                {
                    noPrereqCourses.push(neighbor);
                }
            }
        }

        return finishedCount == numCourses;
    }
};

int main()
{
    Solution solution;

    int numCourses = 4;
    vector<vector<int>> prerequisites = {{1, 0}, {2, 1}, {3, 2}};

    if (solution.canFinish(numCourses, prerequisites))
    {
        cout << "All courses can be finished." << endl;
    }
    else
    {
        cout << "Not possible to finish all courses." << endl;
    }

    return 0;
}
