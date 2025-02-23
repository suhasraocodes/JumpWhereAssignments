#include <bits/stdc++.h>
using namespace std;

bool isValid(string s)
{
    stack<char> st;
    for (auto ch : s)
    {
        if (ch == '{' || ch == '(' || ch == '[')
        {
            st.push(ch);
        }
        else if (ch == '}')
        {
            if (st.empty())
            {
                return false;
            }
            else if (st.top() == '{')
            {
                st.pop();
            }
            else
            {
                return false;
            }
        }
        else if (ch == ')')
        {
            if (st.empty())
            {
                return false;
            }
            else if (st.top() == '(')
            {
                st.pop();
            }
            else
            {
                return false;
            }
        }
        else if (ch == ']')
        {
            if (st.empty())
            {
                return false;
            }
            else if (st.top() == '[')
            {
                st.pop();
            }
            else
            {
                return false;
            }
        }
    }
    if (st.empty())
    {
        return true;
    }
    return false;
}

int main()
{
    string s;
    cin >> s;
    cout << boolalpha << isValid(s);
    return 0;
}