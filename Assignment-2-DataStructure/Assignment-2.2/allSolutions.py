# Checking for Duplicate Elements  
class Solution:  
    def hasDuplicate(self, nums: List[int]) -> bool:  
        return len(set(nums)) != len(nums)  

# Finding the Missing Element  
class Solution:  
    def findMissing(self, nums: List[int]) -> int:  
        n = len(nums)  
        total = (n * (n + 1)) // 2  
        return total - sum(nums)  

# Identifying Absent Numbers in an Array  
class Solution:  
    def missingNumbers(self, nums: List[int]) -> List[int]:  
        seen = set(nums)  
        return [i for i in range(1, len(nums) + 1) if i not in seen]  

# Pair Sum Finder  
class Solution:  
    def locateTwoSum(self, nums: List[int], target: int) -> List[int]:  
        num_map = {}  
        for i, val in enumerate(nums):  
            if target - val in num_map:  
                return [i, num_map[target - val]]  
            num_map[val] = i  

# Counting Smaller Numbers  
class Solution:  
    def countSmallerNumbers(self, nums: List[int]) -> List[int]:  
        sorted_nums = sorted(nums)  
        rank = {val: i for i, val in enumerate(sorted_nums) if val not in rank}  
        return [rank[num] for num in nums]  

# Minimum Time to Travel Points  
class Solution:  
    def minTravelTime(self, points: List[List[int]]) -> int:  
        total_time = 0  
        x1, y1 = points.pop()  
        while points:  
            x2, y2 = points.pop()  
            total_time += max(abs(y2 - y1), abs(x2 - x1))  
            x1, y1 = x2, y2  
        return total_time  

# Spiral Matrix Traversal  
class Solution:  
    def matrixInSpiral(self, matrix: List[List[int]]) -> List[int]:  
        result = []  
        while matrix:  
            result.extend(matrix.pop(0))  
            if matrix and matrix[0]:  
                for row in matrix:  
                    result.append(row.pop())  
            if matrix:  
                result.extend(matrix.pop()[::-1])  
            if matrix and matrix[0]:  
                for row in reversed(matrix):  
                    result.append(row.pop(0))  
        return result  

# Counting Islands in a Grid  
class Solution:  
    def countIslands(self, grid: List[List[str]]) -> int:  
        if not grid:  
            return 0  

        def explore(r, c):  
            queue = deque()  
            visited.add((r, c))  
            queue.append((r, c))  

            while queue:  
                row, col = queue.popleft()  
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:  
                    nr, nc = row + dr, col + dc  
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1' and (nr, nc) not in visited:  
                        queue.append((nr, nc))  
                        visited.add((nr, nc))  

        rows, cols = len(grid), len(grid[0])  
        visited = set()  
        island_count = 0  

        for r in range(rows):  
            for c in range(cols):  
                if grid[r][c] == '1' and (r, c) not in visited:  
                    explore(r, c)  
                    island_count += 1  

        return island_count  

# Best Stock Trading Strategy  
class Solution:  
    def maximizeProfit(self, prices: List[int]) -> int:  
        left, right, max_profit = 0, 1, 0  
        while right < len(prices):  
            if prices[left] < prices[right]:  
                max_profit = max(max_profit, prices[right] - prices[left])  
            else:  
                left = right  
            right += 1  
        return max_profit  

# Squaring and Sorting an Array  
class Solution:  
    def squareAndSort(self, nums: List[int]) -> List[int]:  
        return sorted(x * x for x in nums)  

# Finding Unique Number  
class Solution:  
    def findUnique(self, nums: List[int]) -> int:  
        result = 0  
        for num in nums:  
            result ^= num  
        return result  

# Minimum Coins for a Given Amount  
class Solution:  
    def minCoinsNeeded(self, coins: List[int], amount: int) -> int:  
        dp = [float('inf')] * (amount + 1)  
        dp[0] = 0  
        for i in range(1, amount + 1):  
            for coin in coins:  
                if i >= coin:  
                    dp[i] = min(dp[i], 1 + dp[i - coin])  
        return dp[amount] if dp[amount] != float('inf') else -1  

# Steps to Climb a Staircase  
class Solution:  
    def waysToClimb(self, n: int) -> int:  
        if n == 1:  
            return 1  
        dp = [0] * (n + 1)  
        dp[1], dp[2] = 1, 2  
        for i in range(3, n + 1):  
            dp[i] = dp[i - 1] + dp[i - 2]  
        return dp[n]  

# Finding the Maximum Subarray Sum  
class Solution:  
    def maxSumSubarray(self, nums: List[int]) -> int:  
        dp = [0] * len(nums)  
        dp[0] = nums[0]  
        for i in range(1, len(nums)):  
            dp[i] = max(nums[i], dp[i - 1] + nums[i])  
        return max(dp)  

# Counting Bits in Binary Numbers  
class Solution:  
    def countBinaryBits(self, n: int) -> List[int]:  
        dp = [0] * (n + 1)  
        offset = 1  
        for i in range(1, n + 1):  
            if offset * 2 == i:  
                offset = i  
            dp[i] = 1 + dp[i - offset]  
        return dp  

# Immutable Range Sum Query  
class NumArray:  
    def __init__(self, nums: List[int]):  
        self.accumulated = [0]  
        for num in nums:  
            self.accumulated.append(self.accumulated[-1] + num)  

    def getRangeSum(self, left: int, right: int) -> int:  
        return self.accumulated[right + 1] - self.accumulated[left]  

# Generating Letter Case Permutations  
class Solution:  
    def generateCasePermutations(self, s: str) -> List[str]:  
        result = []  

        def backtrack(sub="", index=0):  
            if len(sub) == len(s):  
                result.append(sub)  
                return  
            if s[index].isalpha():  
                backtrack(sub + s[index].swapcase(), index + 1)  
            backtrack(sub + s[index], index + 1)  

        backtrack()  
        return result  


#78. Subsets
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtrack(start,path):
            res.append(path[:])
            for i in range(start,len(nums)):
                path.append(nums[i])
                backtrack(i+1,path)
                path.pop()
        res=[]
        backtrack(0,[])
        return res

#77. Combinations
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        def backtrack(start,path):
            if len(path)==k:
                result.append(path[:])
                return
            for i in range(start,n+1):
                path.append(i)
                backtrack(i+1,path)
                path.pop()
        result=[]
        backtrack(1,[])
        return result

#44. Permutations
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(start,end):
            if start==end:
                result.append(nums[:])
                return
            for i in range(start,end):
                nums[start],nums[i]=nums[i],nums[start]
                backtrack(start+1,end)
                nums[start],nums[i]=nums[i],nums[start]
        result=[]
        backtrack(0,len(nums))
        return result
    
#Linked List
#876. Middle of the Linked List
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow=fast=head
        while fast and fast.next:
            slow=slow.next
            fast=fast.next.next
        return slow

#141. Linked List Cycle
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow,fast=head,head
        while (fast) and fast.next:
            slow=slow.next
            fast=fast.next.next
            if slow==fast:
                return True
        return False

#206. Reverse Linked List      
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev=None
        curr=head
        while (curr!=None):
            next_pointer=curr.next
            curr.next=prev
            prev=curr
            curr=next_pointer
        return prev
#203. Remove Linked List Elements
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        if not head:
            return head
        while head.val==val:
            if head.next:
                head=head.next
            else:
                head=None
                return head
        curr=head.next
        prev=head
        while curr!=None:
            if curr.val==val:
                if curr.next==None:
                    prev.next=None
                prev.next=curr.next
                curr=curr.next
                continue
            curr=curr.next
            prev=prev.next
        return head
#or
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        dummy=ListNode(-1)
        dummy.next=head
        curr=dummy
        while curr.next!=None:
            if curr.next.val==val:
                curr.next=curr.next.next
            else:
                curr=curr.next
        return dummy.next

#92. Reverse Linked List II
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        dummy=ListNode(-1,head)
        left_prev,curr=dummy,head
        for i in range(left-1):
            left_prev,curr=curr,curr.next
        prev=None
        for i in range(right-left+1):
            next_ptr=curr.next
            curr.next=prev
            prev,curr=curr,next_ptr
        left_prev.next.next=curr
        left_prev.next=prev
        return dummy.next

#234. Palindrome Linked List
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        fast,slow=head,head
        while (fast) and (fast.next):
            fast=fast.next.next
            slow=slow.next
        
        prev=None
        while slow!=None:
            n_ptr=slow.next
            slow.next=prev
            prev=slow
            slow=n_ptr
        
        left=head
        right=prev
        while right!=None:
            if left.val!=right.val:
                return False
            left=left.next
            right=right.next
        return True

#21. Merge Two Sorted Lists
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        curr=ListNode(-1)
        dummy=curr
        while list1 and list2:
            if list1.val<list2.val:
                dummy.next=list1
                list1=list1.next
            else:
                dummy.next=list2
                list2=list2.next
            dummy=dummy.next
        if list1:
            dummy.next=list1
        if list2:
            dummy.next=list2
        return curr.next

#155. Min Stack
class MinStack:
    def __init__(self):
        self.stack=[]

    def push(self, val: int) -> None:
        if not self.stack:
            current_min=val
        else:
            current_min=min(val,self.stack[-1][1])
        self.stack.append((val,current_min))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]

#20. Valid Parentheses
class Solution:
    def isValid(self, s: str) -> bool:
        stack=[]
        hashmap={')':'(',   '}':'{',   ']':'['}
        for x in s:
            if stack and (x in hashmap and stack[-1]== hashmap[x]):
                stack.pop()
            else:
                stack.append(x)
        return not stack

#150. Evaluate Reverse Polish Notation
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack=[]
        for t in tokens:
            if t not in "+-*/":
                stack.append(int(t))
            else:
                r,l=stack.pop(),stack.pop()
                if t=="+":
                    stack.append(l+r)
                elif t=="-":
                    stack.append(l-r)
                elif t=="*":
                    stack.append(l*r)
                else:
                    stack.append(int(float(l)/r))
        return stack.pop()

#Sorting a Stack
def sortstack(stack):
    tmpstack=[]
    while stack:
        num=stack.pop()
        while (tmpstack and tmpstack[-1]<num):
            stack.append(tmpstack.pop())
        tmpstack.append(num)
    return tmpstack

#Queue
#225 Implement Stack using Queues
class MyStack:
    def __init__(self):
        self.queue=deque()

    def push(self, x: int) -> None:
        self.queue.append(x)
        
    def pop(self) -> int:
        for i in range(len(self.queue)-1):
            self.push(self.queue.popleft())
        return self.queue.popleft()

    def top(self) -> int:
        return self.queue[-1]

    def empty(self) -> bool:
        return len(self.queue)==0

#Reverse first K elements of Queue using stack
def reverse_first_k_elements(k,q):
    stack=[]
    for i in range(k):
        stack.append(q.popleft())
    while stack:
        q.append(stack.pop())
    for i in range(len(q)-k):
        q.append(q.popleft())
    return q

#Binary Trees
#637. Average of Levels in Binary Tree
class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        queue=deque([root])
        result=[]

        while queue:
            level=[]
            for i in range(len(queue)):
                node=queue.popleft()
                level.append(node.val)
                print(level)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(sum(level)/len(level))
        return result

#111. Minimum Depth of Binary Tree
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        queue=deque([(root,1)])
        while queue:
            node,level=queue.popleft()
            if not node.left and not node.right:
                return level
            if node.left:
                queue.append((node.left,level+1))
            if node.right:
                queue.append((node.right,level+1))
        return 0

#104. Maximum Depth of Binary Tree
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        queue=deque([(root,1)])
        while queue:
            node,level=queue.popleft()
            if node.right:
                queue.append((node.right,level+1))
            if node.left:
                queue.append((node.left,level+1))
        return level
    #or
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        queue=deque([(root,1)])
        return  max(self.maxDepth(root.left),self.maxDepth(root.right))+1

#Max/min value of binary tree
def largest(root):
    queue=deque([root])
    max_node=0
    while queue:
        curr_node=queue.popleft()
        if curr_node.left:
            queue.append(curr_node.left)
        if curr_node.right:
            queue.append(curr_node.right)
        if curr_node.val>max_node:
            max_node=curr_node.val
    return max_node

#102. Binary Tree Level Order Traversal
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root==None:
            return []
        queue=deque([root])
        tree=[]
        while queue:
            level=[]
            for i in range(len(queue)):
                node =queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            tree.append(level)
        return tree

#100. Same Tree
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        stack=[(p,q)]
        while stack:
            node1,node2=stack.pop()
            if not node1 and not node2:
                continue
            elif None in [node1,node2] or node1.val!= node2.val:
                return False
            stack.append((node1.right,node2.right))
            stack.append((node1.left,node2.left))
        return True

#112. Path Sum
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        stack=[(root,root.val)]
        while stack:
            curr,val=stack.pop()
            if not curr.left and not curr.right and val==targetSum:
                return True
            if curr.right:
                stack.append((curr.right,val+curr.right.val))
            if curr.left:
                stack.append((curr.left,val+curr.left.val))

        return False

#543. Diameter of Binary Tree
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.diameter=0
        def depth(root):
            if not root:
                return 0
            left_depth=depth(root.left)
            right_depth=depth(root.right)

            self.diameter=max(self.diameter,left_depth+right_depth)
            return 1+max(left_depth,right_depth)
        depth(root)
        return self.diameter
    
#226. Invert Binary Tree
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        stack=[root]
        while stack:
            curr=stack.pop()
            if curr:
                curr.left,curr.right=curr.right,curr.left
                stack.extend([curr.right,curr.left])
        return root

#236. Lowest Common Ancestor of a Binary Tree
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        queue=deque([root])
        parent={root:None}
        while queue:
            node=queue.popleft()
            if node.left:
                queue.append(node.left)
                parent[node.left]=node
            if node.right:
                queue.append(node.right)
                parent[node.right]=node
            if p in parent and q in parent:
                break
        ancestors=set()
        while p:
            ancestors.add(p)
            p=parent[p]
        while q:
            if q in ancestors:
                return q
            q=parent[q]

#700. Search in a Binary Search Tree
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        while root:
            if root.val==val:
                return root
            elif root.val<val:
                root=root.right
            else:
                root=root.left
        return None

#701. Insert into a Binary Search Tree
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        new_node=TreeNode(val)
        if not root:
            return new_node
        curr=root
        while True:
            if val<curr.val:
                if curr.left:
                    curr=curr.left
                else:
                    curr.left=new_node
                    break
            else:
                if curr.right:
                    curr=curr.right
                else:
                    curr.right=new_node
                    break
        return root

#108. Convert Sorted Array to Binary Search Tree
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        mid=len(nums)//2
        root=TreeNode(nums[mid])
        root.left=self.sortedArrayToBST(nums[:mid])
        root.right=self.sortedArrayToBST(nums[mid+1:])
        return root

#653. Two Sum IV - Input is a BST
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        queue=deque([root])
        num=set()
        while queue:
            node=queue.popleft()
            if (k-node.val) in num:
                return True
            else:
                num.add(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return False

#235. Lowest Common Ancestor of a Binary Search Tree
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        small=min(p.val,q.val)
        large=max(p.val,q.val)
        while root:
            if root.val>large:
                root=root.left
            elif root.val<small:
                root=root.right
            else:
                return root
        return None

#530. Minimum Absolute Difference in BST
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        mindiff=float('inf')
        prev_val=float('-inf')
        stack=[]
        while root or stack:
            if root:
                stack.append(root)
                root=root.left
            else:
                root=stack.pop()
                mindiff=min(mindiff,root.val-prev_val)
                prev_val=root.val
                root=root.right
        return mindiff

#1382. Balance a Binary Search Tree
class Solution:
    def balanceBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def inorder_traversal(node: Optional[TreeNode]) -> List[int]:
            if not node:
                return []
            return inorder_traversal(node.left) + [node.val] + inorder_traversal(node.right)

        def sorted_list_to_bst(start: int, end: int) -> Optional[TreeNode]:
            if start > end:
                return None
            mid = (start + end) // 2
            root = TreeNode(values[mid])  
            root.left = sorted_list_to_bst(start, mid - 1) 
            root.right = sorted_list_to_bst(mid + 1, end)  
            return root

        values = inorder_traversal(root)

        return sorted_list_to_bst(0, len(values) - 1)

#450. Delete Node in a BST
class Solution:
     def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root:
            return None
        
        parent = None
        current = root

        while current and current.val != key:
            parent = current
            if key < current.val:
                current = current.left
            else:
                current = current.right

        if not current:
            return root

        if not current.left and not current.right:
            if not parent:
                return None 
            if parent.left == current:
                parent.left = None
            else:
                parent.right = None

        elif not current.left or not current.right:
            child = current.left if current.left else current.right
            if not parent:
                return child  
            if parent.left == current:
                parent.left = child
            else:
                parent.right = child

        else:
            successor_parent = current
            successor = current.right
            while successor.left:
                successor_parent = successor
                successor = successor.left
            
            current.val = successor.val
            if successor_parent.left == successor:
                successor_parent.left = successor.right
            else:
                successor_parent.right = successor.right
        
        return root

#230. Kth Smallest Element in a BST
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val 
            
            root = root.right 

#Heap
#215. Kth Largest Element in an Array
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return heapq.nlargest(k,nums)[-1]
#or
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap=[]
        for i in nums:
            heapq.heappush(heap,i)
        for i in range(len(nums)-k):
            heapq.heappop(heap)
        return heapq.heappop(heap)

#K Closest
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap=[]
        for (x,y) in points:
            dist=-(x*x+y*y)
            if len(heap)==k:
                heapq.heappushpop(heap,(dist,x,y))
            else:
                heapq.heappush(heap,(dist,x,y))
        return [(x,y) for (dist,x,y) in heap]

#347. Top
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count=Counter(nums)
        heap=[]
        for num,freq in count.items():
            if len(heap)<k:
                heapq.heappush(heap,(freq,num))
            elif freq>heap[0][0]:
                heapq.heapreplace(heap,(freq,num))
        top_k=[num for freq,num in heap]
        return top_k

#Task 
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        counts=Counter(tasks)
        heap=[]
        for count in counts.values():
            heap.append(-count)
        heapq.heapify(heap)
        time=0
        wait=deque()
        while heap or wait:
            time+=1
            if heap:
                cur=heapq.heappop(heap)
                cur+=1
                if cur!=0:
                    wait.append((cur,time+n))
            if wait and wait[0][1]==time:
                heapq.heappush(heap,wait.popleft()[0])
        return time
    
#Clone Graph
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return node
        queue=deque([node])
        clones={node.val:Node(node.val)}
        while queue:
            curr=queue.popleft()
            curr_clone=clones[curr.val]

            for neighbor in curr.neighbors:
                if neighbor.val not in clones:
                    clones[neighbor.val]=Node(neighbor.val)
                    queue.append(neighbor)
                curr_clone.neighbors.append(clones[neighbor.val])
        return clones[node.val]

#Cheapest Flights Within K Stops
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        prices = [float("inf")] * n
        prices[src] = 0

        for i in range(k + 1):
            tmpPrices = prices.copy()

            for from_node, to_node, cost in flights:
                if prices[from_node] == float("inf"):
                    continue
                if prices[from_node] + cost < tmpPrices[to_node]:
                    tmpPrices[to_node] = prices[from_node] + cost

            prices = tmpPrices

        if prices[dst] == float("inf"):
            return -1
        else:
            return prices[dst]

#Course Schedule
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adj = {course: [] for course in range(numCourses)}
        for course, pre in prerequisites:
            adj[course].append(pre)

        for course in range(numCourses):
            stack = [(course, set())]
            while stack:
                cur_course, visited = stack.pop()
                if cur_course in visited:
                    return False  
                visited.add(cur_course)
                for pre in adj[cur_course]:
                    stack.append((pre, visited.copy()))
            adj[course] = []  

        return True
        