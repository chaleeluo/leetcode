# 问题一： 64. 最小路径和 ###########################################################################
'''
给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
输入: [
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 7
解释: 因为路径 1→3→1→1→1 的总和最小。
'''
def minPathSum(grid) -> int:
        for i in range(1,len(grid[0])):           #先处理第一行第一列，都在原始矩阵上面修改
            grid[0][i] = grid[0][i] + grid[0][i-1]
        for i in range(1,len(grid)):
            grid[i][0] = grid[i][0] + grid[i-1][0]
        for i in range(1,len(grid)):              #双重循坏遍历每个节点
            for j in range(1,len(grid[0])):
                grid[i][j] = grid[i][j] + min(grid[i][j-1],grid[i-1][j])
        print(grid)
        return grid[-1][-1]  



# 问题一： 62. 不同路径 ###########################################################################
'''
一个机器人位于一个 m x n 网格的左上角,机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角
问总共有多少条不同的路径？
输入: m = 3, n = 2
输出: 3
解释:从左上角开始，总共有 3 条路径可以到达右下角。
1. 向右 -> 向右 -> 向下
2. 向右 -> 向下 -> 向右
3. 向下 -> 向右 -> 向右
'''
# 思路一：排列组合 ###############
# 比如m=3, n=2，我们只要向下 1 步，向右 2 步就一定能到达终点。
# C_{m + n - 2} ^{m - 1} 或者 C_{m + n - 2} ^{n - 1} 
import math 
def Paths(m, n):
        return int(math.factorial(m+n-2)/math.factorial(m-1)/math.factorial(n-1))

# 思路二：dp ###############
# 动态方程：dp[i][j] = dp[i-1][j] + dp[i][j-1]
# 注意，对于第一行 dp[0][j]，或者第一列 dp[i][0]，由于都是在边界，所以只能为 1
# 时间复杂度：O(m*n) 空间复杂度：O(m * n)
def Paths(m, n):
    dp = [[1]*n] + [[1]+[0] * (n-1) for _ in range(m-1)]
    # print(dp)
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    # print(dp)
    return dp[-1][-1]
# Paths(4,3)
# [[1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
# [[1, 1, 1], [1, 2, 3], [1, 3, 6], [1, 4, 10]]
# 10

# 优化：其实每次只需要 dp[i-1][j],dp[i][j-1]
# 优化1：空间复杂度 O(2n)
def Paths(m, n):
    pre = [1] * n
    cur = [1] * n
    # print(pre, cur)
    for i in range(1, m):
        for j in range(1, n):
            cur[j] = pre[j] + cur[j-1]
        pre = cur[:]
        # print(pre, cur)
    return pre[-1]
# Paths(4,3)
# [1, 1, 1] [1, 1, 1]
# [1, 2, 3] [1, 2, 3]
# [1, 3, 6] [1, 3, 6]
# [1, 4, 10] [1, 4, 10]
# 10
# 优化1：空间复杂度 O(n)
def Paths(m, n):
    cur = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            cur[j] += cur[j-1]
            print(cur)
    return cur[-1]
# Paths(4, 3)
# [1, 2, 1]
# [1, 2, 3]
# [1, 3, 3]
# [1, 3, 6]
# [1, 4, 6]
# [1, 4, 10]
# 10







