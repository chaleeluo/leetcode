# 问题一： 53. 最大子序和 ###########################################################################
'''
输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
'''

def maxSubArray(nums):

    for i in range(1, len(nums)):
        nums[i]= nums[i] + max(nums[i-1], 0)
        print(nums)
    return max(nums)



# 问题二：152. 乘积最大子序列 ###########################################################################
'''
输入: [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
'''

def maxProduct(nums):
    if nums is None: return 0

    dp = [[0 for _ in range(2)] for _ in range(2)]
    dp[0][1] = dp[0][0] = res = nums[0]                                  # 第一位用01滚动0~n范围数组   第二位表示正负

    for i in range(1, len(nums)):
        x, y = i % 2, (i - 1) % 2                                        # 滚动数组
        dp[x][0] = max(dp[y][0] * nums[i], dp[y][1] * nums[i], nums[i])  # 最大值更新
        dp[x][1] = min(dp[y][0] * nums[i], dp[y][1] * nums[i], nums[i])  # 最小值更新（防止乘以负数的情况）
        res = max(res, dp[x][0])
    return res

def maxProduct(nums):   # 把上面的二维数组DP[][]用curMax,curMin表示更加快速
    if nums is None: return 0

    curmin = curmax = res = nums[0]
    for i in range(1, len(nums)):
        curmin, curmax = curmin * nums[i], curmax * nums[i]
        curmin, curmax = min(curmin, curmax, nums[i]), max(curmin, curmax, nums[i])
        res = max(res, curmax)
    return res

def maxProduct(A):      #最快的方法
    '''
    先计算从左到右的相乘的最大值，再计算从右到左的最大值；再将两组最大值相比
    '''
    B = A[::-1]
    for i in range(1, len(A)):
        A[i] *= A[i - 1] or 1
        B[i] *= B[i - 1] or 1
    return max(max(A),max(B))



# 问题三：300. 最长上升子序列 ###########################################################################
'''
输入: [10,9,2,5,3,7,101,18]
输出: 4 
解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
'''

def largestlen(nums):
    '''求长度'''
    if len(nums) <= 1:
        return len(nums)

    dp = [1] * len(nums)                       # 定义初始状态
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)  # 到当前状态的最长子序列
    return max(dp)

def largestarr(arr):
    '''拓展：求序列'''
    res = []
    dp = [0] * len(arr)                        # 定义初始状态
    n = len(arr)
    for i in range(n - 2, -1, -1):             #从后往前，同时满足两个条件+1
        for j in range(n - 1, i, -1):
            if arr[i] < arr[j] and dp[i] <= dp[j]:
                dp[i] += 1                     #保存当前到末尾位置的最长序列长度-1
        # print(dp)
    maxnum = max(dp)
    # print(dp, maxnum)
    for i in range(n):
        if dp[i] == maxnum:
            res.append(arr[i])
            maxnum -= 1
    return res

# [0, 0, 0, 0, 0, 0, 0, 0]
# [0, 0, 0, 0, 0, 1, 0, 0]
# [0, 0, 0, 0, 2, 1, 0, 0]
# [0, 0, 0, 2, 2, 1, 0, 0]
# [0, 0, 3, 2, 2, 1, 0, 0]
# [0, 1, 3, 2, 2, 1, 0, 0]
# [1, 1, 3, 2, 2, 1, 0, 0]
# [1, 1, 3, 2, 2, 1, 0, 0] 3
# out:[2, 5, 7, 101]


