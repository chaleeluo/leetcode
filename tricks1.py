# 场景一： 列表转整数   int(''.join(str(t) for t in 列表)) ######################################################
#         整数按位拆开 int(x) for x in str(整数)          ######################################################
# 66. 加一
'''
输入: [4,3,2,1]
输出: [4,3,2,2]
解释: 输入数组表示数字 4321
'''
def plusone(nums):
	i = len(nums) - 1
	if nums[i] < 9:
		nums[i] += 1
		return nums
	else:
		newnum = int(''.join(str(t) for t in nums)) + 1
		return [int(x) for x in str(newnum)]


# 场景二： 同时获取索引和元素  for i,num in enumerate(列表)   ####################################################
#         同时获取字典的键值  for key,val in 字典.items()    ####################################################
#         对字符串出现的字母计数：collections.Counter(字符串) ####################################################
# 1. 两数之和
'''
输入: nums = [2, 7, 11, 15], target = 9
输出: [0, 1]
解释: nums[0] + nums[1] = 2 + 7 = 9
'''
def twosum(nums, target):
	d = {}
	for i, num in enumerate(nums):
		if num in d:
			return [i, d[num]]
		else:
			d[target - num] = i

# 387. 字符串中的第一个唯一字符
'''
输入: "loveleetcode"
输出: 2
'''
def firstuniqchar(s):
	import collections
	d = collections.Counter(s)
	for i in range(len(s)):
		if d[s[i]] == 1:
			return i
	return -1


# 场景三： 字典返回对应的值或者不存在返回-1     字典.get(键, -1)  #############################################
#         字符串返回索引或者不存在返回-1     str.find(str, beg=0, end=len(string)) ##########################
# 496. 下一个更大元素 I : 找到 nums1 中每个元素在 nums2 中的下一个比其大的值。
'''
输入:
nums1 = [4,1,2], nums2 = [1,3,4,2].
输出: [[-1,3,-1]
'''
def nextgreater(nums1, nums2):
  l = []
  d = {}
  for n in nums2:
    while len(l) and l[-1] < n:           #遍历nums2 同时遍历l
      d[l.pop()] = n
    l.append(n)

  for i in range(len(nums1)):
    nums[i] = d.get(nums[i], -1)          #原地改
  return nums1
# 28. 实现strStr()：在 haystack 字符串中找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。
'''
输入:
haystack = "hello", needle = "ll"
输出:
2
'''
def strStr(haystack, needle):               # return haystack.find(needle)
  l = len(needle)
  for i in range(len(haystack)-l+1):
      if haystack[i:i+l] == needle:
          return i
  return -1


# 场景四： 通过i,j来推断这个点在哪个3*3网格内   3*(i//3)+(j//3) ###################################################
# 36. 有效的数独 : 判断一个 9x9 的数独是否有效。只需要根据以下规则，验证已经填入的数字是否有效即可。
# 数字 1-9 在每一行只能出现一次。数字 1-9 在每一列只能出现一次。数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
'''
输入:
[
  ["5","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
输出: true
'''
def isValidSudoku(board):
  dic_row = [{},{},{},{},{},{},{},{},{}] 
  dic_col = [{},{},{},{},{},{},{},{},{}] 
  dic_box = [{},{},{},{},{},{},{},{},{}] 
  
  for i in range(len(board)):
      for j in range(len(board)):
          num = board[i][j]
          if num == ".":
              continue
          if num not in dic_row[i] and num not in dic_col[j] and num not in dic_box[3*(i//3)+(j//3)]:
              dic_row[i][num] = 1
              dic_col[j][num] = 1
              dic_box[3*(i//3)+(j//3)][num] = 1 # 利用地板除 向下取余,将矩阵划分为九块
          else:
              return False
  return True








