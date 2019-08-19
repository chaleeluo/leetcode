
# 时间复杂度对比 https://www.bigocheatsheet.com/

################################### 一/链表linked list/数组array #######################################
# 206 反转链表
def function(self, head):
	cur, pre = head, None
	while cur:
		cur.next, pre, cur = pre, cur, cur.next
	return pre

# 24 链表交换相邻元素：两两交换需要记录三个节点
def function(self, head):
	pre, pre.next = self, head
	while pre.next and pre.next.next:
		a, b = pre.next, pre.next.next
		pre.next, b.next, a.next = b, a, b.next
		pre = a
	return self.next           # 注意返回值
	
# 141 判断链表是否有环
# 方法1看是否能走到最后空指针，方法2所有指针存放set判重， 方法3快慢指针不用存放所有指针（时间也是On 但空间优化）
def function(self, head):
	fast = slow = head
	while slow and fast and fast.next:
		slow = slow.next
		fast = fast.next.next
		if slow is fast:
			return True
	return False

# 扩展：判断是否有环2；反转k组节点

############################################ 二/堆栈stack/队列queue ###################################
# 20 大中小括号是否合法
def function(self, s):
	stack = []
	par_map = {')':'(', ']':'[', '}':'{' }
	for c in s:
		if c not in paren_map:
			stack.append(c)
		elif not stack or par_map[c] != stack.pop():
			return False
	return not stack

# 232 225 堆栈实现队列/队列实现堆栈 --> 负负得正

# 优先队列priority queue
# 实现（goole搜索查看多种结构堆）：堆heap（binanry，binomial，fibonacci）/二叉搜索树 binanry search tree

# 703 实时判断数据流中第k大元素 
# 方法1： 依次排序 N* K*logK
# 方法2： 维护大小为k的最小堆 N* (1 OR log(2)K)
import heapq
class kthLargest(object):
	"""docstring for ClassName"""
	def __init__(self, k, nums):
		super(kthLargest, self).__init__()
		self.nums = nums
		self.size = len(nums)
		self.k = k
		heapq.heapify(self.nums)
		while self.size > self.k:
			heapq.heapify(self.nums)
			self.size -= 1
	def add(self, val):
		if self.size < self.k:
			heapq.heappop(self.nums)
			self.size -= 1
		elif self.nums[0] < val:
			heapq.heapreplace(self.nums, val)
		return self.nums[0]


# 239 返回滑动窗口最大值
# 方法1： 维护最大堆 N* logK
# 方法2： 维护双端队列deque   N
def function(self, nums, k):
	if not nums: return[]
	win, res = [], []
	for i, x in enumerate(nums):
		if i >= k and win[0] <= i-k:  #元素超出了左界则踢出
			win.pop(0)
		while win and nums[win[-1]] <= x:  #窗口中最大的都比x小则全部踢出
			win.pop()
		win.append(i)
		if i >= k-1:
			res.append(nums[win[0]])
	return res

############################################ 三/映射map/集合set ###################################
# 239 两个字符串字母是否相同
# 方法1： 排序 N* logN
return sorted(s) == sorted(t)
# 方法2： map计数   N
def function(self, s, t):
	dic1, dic2 = {}, {}               #直接用字典
	for i in s:
		dic1[i] = dic1.get(i, 0) + 1
	for i in t:
		dic2[i] = dic2.get(i, 0) + 1
	return dic1 == dic2


def function(self, s, t):
	dic1, dic2 = [0]*26, [0]*26       #数组表示字典, 手建哈希表
	for i in s:
		dic1[ord(i) - ord('a')] += 1
	for i in t:
		dic2[ord(i) - ord('a')] += 1
	return dic1 == dic2


# 两数之和  15三数之和 18四数之和
# 方法1： set查询-(a+b)是否存在 N* N
# 方法2： sortfind 先排序后查找 N* N但不需要重开一个数组
# 		 loop a：找b和c

def threeSum(self, nums):
	if len(nums) < 3: return[]
	nums.sort()
	res = set()
	for i, v in enumerate(nums[:-2]):
		if i > 0 and v == nums[i-1]:   # 相同数字跳过
			continue
		d = {}
		for x in nums[i+1:]:
			if x not in d:
				d[-v-x] = 1
			else:
				res.add((v, -v-x, x))     # set的添加
	return map(list, res)


def threeSum(self, nums):
	res = []
	nums.sort()
	for i in range(len(nums) - 2):
		if i > 0 and nums[i] == nums[i-1]:
			continue
		l, r = i+1, len(nums)-1
		while l < r:
			s = nums[i] + nums[l] + nums[r]
			if s < 0: l += 1
			elif s > 0: r -= 1
			else:
				res.append((nums[i], nums[l], nums[r]))
				while l < r and nums[l] == nums[l+1]:     #对重复的直接跳过
					l += 1
				while l < r and nums[r] == nums[r-1]:
					r -= 1
				l += 1
				r -= 1
	return map(list, res)

############################################ 四/树tree #####################################
# 二叉搜索树：有序/排序二叉树，可以是空树或者左子树节点<根节点<有子树节点
# 平衡二叉树：左右子树高度差不超过1，可以是空树

# 98 验证二叉搜索树
# 方法1： 中序遍历--是否升序，可以直接只判断当前节点是否大于之前的   N
# 方法2： 递归：左子树的最大值 < 根节点 > 右子树的最小值           N
def isValidBST(self, root):          #方法1 法a
	inorder = self.inorder(root)
	return inorder == list(sorted(set(inorder)))

def inorder(self, root):
	if not root:
		return []
	return self.inorder(root.left) + [root.val] + self.inorder(root.right)

# 排序只查看当前节点和前继节点大小关系     #方法1 法b
def isValidBST(self, root):
	self.prev = None
	return self.helper(root)

def helper(self, root):       #这里也是一个中序遍历！
	if not root:
		return True
	if not self.helper(root.left):
		return False
	if self.prev and self.prev.val >= root.val:
		return False
	self.prev = root
	return self.helper(root.right)

#方法2
def isValidBST(self, root):
	if not root:
		return True
	return self.isVal(root, -(2**32), 2**32)

def isVal(self, root, minv, maxv): 
	if not root.left:
		return minv < root.left < root.val
	if not root.right:
		return root.val < root.right < maxv
	return self.isVal(root.left, minv, root.val) and self.isVal(root.right, root.val, maxv)



# 235 236 二叉搜索树的最近公共祖先
def lowest(self, root, p, q):     # 递归写法
	while root:
		if p.val < root.val > q.val:
			root = root.left
		if p.val > root.val < q.val:
			root = root.right
		else:
			return root

def lowest(self, root, p, q):     # 非递归写法
	while root:
		if p.val < root.val > q.val:
			return self.lowest(root.left, p, q)
		if p.val > root.val < q.val:
			return self.lowest(root.left, p, q)
		return root

# 前中后序遍历（实际一般不用，一般是用dfs/bfs搜索）
def preorder():
	if root :
		res.append(root.val)
		res += self.preorder(root.left)        # 前中后序列遍历只需要修改这三行代码
		res += self.preorder(root.right)

############################## 五/递归recursion/分治divide & conquer ##########################
# 169 求众数 保证都有（ > n/2 ）
# 方法1： 暴力   N*N
# 方法2： map    N
# 方法3： 排序   N * logN
# 方法4： 分治   N * logN 比较左右的众数是否相等，不相等的话计数比较大小

# 50 x的n次方
# 方法1： 调库pow(x, n)
# 方法2： 暴力 N
# 方法3： 分治
def myPow(self, x, n):        # 递归
	if not n: return 1
	if n < 0:
		return 1 / self.myPow(x, -n)
	if n % 2:
		return x * self.myPow(x, n-1)  #奇数的话多计算一次
	return self.myPow(x*x, n/2)

def myPow(self, x, n):        # 非递归(使用位运算)
	if n < 0:
		x = 1 / x
		n = -n
	pow = 1
	while n:
		if n & 1:        # 当前二进制位
			pow *= x
		x *= x           # 自身相乘
		n >>= 1          # n右移一位，相当于除以2
	return pow           

################################################# 六/位运算 ##############################################
# 符号：与& 或| 异或^ 取反~ 左移<< 右移>>
# 异或XOR：相同为0， 不同为1（可以用“不进位加法”理解）
x ^ x = 0
x ^ 0 = 0 
x ^ 1s = ~x    #  ( 1s即为所有数均为1的二进制数，1s = ~0)
x ^ (~x) = 1s

a ^ b = c    ## swap: 既有 a ^ c = b, b ^ c = a
a ^ b ^ c =  (a ^ b) ^ c = a ^ (b ^ c)   ## associative
# 与小trick
判断奇偶：       x & 1 == 0 等价于 x % 2 == 1
清零最低位的1：   x = x & (x-1)
得到最低位的1：   x & -x   # -x即为取反再加1

# 191 统计位1的个数
# 方法1： 每一位模2为1则计数，然后移位 x=x>>1
# 方法2： x = x & (x-1) 每次打掉最后一个1，只要不等于全0则继续计数
def hammingWeight(self, n):
	res = 0
	mask = 1
	for i in range(32):
		if n & mask:
			res += 1
		n = n << 1
	return res

def hammingWeight(self, n):
	while n > 0:
		n &= (n-1)
	return res

# 231：2的幂次方问题
# 方法1： 模2
# 方法2： log(2)看是否是整数
# 方法3： 位运算---2的幂只有最高位上是1
return n > 0 and n & (n-1) == 0

# 338：比特位计数--计算二进制数中的1的数目并将它们作为数组返回
# 方法1： 依次对每个数字计数
# 方法2： 去掉最低位的1
def function(n):
	res = [0]*(n+1)
	for i in range(1, n):
		res[i] = res[i & (i-1)] + 1   # i清零最低位的1后计数比之前的i少一个

# 52 N皇后：返回 n 皇后不同的解决方案的数量
class Solution:
    def totalNQueens(self, n: int) -> int:
        def DFS(n: int, row: int, cols: int, left: int, right: int):
            """ 深度优先搜索
            :param n: N皇后个数
            :param row: 递归的深度
            :param cols: 可被攻击的列
            :param left: 左侧斜线上可被攻击的列
            :param right: 右侧斜线上可被攻击的列
            """
            if row >= n:            # 终止条件：所有层遍历完毕
                self.res += 1
                return

            # 获取当前可用的空间
            bits = (~(cols | left | right)) & ((1 << n) - 1)

            # 遍历可用空间
            while bits:
                # 获取一个位置
                p = bits & -bits
                DFS(n, row + 1, cols | p, (left | p) << 1, (right | p) >> 1)
                bits = bits & (bits - 1)

        if not (n == 1 or n >= 4):
            # N皇后问题只有在 N 大于等于 4 或等于 1 的时候才有解
            return 0
        self.res = 0
        DFS(n, 0, 0, 0, 0)
        return self.res


################################################# 七/贪心算法greedy #######################################
# 问题能够分成子问题最优解 但不能回退
# 122 买卖股票---贪心：交易无数次
def maxProfit(self, prices):
    i = res = 0
    while i < len(prices)-1:
        if prices[i] < prices[i+1]:
            res += prices[i+1]-prices[i]
        i += 1
    return res


################################################# 八/DFS & BFS ##########################################
# bfs适合非递归写法，dfs适合递归写法
# 102 树的层次遍历
import collections

def function(self, root):             # bfs
	if not root: return []
	res = []
	queue = collections.deque()
	queue.append(root)
	# seen = set(root)    如果该题不是树，而是图，一定要记得记录访问过的节点

	while queue:
		cur = []                      #准备一个数组
		for _ in range(len(queue)):
			node = queue.popleft()
			cur.append(node.val)
			if node.left: queue.append(node.left)
			if node.right: queue.append(node.right)
		res.append(cur)
	return res


def function(self, root):                 # dfs
	if not root: return []
	self.res = []
	self._dfs(root, 0)
	return self.res
def _dfs(self, node, level):
	if not node: return
	if len(self.res) < level + 1:   # 说明当前层没有加入到结果中
		self.res.append([])
	self.res[level].append(node.val)  

	self._dfs(node.left, level+1)
	self._dfs(node.right, level+1)

# 104 111 树的最大最小深度
def maxlength(self, root):                # 递归
	if not root: return 0
	return 1 + max(self.maxlength(root.right), self.maxlength(root.left))


def minlength(self, root):                # 递归
	if not root: return 0
	if root.left and root.right:
		return 1 + min(self.minlength(root.right), self.minlength(root.left))
	elif root.left:
		return 1 + self.minlength(root.right)
	elif root.right:
		return 1 + self.minlength(root.left)
	else:
		return 1

def function(self, root):                       # 迭代：使用 BFS 策略访问每个结点，同时在每次访问时更新最大深度,同时记录最小深度
	s = []                                      # 迭代：使用 DFS 策略访问每个结点，在每次访问到根节点后更新最大最小深度
	if not root:
		s.append((1, root))

	depth = 0
	while s:
		cur_depth, root = s.pop()
		if root:
			depth = max(depth, cur_depth)
			s.append((cur_depth + 1, root.left))
			s.append((cur_depth + 1, root.right))
		else:
			mindepth = depth
	return mindepth, depth

# 22 生成括号
def function(self, n):        # 在dfs的基础上加上剪枝，避免全是左/右括号的情况
	self.res = []
	self._dfs(0, 0, n, '')
	return self.res

def _dfs(self, l, r, n, lst):
	if l == r == n:
		self.res.append(lst)
		return
	if left < n:
		self._dfs(l+1, r, n, lst+'(')
	if left > right and right < n:        # 剪枝
		self._dfs(l, r+1, n, lst+')')

########################################## 九/剪枝：通常和搜索配合使用 ########################################
# 51 52 N皇后：返回 n 皇后不同的解决方案的数量
def function(self, n):
	if n < 1: return []
	self.n = n
	self.res = []
	self._dfs([], [], [])
	return [['.'*i + 'Q' + '.'*(n-i-1) for i in sol] for sol in res]
def _dfs(queues, xy_dif, xy_sum):
	p = len(queues)
	if p == self.n:
		self.res.append(queues)
		return None
	for q in range(self.n):
		if q not in queues and p-q not in xy_dif and p+q not in xy_sum:
			self._dfs(queues+[q], xy_dif+[p-q], xy_sum+[p+q])
	

# 36 37 数独问题


################################################ 十/二分查找 #############################################
# 69 实现一个求解平方根的函数
def function(self, x):                  # 二分查找
	l, r = 0, x
	while  l < r:
		m = (1+r) / 2
		if m**2 <= x < (m+1)**2:
			return m
		elif m**2 < x:
			l = m + 1
		else:
			r = m - 1
	return l

def function(self, x):           # 牛顿迭代法
	r = x
	while  r * r > x:
		r = (r + x / r) / 2
	return r

####################################### 十一/字典树/前缀树（Prefix Tree） #######################################
# 用于统计和排序大量字符串，比如文本词频统计，最大限度的减少无谓的字符串比较
class Trie(object):

	def __init__(self):
		super(Trie, self).__init__()
		self.root = {}
		self.end_of_word = '#'

	def insert(self, word):               # 插入
		node = self.root
		for char in word:
			node = node.setdefault(char, {})
		node[self.end_of_word] = self.end_of_word

	def search(self, word):               # 搜索
		node = self.root
		for char in word:
			if char not in node:
				return False
			node = node[char]
		return self.end_of_word in node

	def startsWith(self, prefix):         # 判断是否是前缀
		node = self.root
		for char in prefix:
			if char not in node:
				return False
			node = node[char]
		return True

# 79 212 二位网格中的单词搜索问题
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, -1]

end_of_word = '#'

class ClassName(object):

	def findWords(self, board, words):
		if not board or not board[0]: return []
		if not words: return []

		self.res = set()

		root = collections.defaultdict()   #将words全部插入到字典树root
		for word in words:
			node = root
			for char in word:
				node = node.setdefault(char, collections.defaultdict())
			node[end_of_word] = end_of_word

		self.m, self.n = len(board), len(board[0])

		for i in range(self.m):
			for j in range(self.n):
				if board[i][j] in root:     #说明字典树的起始点字符 在board里面
					self._dfs(board, i, j, '', root)

		return list(self.res)

	def _dfs(self, board, i, j, cur_word, cur_dict):

		cur_word += board[i][j]                # 加入当前字符
		cur_dict = cur_dict[board[i][j]]       # 往下继续探一层得到新的字典

		if end_of_word in cur_dict:
			self.res.add(cur_word)

		tmp, board[i][j] = board[i][j], '@'    # 保存当前字符 并进行位置填词
		for k in range(4):
			x, y = i+dx[k], j+dy[k]
			if 0 <= x < self.m and 0 <= y < self.n and board[x][y] != '@' and board[x][y] in cur_dict:
				self._dfs(board, x, y, cur_word, cur_dict)     # m没访问过且在字典里面
		board[i][j] = tmp

################################################ 十二/动态规划 #############################################
# 比较 DP vs 回溯 vs 贪心
# 回溯（递归）--- 重复计算
# 贪心       --- 局部最优
# DP        --- 记录局部最优子结构/多种记录值
# 即递归+记忆化  ——> 递推 --> 状态定义 --> 状态转移方程 --> 最优子结构 ：比如斐波那契数列和计算路径数量，记忆化能大大优化时间复杂度

# 70 爬楼梯
# 120 三角形的最小路径和
# 152 乘积最大子序列
# 121 交易一次    122 交易无数次   123 交易两次   188 买卖k次  309 隔一天才能买  714 有手续费

# 300 最长上升子序列
# 1.暴力2^n 2.DP N^2 3.二分维护数组 N*logN
# 322 零钱兑换 凑成总金额所需的最少的硬币个数
# 72 编辑距离
def function(self, word1, word2):
	m, n = len(word1), len(word2)
	dp = [[0 for _ in range(n+1)] for _ in range(m+1)]

	for i in range(m+1) : dp[i][0] = i                # 初始状态
	for j in range(n+1) : dp[0][j] = j

	for i in range(1, m+1):
		for j in range(1, n+1):
			dp[i][j] = min(dp[i-1][j-1] + (0 if word1[i-1] == word2[j-1] else 1),
				dp[i-1][j] + 1,
				dp[i][j-1] + 1)
	return dp[m][n]

################################################ 十三/并查集 #############################################
# 适用于分布式/mapreduce ，是一种树形结构，union和find
# 优化一： with union by rank
# 优化二： 调用find时进行路径压缩

# 200 岛屿的个数 547 朋友圈
# 方法1 染色floodfill：碰到1则把当前和周围染成0 同时cnt+1
# 方法2 并查集：初始化1节点 ---> 遍历所有节点相邻1合并 ---> 遍历找parents
class Solution(object):           # 染色	
	self.dx = [-1, 1, 0, 0]
	self.dy = [0, 0, -1, 1]

	def numIsland(self, grid):
		super(Solution, self).__init__()
		if not grid or not grid[0] : return 0
		self.max_x, self.max_y, self.grid = len(grid), len(grid[0]), grid
		self.seen = set()
		return sum([self.floodfill_dfs(i, j) for i in range(self.max_x) for j in range(self.max_y)])
	def floodfill_dfs(self, x, y):
		if not self._is_valid(x, y):    
			return 0
		self.seen.add((x, y))
		for k in range(4):
			self.floodfill_dfs(x + dx[k], y + dy[k])
		return 1
	def _is_valid(self, x, y):
		if x < 0 or x >= self.max_x or y < 0 or y >= max_y:
			return False
		if self.grid[x][y] == '0' or ((x, y) in self.seen):
			return False
		return True

class Solution(object):           # 并查集
	def numIsland(self, grid):
		super(Solution, self).__init__()
		if not grid or not grid[0] : return 0

		uf = UnionFind(grid)
		directions = [(0,1), (0,-1), (-1,0), (1,0)]
		m, n = len(grid), len(grid[0])

		for i in range(m):
			for j in range(n):
				if grid[i][j] == '0':
					continue
				for d in directions:
					nr, nc = i + d[0], j + d[1]
					if nc >= 0 and nc >= 0 and nr < m and nc < n and grid[nr][nc] == '1':
						uf.union(i*n+j, nr*n+nc)
		return uf.cnt
class UnionFind(object):
	"""docstring for UnionFind"""
	def __init__(self, gird):
		super(UnionFind, self).__init__()
		m, n = len(grid), len(grid[0])
		self.cnt = 0
		self.parent = [-1] * (m*n)
		self.rank = [0] * (m*n)
		for i in range(m):
			for j in range(n):
				if grid[i][j] == '1':
					self.parent[i*n + j] = i*n + j
					self.cnt += 1
	def find(self, i):
		if self.parent[i] != i:
			self.parent[i] = self.find(self.parent[i])
		return self.parent[i]
	def union(self, x, y):
		rootx = self.find(x)   # 分别找到xy的parent放到root中
		rooty = self.find(y)
		if rootx != rooty:
			if self.rank[rootx] > self.rank[rooty]:   #并查集优化
				self.parent[rooty] = rootx
			elif self.rank[rootx] < self.rank[rooty]:
				self.parent[rootx] = rooty
			else:
				self.parent[rooty] = rootx
				self.rank[rootx] += 1
			self.cnt -= 1


################################################ 十四/LRU Cache #############################################
# 最近最少使用 last recently used  双向链表实现缓存机制
# 还有最近不常用last frequently used /MRU /CAR等等
# 146 LRU缓存机制
class ListNode:
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.hashmap = {}
        # 新建两个节点 head 和 tail
        self.head = ListNode()
        self.tail = ListNode()
        # 初始化链表为 head <-> tail
        self.head.next = self.tail
        self.tail.prev = self.head
    # 因为get与put操作都可能需要将双向链表中的某个节点移到末尾，所以定义一个方法
    def move_node_to_tail(self, key):
            # 先将哈希表key指向的节点拎出来，为了简洁起名node
            #      hashmap[key]                               hashmap[key]
            #           |                                          |
            #           V              -->                         V
            # prev <-> node <-> next         pre <-> next   ...   node
            node = self.hashmap[key]
            node.prev.next = node.next
            node.next.prev = node.prev
            # 之后将node插入到尾节点前
            #                 hashmap[key]                 hashmap[key]
            #                      |                            |
            #                      V        -->                 V
            # prev <-> tail  ...  node                prev <-> node <-> tail
            node.prev = self.tail.prev
            node.next = self.tail
            self.tail.prev.next = node
            self.tail.prev = node
    def get(self, key: int) -> int:
        if key in self.hashmap:
            # 如果已经在链表中了久把它移到末尾（变成最新访问的）
            self.move_node_to_tail(key)
        res = self.hashmap.get(key, -1)
        if res == -1:
            return res
        else:
            return res.value
    def put(self, key: int, value: int) -> None:
        if key in self.hashmap:
            # 如果key本身已经在哈希表中了就不需要在链表中加入新的节点
            # 但是需要更新字典该值对应节点的value
            self.hashmap[key].value = value
            # 之后将该节点移到末尾
            self.move_node_to_tail(key)
        else:
            if len(self.hashmap) == self.capacity:
                # 去掉哈希表对应项
                self.hashmap.pop(self.head.next.key)
                # 去掉最久没有被访问过的节点，即头节点之后的节点
                self.head.next = self.head.next.next
                self.head.next.prev = self.head
            # 如果不在的话就插入到尾节点前
            new = ListNode(key, value)
            self.hashmap[key] = new
            new.prev = self.tail.prev
            new.next = self.tail
            self.tail.prev.next = new
            self.tail.prev = new

########################################## 十五/布隆过滤器bloom filter ##########################################
# 一个很长的二进制向量和一个映射函数 可用于检索一个元素是否在一个集合中
# 优点是时间和空间效率远远超过一般算法， 缺点是删除困难和存在一定的误识别率（判断一个元素在一个集合中的时候发生，判断不在不会有误识别率）
# 应用于比特币（redis vs bloomfilter） 分布式系统（mapreduce）
















