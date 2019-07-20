# 问题一： 20. 有效的括号 ###########################################################################
'''给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
输入: "([)]"
输出: false
输入: "{[]}"
输出: true
'''
def isValid(self, s):
	d = {'(':')', '[':']', '{':'}'}
	a = []
	l = list(s)
	for i in l:
		if a == []:
			a.append(i)
		elif d.get(a[-1]) == None:
			return False
		elif i == d[a[-1]]:
			a.pop(-1)
		else:
			a.append(i)
		return a == 


# 问题二： 22. 括号生成 ###########################################################################
'''给出 n 代表生成括号的对数，请你写出一个函数，使其能够生成所有可能的并且有效的括号组合。
例如，给出 n = 3，生成结果为：
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
'''
class Solution(object):
    def generateParenthesis(self, N):
        ans = []
        def backtrack(S = '', left = 0, right = 0):         # 分治/递归/回溯  都开一个辅助函数
            if len(S) == 2 * N:
                ans.append(S)
                return
            if left < N:
                backtrack(S+'(', left+1, right)
            if right < left:                                # 剪枝
                backtrack(S+')', left, right+1)

        backtrack()
        return ans


# 问题三： 1021. 删除最外层的括号 ###########################################################################
'''
输入："(()())(())"
输出："()()()"
解释：
输入字符串为 "(()())(())"，原语化分解得到 "(()())" + "(())"，
删除每个部分中的最外层括号后得到 "()()" + "()" = "()()()"。
输入："()()"
输出：""
解释：
输入字符串为 "()()"，原语化分解得到 "()" + "()"，
删除每个部分中的最外层括号后得到 "" + "" = ""。
'''
# 计数法(找规律)：设置一个计数器 count，左括号 +1，右括号减 1，等于 0 则找到外括号的终点。
# 				并且 0 后面的一个括号肯定也是外括号，可以直接跳过。
def removeOuterParentheses(s):
	target = ''
	cnt, i = 1, 1
	while i < len(s):
		cnt += 1 if s[i] == '(' else -1
		if cnt == 0:
			i += 2
			cnt += 1
			continue
		target += s[i]
		i += 1
	return target
# 双指针法：一个指针p记录最外层左括号的位置，一个指针q记录最外层右括号的位置，当匹配到的时候，再把字符串切片相加。
def removeOuterParentheses(s):
	target = ''
	cnt, p, q = 0, 0, 0
	while q < len(s):
		cnt += 1 if s[q] == '(' else -1
		if cnt == 0:
			s = s[0: p] + s[p+1: q] + s[q+1:]
			q -= 1
			p = q              # 后指针赋值给前指针
			continue           # 记得要跳出啊啊啊啊啊啊啊啊啊！
		q += 1
	return s





















