# 场景一： lambda 匿名函数 ######################################################
def quadratic(a, b, c):
	return lambda x: a*x*x + b*x + c

print(quadratic(1, -1, 2)(5))

# 场景二： decorator 装饰器 #####################################################
import time
def is_prime(num):
	'''判断是否是质数'''
	if num < 2:
		return False
	elif num == 2:
		return True
	else:
		for i in range(2,num):
			if num % i == 0:
				return False
		return True

def display_time(func):       # 传入函数
	def wrapper(*args):       # 函数传入参数
		t1 = time.time()
		res = func()          # 传入结果
		t2 = time.time()
		print('Total time: {:.4} s'.format(t2 - t1))   # 这里可以传入修饰函数部分
		return res
	return wrapper

@display_time           #语法糖！！
def count_prime_nums(maxnum):
	'''对质数计数'''
	cnt = 0
	for i in range(2, maxnum):
		if is_prime(i):
			cnt += 1
	return cnt
count = count_prime_nums(1000)
print(count)


# 场景三： map(函数f，list)，把函数f 依次作用在list 的每个元素上，返回一个新的list ###################################
#         zip为打包为元祖，*zip为解压元祖。                  ####################################################
#         filter(函数f，list)，把函数f 依次作用在list 的每个元素上，返回函数f 判断为true的一个新list #################

a, b, c = 'flower', 'flow', 'flight'     # 三个对象
fs = ['flower', 'flow', 'flight']        # 一个对象
mat = []
mat[:] = map(list, zip(*fs))
'''
list(zip(a, b, c)) =  [('f', 'f', 'f'), ('l', 'l', 'l'), ('o', 'o', 'i'), ('w', 'w', 'g')]     
list(zip(*fs) =  [('f', 'f', 'f'), ('l', 'l', 'l'), ('o', 'o', 'i'), ('w', 'w', 'g')]
mat = [['f', 'f', 'f'], ['l', 'l', 'l'], ['o', 'o', 'i'], ['w', 'w', 'g']]
'''
# 48. 旋转图像 : 给定一个 n × n 的二维矩阵表示一个图像。将图像顺时针旋转 90 度。原地！！！
'''
输入:
matrix = [[1,2,3],
          [4,5,6],
          [7,8,9]]
输出: [[7,4,1],
      [8,5,2],
      [9,6,3]]
'''
def rotate(matrix):
	matrix[:] = map(list, zip(*matrix[::-1]))

# 125. 验证回文串 : 给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。
'''
输入:"A man, a plan, a canal: Panama"
输出: true
'''
def ispalindrome(s):
	s = list(filter(str.isalnum, s.lower()))
	return s == s[::-1]
