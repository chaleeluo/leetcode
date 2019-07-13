# 场景一： 输入输出 ###########################################################################
a = input()
b = input()
c = input().split()    #任意个赋给一个
d,e = input().split()  #多个赋给多个 
h = int(input())       #一个赋多个 
j = [int(i) for i in input().split()]  #适用于输入数组
k = [int(i) for i in input().split()]  
print(a,'a')
print(b,'b',len(b))
print(c,'c',len(c))
print(d,'d')
print(e,'e')
print(h,'h')
print(j,'j')
print(k,'k')
# ######### 输入 #####
# 4
# 2 1
# 2 1 #任意个赋给一个
# 2 1 #多个赋给多个
# 4   #一个赋多个
# 4
# 2 1
# ######### 输出 #####
# 4 a
# 2 1 b 3        # b的长度是3，因为空格也会计入，所以这种情况适用于输入字符串并遍历
# ['2', '1'] c 2 #无所谓几个 拆分后长度便为2了
# 2 d            #必须多个
# 1 e            #必须多个
# 4 h            #必须一个
# [4] j
# [2, 1] k       #适用于输入数组



# 场景二： 分隔符 #############################################################################
# split()的默认分隔符就是所有空字符(空格，换行，制表符)，所以split(',')就会默认把多余的空格都去掉
#示例  把{31, 18, 19, 1, 25}转化为[31, 18, 19, 1, 25]
import sys
l = sys.stdin.readline().strip()
s= [int(i) for i in l.strip('{').strip('}').split(',')]  



# 场景三： 矩阵&元组 ###########################################################################
nums = int(input())
tup = []
mat = []
for i in range(nums):
    #read_list = list(map(int, input().split()))
    read_list = [int(i) for i in input().split()]
    tup.append((read_list[0], read_list[1]))
    mat.append(read_list)
numpymat = np.mat(mat)  #可以转化为numpy形式的矩阵

print('read_list',read_list)
print(tup)
print(mat)
for i in tup:
	print(i)
	print(i[0], i[1])
# ##### 输入 ######
# 2
# 1 1
# 2 2
# ##### 输出 ######
# read_list [2, 2]
# [(1, 1), (2, 2)]
# [[1, 1], [2, 2]]
# (1, 1)
# 1 1
# (2, 2)
# 2 2


# 场景四： sys.stdin读入 ###########################################################################
# while & for ：第一个程序是为了获得每一行内容，第二个程序是为了迭代每行内容的每个元素
import sys
try:
    while True:
        line = sys.stdin.readline().strip()
        if not line:
            break
        print(line)
except:
    pass

try:
    for l in sys.stdin.readline().strip():
        if not line:
            break
except:
    pass




