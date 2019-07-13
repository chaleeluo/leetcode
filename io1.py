# 场景一： 文件夹读写 ###########################################################################
# 读取一个文件夹的多个文件
path = '/文件夹'
files = os.listdir(path)
# 循环读取文件中‘code','close'两列，并添加一列'rank'
data_list = []
for file in  files:
    tmp = pd.read_csv(path + file)[['code', 'close']]
    tmp['rank'] = num_filter.findall(file)[0]
    tmp.to_csv('/新文件夹'+'tmp'+file)
    data_list.append(tmp)



# 场景二： 文件读写 ###########################################################################
# 读取TXT文件并逐行写入另外一个TXT文件
f = open("文件路径和全名")  
for line in f.readlines():  
    tmp = line.split('\t')[0:2]
    with open("要写入的文件路径和文件名","a") as myf:
        myf.write(tmp['写入列1名', '写入列2名'])  



# 场景三： 大文件处理 ###########################################################################
# 1. 一般读取：
f = open(filename,'r')
f.read()
# 在文件较大时引发 MemoryError（内存溢出），跟read()类似的还有：read(参数)、readline()、readlines()
# read(参数)：通过参数指定每次读取的大小长度,这样就避免了因为文件太大读取出问题。
# readline()：每次读取一行
# readlines()：读取全部的行，构成一个list，通过list来对文件进行处理，但是这种方式依然会造成MemoyError

# 2. 如何解决：自动管理是最完美的方式
with open('filename', 'r', encoding = 'utf-8') as f: 
	'''
	with 负责打开和关闭文件（包括在内部块中引发异常时）
	for line in f 将文件对象f视为一个可迭代的数据类型，自动使用 IO 缓存和内存管理'''
	for line in f: 
		do_something(line)

# 3. 以下可参考：
# (1)使用 fileinput 模块
import fileinput 
for line in fileinput.input(['filename']): 
	'''
	会按照顺序读取行，但是在读取之后不会将它们保留在内存中'''
	do_something(line)        
# (2)逐行读取 
while() 循环+readline() 来逐行读取：
with open('filename', 'r', encoding = 'utf-8') as f:
    while True:
        line = f.readline()  
        if not line:  
            break
        do_something(line)
# (3)分块读取
def chunks(file_obj, chunk_size = 1024*1024):
    """
    默认块大小：1M 
    使用 iter 和 yield指定每次读取的长度"""
    while True:
        data = file_obj.read(chunk_size)  
        if not data:
            break
        yield data

with open('filename', 'r', encoding = 'utf-8') as f:
    for chuck chunks(f):
        do_something(chunk)






