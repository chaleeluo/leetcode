# 问题一：红绿灯问题 （三个线程按顺序输出打印123123123） #######################################################
'''
在这里我们可以将红绿灯的各种颜色的切换看作是上锁与释放锁。
在初始时刻对于红黄绿的三种锁中，只有红灯的锁是释放的，而黄灯和绿灯的锁是被锁着的，
线程进入显示为红灯后，应将其锁给锁住，而不让其他线程进入（红灯亮着的时候，三个锁都是锁住的）；
当红灯显示时间结束后，下一个为黄灯，因此需要将黄灯的锁给释放掉，从而让黄灯的线程进入，进入后再将锁锁住；
当黄灯显示完毕后，将绿灯的锁给打开，从而让绿灯显示，以此类推。
'''
# 关键在于如何控制上锁lock.acquire()与锁的释放lock.release()

import time

red_lock = threading.Lock()      # 红灯锁
yellow_lock = threading.Lock()   # 黄灯锁
green_lock = threading.Lock()    # 绿灯锁

count = 18   # 为避免一直循环，我们在这里假设每个数字输出6次，3×6=18

def red():
	global count
	while count >= 0:
		red_lock.acquire()      # 将红灯的锁给锁住
		print(1, end = '-')     # 将红灯表示为1
		# print('id:', threading.get_ident())  # 查看线程id
		yellow_lock.release()   # 下一个为黄灯亮，将黄灯的锁给释放
		count -= 1

def yellow():
	global count
	while count >= 0:
		yellow_lock.acquire()   
		print(2, end = '-')     
		# print('id:', threading.get_ident())
		green_lock.release()    
		count -= 1
	
def green():
	global count
	while count >= 0:
		green_lock.acquire()   
		print(3, end = '-')    
		# print('id:', threading.get_ident())
		red_lock.release()     
		count -= 1

if __name__ == '__main__':
	thread_list = []
	func_list = [red, yellow, green]

	for func in func_list:         # 创建三个线程
		th = threading.Thread(target = func)
		thread_list.append(th)

	yellow_lock.acquire()          # 红灯先亮，因此将黄灯和绿灯的锁给锁住，以阻塞线程2和3的执行
	green_lock.acquire()   

	for th in thread_list:         # 启动
		# print(time.time())       
		th.start()

	for th in thread_list:         # 返回结果
		th.join()
