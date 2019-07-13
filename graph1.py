# 明星问题：任何人都认识明星，但明星不认识任何人
# 应用场景：1.处理 Linux 中软件包依赖问题  2.多线程的死锁问题


###### 最简单的方法，两层循环 ###########################################################################
def star1(G):                       # 寻找二维数组G中的明星
	n = len(G)
	for u in range(n):
		for v in range(n):          # 遍历数组中的每个元素
			if u == v:
				continue            # 相同的人跳过
			if G[u][v]:
				break               # 明星认识路人则结束
			if not G[v][u]:
				break               # 路人不认识明星则结束
		else:
			return u                # u是明星
	return None


###### 由于非明星认识则无需继续进行,所以排除掉非明星即可.下面将其变为一个线性级的算法 ###########################
def star2(G):
	n = len(G) 
	u, v = 0, 1                     # 假设u是明星
	for c in range(2, n+1):
		if G[u][v]:
			u = c                   # u认识v,说明u不是明星,看c
		else:
			v = c                   # u是明星, c赋值给v,遍历G[u][v]的下一个元素
	if u == n:
		c = v                       # 最后一次遍历，u=v则说明u不是明星，则v是，用中间变量c保存明星
	else:
		c = u                       # 用中间变量c保存明星u	

	for v in range(n):              # 验证c
		if c == v:                  # 相同的人跳过跳过
			continue
		if G[c][v]:                 # 明星认识路人则结束
			break
		if not G[v][c]:             # 路人不认识明星则结束
			break
	else:
		return c
	return None


if __name__ == '__main__':
	from random import randrange
	n = 100
	G = [[randrange(2) for i in range(n)] for _ in range(n)]
	x = randrange(n)            # 设置一个明星 x
	for i in range(n):
		G[i][x] = True          # 所有人都认识明星 x
		G[x][i] = False         # 明星 x 不认识任何人 
	print(star1(G))
	print(star2(G))






