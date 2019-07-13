# 场景一： BFS & DFS ###########################################################################
graph = {
	'A': ['B', 'C'],
	'B': ['A', 'C', 'D'],
	'C': ['A', 'B', 'D', 'E'],
	'D': ['B', 'C', 'E', 'F'],
	'E': ['C', 'D'],
	'F': ['D']
}
############## BFS ######################
def BFS(graph, s):
	'''
	使用队列'''
	queue = []                       #保存待遍历节点
	queue.append(s)
	seen = set()
	seen.add(s)                      #记录queue中的节点
	while (len(queue) > 0):
		vertex = queue.pop(0)        #取待遍历节点的第一个，保存为最终结果
		nodes = graph[vertex]
		for w in nodes:
			if w not in seen:
				queue.append(w)
				seen.add(w)
		print(vertex)
print(BFS(graph, 'A'))               # ABCDEF

############### DFS ######################
def DFS(graph, s):
	'''
	使用栈'''
	stack = []                       
	stack.append(s)
	seen = set()
	seen.add(s)                      
	while (len(stack) > 0):
		vertex = stack.pop()         # 只用把里面的0去掉
		nodes = graph[vertex]
		for w in nodes:
			if w not in seen:
				stack.append(w)
				seen.add(w)
		print(vertex)
print(BFS(graph, 'E'))               # EDFBAC

# 场景二： 并查集 & dijkstra ########################################################################
################ 并查集(bfs) ##################
def BFS(graph, s):
	'''
	返回父节点数组'''
	queue = []                       
	queue.append(s)
	seen = set()
	seen.add(s)    
	parent = {s: None}               # 定义父节点数组

	while (len(queue) > 0):
		vertex = queue.pop(0)       
		nodes = graph[vertex]
		for w in nodes:
			if w not in seen:
				queue.append(w)
				seen.add(w)
				parent[w] = vertex   # 添加关系
		print(vertex)
	return parent
print(BFS(graph, 'A'))               # ABCDEF

################ 最短路径(dijkstra) ##################
import heapq
import math

graph = {
	'A': {'B':5, 'C':1},
	'B': {'A':5, 'C':2, 'D':1},
	'C': {'A':1, 'B':2, 'D':4, 'E':8},
	'D': {'B':1, 'C':4, 'E':3, 'F':6},
	'E': {'C':8, 'D':3},
	'F': {'D':6}
}

def init_distance(graph, s):          # 初始化路径长度
	distance = {s:0}
	for vertex in graph:
		if vertex != s:
			distance[vertex] = math.inf
	return distance
def dijkstra(graph, s):
	'''
	把BFS改成dijkstra算法，只需要把队列改成优先队列即可'''
	pqueue = []                       # 优先队列
	heapq.heappush(pqueue, (0, s))
	seen = set()
	parent = {s: None}
	distance = init_distance(graph, s)

	while (len(pqueue) > 0):
		pair = heapq.heappop(pqueue)
		dist, vertex = pair[0], pair[1]
		seen.add(vertex)

		nodes = graph[vertex].keys()  # graph字典的value的key 即A:BC
		for w in nodes:
			if w not in seen:
				if dist + graph[vertex][w] < distance[w]:
					heapq.heappush(pqueue, (dist + graph[vertex][w], w))
					parent[w] = vertex
					distance[w] = dist + graph[vertex][w]
		return parent, distance
parent, distance = dijkstra(graph, 'A')
print(parent)
print(distance)
# {'A': None, 'B': 'A', 'C': 'A', 'D': 'B', 'E': 'C', 'F': 'D'}
# {'A': None, 'B': 'A', 'C': 'A'}
# {'A': 0, 'B': 5, 'C': 1, 'D': inf, 'E': inf, 'F': inf}














