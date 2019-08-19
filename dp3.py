# 股票买卖系列
# 121 交易一次    122 交易无数次   123 交易两次   188 买卖k次  309 隔一天才能买  714 有手续费
# 121
def maxProfit(self, prices):
    if len(prices) < 2: return 0
    maxp, minnum = 0, prices[0]
    for p in prices:
            minnum = min(minnum, p)         # 定义到当前时间段的最小值，即2️⃣状态更新：用当前值-到当前值的最小值
            maxp = max(maxp, p - minnum)    # 1️⃣定义状态为到当前时间段的最大收益
    return maxp

# 122
def maxProfit(self, prices):
	if len(prices) < 2: return 0
    i = 0
    res = 0
    while i < len(prices)-1:
        if prices[i+1] > prices[i]:
            res += prices[i+1]-prices[i]
        i += 1
        return res

# 123 可以拆分成找到一个位置j 使得左右两边各交易一次（各交易一次代码可参考121）后和最小
def maxProfit(self, prices):        # 方法1:复杂度n的平方
	if len(prices) < 2: return 0
	def maxp(a, b):
		if b == a: return 0
		maxp, minnum = 0, prices[a]
		for i in range(a, b):
			minnum = min(minnum, prices[i])
			maxp = max(maxp, prices[i]-minnum)
		return maxp

	nums = [maxp(0,j) + maxp(j, len(prices)) for j in range(len(prices))] 
	return nums


def maxProfit(self, prices):        # 方法2:复杂度n+n  左右已经用数组存起来了 
	if len(prices) < 2: return 0

	l, r = [0]*len(prices), [0]*len(prices)
	minnum, maxnum = prices[0], prices[-1]    
    for i in range(1, len(prices)):
        left[i] = max(left[i-1], prices[i]-minnum)
        minnum = min(minnum, prices[i])
    for j in range(len(prices)-2, -1, -1):
        right[j] = max(right[j+1], maxnum-prices[j])
        maxnum = max(maxnum, prices[j])

    res = [left[k]+right[k] for k in range(len(prices))]
    return max(res)


def maxProfit(self, prices):        # 方法3:巧方法
    import sys
    buya, sella, buyb, sellb = -sys.maxsize, 0, -sys.maxsize, 0
    for price in prices:
        buya = max(buya, -price)   #第一次买入手上的钱
        sella = max(sella, price+buya)  #第一次卖出后手上的钱
        buyb = max(buyb, sella-price)   #第二次买入手上的钱，已经记录了最大的buyb作为buyb
        sellb = max(sellb, price+buyb)  #第二次卖出后手上的钱
        print(price,buya,sella,buyb,sellb)
    return sellb

# 188 买卖k次
def greedy(prices):
    res = 0
    for i in range(1, len(prices)):
        res += max(0, prices[i]-prices[i-1])
    return res
def maxProfit(k, prices):             #方法1：同时更新局部和全局最大收益
	if k <= 0 or len(prices) < 2:
		return 0
	if k > len(prices):    # 注意当k大于天数时，直接用贪心算法。
		return greedy(prices)
    '''
    local[i][j]表示第i天最多进行了j笔交易，且第j笔是在第i天完成的的最大收益； 
    global[i][j]表示第i天进行了j笔交易的最大收益，是目前为止全局最优，不规定第j笔在哪天完成。
    则递推公式为：
    local[i][j] = max(global[i-1][j-1]+max(0, prices[i]-prices[i-1]), local[i-1][j]+prices[i]-prices[i-1])；
    global[i][j] = max(local[i][j], global[i-1][j])；
    双重循环，计算最后取global[-1][-1]；
    '''
	g = l = [[0 for _ in range(k+1)] for _ in range(len(prices))]
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i-1]
        for j in range(1, k+1):
            l[i][j] = max( g[i-1][j-1]+max(0,diff), l[i-1][j]+diff )
'''
第一个是全局到i-1天进行j-1次交易，如果今天是赚钱的话,加上今天的交易，也就是前面只要j-1次交易，最后一次交易取当前天），
第二个量则是取local第i-1天j次交易，然后加上今天的差值
（这里因为local[i-1][j]比如包含第i-1天卖出的交易，所以现在变成第i天卖出，并不会增加交易次数，
而且这里无论diff是不是大于0都一定要加上，因为否则就不满足local[i][j]必须在最后一天卖出的条件了）
'''
            g[i][j] = max( l[i][j], g[i-1][j] )
        return g[-1][-1]

import sys
def maxProfit(k, prices):    #方法2：通式
    if k <= 0 or len(prices) < 2:
        return 0
    if k > len(prices):    
        return greedy(prices)

    mp = [[[0, 0] for i in range(k+1)] for i in range(len(prices))]  
    mp[0][0][0], mp[0][0][1] = 0, -prices[0]

    for i in range(1,k+1):
        mp[0][i][0], mp[0][i][1] = -sys.maxsize, -sys.maxsize
    for i in range(1,len(prices)):
        mp[i][0][0] = mp[i-1][0][0]    #当前天数没有股票
        mp[i][0][1] = max(mp[i-1][0][1], mp[i-1][0][0]-prices[i])  #当前天数持有一股
        for j in range(1,k):
            mp[i][j][0] = max(mp[i - 1][j][0], mp[i - 1][j-1][1] + prices[i])    #当前交易次数没有股票
            mp[i][j][1] = max(mp[i - 1][j][1], mp[i - 1][j][0] - prices[i])      #当前交易次数持有一股
        mp[i][k][0] = max(mp[i - 1][k][0], mp[i - 1][k-1][1] + prices[i])
    return max(mp[len(prices) - 1][i][0] for i in range(k+1))


# 309 隔一天才能买















