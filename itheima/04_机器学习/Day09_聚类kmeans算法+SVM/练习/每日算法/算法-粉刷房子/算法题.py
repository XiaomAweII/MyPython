# 定义min_num函数，用于计算房子最小的总价格
def min_num(list1):
    # 获取列表的长度，即房子的个数
    n = len(list1)
    # 条件判断，如果房子个数为0，则最小成本为0
    if n == 0:
        return 0
    # 创建一个二维数组dp，大小为n*3，用于存储每个房子选择不同颜色时的最小成本
    dp = [[0, 0, 0] for i in range(n)]
    # 循环遍历颜色，初始化第一个房子的成本
    for j in range(3):
        dp[0][j] = list1[0][j]
    # 从第二个房子开始遍历
    for i in range(1, n):
        # 遍历当前房子的三种颜色
        for j in range(3):
            min_cost = float('inf')  # 初始化最小成本为无穷大
            # 遍历前一个房子的三种颜色
            for k in range(3):
                # 如果前一个房子的颜色与当前房子的颜色不同
                if k != j:
                    # 更新最小成本
                    min_cost = min(min_cost, dp[i - 1][k])
            # 更新当前房子选择颜色j时的最小成本
            dp[i][j] = min_cost + list1[i][j]
    # 返回最后一个房子选择三种颜色中的最小成本
    return min(dp[n - 1])

# 在main方法中进行测试
if __name__ == '__main__':
    costs = [[1,2,3],[2,3,4],[3,4,5],[4,5,6]]
    # 调用min_num函数，并打印结果
    print(min_num(costs))  # 输出最小成本