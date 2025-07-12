#1.定义函数，接受一个二维数组作为参数，代表初始的生命状态
def gameOfLife(list1):
    #2.获取输入的二维数组的行数和列数
    rows = len(list1)         #行数
    columns = len(list1[0])   #列数
    #3.创建一个与输入的二维数组相同大小的新二维数组，用于存储下一代的生命状态，初始时所有元素为0
    dp = [[0 for i in range(columns)] for j in range(rows)]
    #4.定义方向偏移量
    directions = [(-1,-1),(-1,0),(0,-1),(1,-1),(-1,1),(0,1),(1,0),(1,1)]
    #5.遍历数组中的每个细胞
    for i in range(rows):
        for j in range(columns):
            num = 0    # 初始化当前细胞周围存活的细胞数量(计数器)
            #6.遍历当前细胞周围的8个方向(即相邻细胞的方向)
            for dx, dy in directions:
                nx, ny = i+dx, j+dy     # 计算相邻细胞的坐标
                #7.检查相邻细胞是否在二维数组的坐标范围内且存活
                if 0 <= nx < rows and 0 <= ny < columns and list1[nx][ny] == 1:
                    num += 1       # 如果相邻细胞存活，则增加计数器
            #8.根据当前细胞的状态和周围存活的细胞数量，更新下一代的状态
            if list1[i][j] == 1:    # 8.1 如果当前细胞存活
                if num < 2 or num > 3:    # 8.1.1如果周围存活的细胞数少于2个或多于3个，则下一代该细胞死亡
                    dp[i][j] = 0
                else:                     # 8.1.2否则，该细胞在下一代继续存活
                    dp[i][j] = 1
            else:                   # 8.2如果当前细胞死亡
                if num == 3:              # 8.2.1如果周围有恰好3个存活的细胞，则该细胞在下一代复活
                    dp[i][j] = 1
    #9.返回更新后的数组，即下一代的生命状态
    return dp
#10.main方法测试
if __name__ == '__main__':
    # list1 = [[1, 1], [1, 0]]
    # print(gameOfLife(list1))
    list1 = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
    print(gameOfLife(list1))

