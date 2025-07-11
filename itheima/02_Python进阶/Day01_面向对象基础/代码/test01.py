def longest(text1, text2):
    len1, len2 = len(text1) + 1, len(text2) + 1     # 这里 +1, range(len1) 才能取到 len个元素
    dp = [[0 for _ in range(len1)] for _ in range(len2)]    # 对 dp 数组做初始化操作 - 全 0 二维数组
    #
    for i in range(1, len2):
    # for i in range(1, len1):
        for j in range(1, len1):        # 开始列出状态转移方程
            # 因为 i 从 1开始, 所以第一轮内循环 i-1 是 text2 第一个字母
            if text1[j - 1] == text2[i - 1]:        # 这一行判断 text1 中是否有和 text2[0] 相同的字母
                dp[i][j] = dp[i - 1][j - 1] + 1     # 如果有相同的, 就把 dp[i][j] 的数值 +1.
                # dp[i - 1][j - 1] 在 i=1, j=1 的时候, 值为初始化得到的 0.
            else:
                # 如果判断不相同, 把前面判断的值拿到后边
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[-1][-1]

if __name__ == '__main__':
    length = longest('abcde', 'def')
    print(length)