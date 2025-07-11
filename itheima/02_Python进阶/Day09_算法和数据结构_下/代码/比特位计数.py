# 给定一个非负整数 n ，请计算 0 到 n 之间的每个数字的二进制表示中 1 的个数，并输出一个数组
# 定义一个函数
# def countBits(n):
#     '''
#     计算0-n之间每个数字的二进制中1的个数
#     :param n: 给定的一个非负整数 n
#     :return:返回result
#     '''
#     # 定义一个空列表
#     result = []
#     # 使用for循环获取0-n之间的整数
#     for i in range(n + 1):
#         # 初始变量count中1的数量为0
#         count = 0
#         num = i
#         while num > 0:
#             # 使用按位与&来判断二进制表示中 1 的个数
#             count += num & 1
#             # 位运算，将num右移以为，相当于num//2
#             # >>= 是右移赋值运算符,表示将 num 向右移动一位，并将结果赋值给 num
#             num >>= 1
#         result.append(count)
#     return result
#
# # 测试代码
# n = 5
# print(countBits(n))


# 定义一个函数
def countBits(n):
    result = []
    for i in range(n + 1):
        #计算 i 的二进制表示中 1 的个数
        count = bin(i).count('1') #将i转化为二进制再使用。count方法计算
        result.append(count)
    return result

# 测试示例
n = 5
print(countBits(n))


