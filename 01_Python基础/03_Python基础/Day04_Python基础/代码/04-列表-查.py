"""
列表 查询 相关函数:
    列表名.index(要查找的元素, 起始索引, 结束索引)       类似于字符串的 index()
    列表名.count(元素值)                             统计元素在列表中出现的总次数.
    元素值 in 列表名                                 判断元素在不在列表中, 在: True, 不在: False
    元素值 not in 列表名                             判断元素在不在列表中, 在: False, 不在: True
"""

# 1. 定义列表
list1 = ['a', 'b', 'c', 'a', 'c']


# 2. 演示列表的 查询 相关的函数.
# 列表名.index(要查找的元素, 起始索引, 结束索引)       类似于字符串的 index()
print(list1.index('a'))                             # 0,  不指定查找范围, 就从列表的开头开始查找, 直至列表末尾.
print(list1.index('a', 1))            # 3,  只指定开头, 不指定结尾, 就从列表的指定位置 找到 列表末尾.
# print(list1.index('a', 1, 3))  # 报错, 指定了开头和结尾, 就在指定区间查找, 包左不包右. 找不到就: 报错. 这点和字符串是一样的.

# 列表名.count(元素值)                             统计元素在列表中出现的总次数.
print(list1.count('a'))         # 2

# 元素值 in 列表名                                 判断元素在不在列表中, 在: True, 不在: False
print('a' in list1)     # True

# 元素值 not in 列表名                             判断元素在不在列表中, 在: False, 不在: True
print('a' not in list1) # False


