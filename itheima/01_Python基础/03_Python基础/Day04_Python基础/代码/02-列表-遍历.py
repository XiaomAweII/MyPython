"""
遍历介绍:
    概述:
        逐个的获取容器类型中的每个元素的过程, 就称之为: 遍历.
    遍历思路:
        思路1: for循环, 直接从 容器类型中获取每个元素.
        思路2: 采用 索引的 方式实现.
"""

# 1. 定义列表.
list1 = [11, 22, 33, 44, 55]


# 2. 遍历列表.
# 思路1: 采用 for循环, 直接获取列表的每个元素.
for value in list1:
    print(value)
print('-' * 28)


# 思路2: 采用 索引的 方式实现, while循环写法.
# print(list1[0])
# print(list1[1])
# print(list1[2])
# print(list1[3])
# print(list1[4])
i = 0
while i < len(list1):   # 细节: 不要直接写数值, 建议写成 列表的长度.
    print(list1[i])     # i 就是列表中每个元素的索引, 例如: 0, 1, 2, 3, 4
    i += 1
print('-' * 28)

# 思路2: 采用 索引的 方式实现, for循环写法.
for i in range(len(list1)):
    print(list1[i])     # i 就是列表中每个元素的索引, 例如: 0, 1, 2, 3, 4

