"""
推导式介绍:
    概述:
        推导式也叫解析式, 属于Python的一种特有写法, 目的是: 简化我们代码编写的.
    分类:
        列表推导式
        集合推导式
        字典推导式
    格式:
        变量名 = [变量名 for ... in ... if 判断条件]
        变量名 = {变量名 for ... in ... if 判断条件}
        变量名 = {变量名1:变量名2 for ... in ... if 判断条件}
"""
# 需求1: 创建1个 0 ~ 9的列表.
# 方式1: 不使用推导式.
list1 = []
for i in range(10):
    list1.append(i)
print(list1)

# 方式2: 列表推导式.
list2 = [i for i in range(10)]      # 效果同上.
print(list2)

# 方式3: 类型转换.
list3 = list(range(10))
print(list3)
print('-' * 28)


# 需求2: 创建1个 0 ~ 9的 偶数 列表.
# 方式1: 不使用推导式.
list1 = []
for i in range(10):
    if i % 2 == 0:
        list1.append(i)
print(list1)

# 方式2: 列表推导式.
list2 = [i for i in range(10) if i % 2 == 0]      # 效果同上.
print(list2)

# 方式3: 类型转换.
list3 = list(range(0, 10, 2))
print(list3)
print('-' * 28)


# 需求3: 创建列表 => [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
# 提示: 循环嵌套, i的值 1, 2    j的值 0, 1, 2
# 方式1: 普通版, 循环嵌套.
list1 = []
for i in range(1, 3):
    for j in range(3):
        # print((i, j))
        # 把 i 和 j 封装成元组.
        # tuple1 = (i, j)
        # list1.append(tuple1)
        list1.append((i, j))
print(list1)

# 方式2: 列表推导式.
list2 = [(i, j) for i in range(1, 3) for j in range(3)]     # 效果同上.
print(list2)
print('-' * 28)
