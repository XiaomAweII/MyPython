"""
set集合介绍:
    概述:
        它属于容器类型的一种, 其元素特点为: 无序, 唯一.
    无序解释:
        这里的无序, 并不是排序的意思, 而是: 元素的存, 取顺序不一致, 例如: 存的时候顺序是1, 2, 3, 取的时候顺序是2, 1, 3
    应用场景:
        去重.
    set集合定义格式:
        set1 = {值1, 值2, 值3...}
        set2 = set()

"""

# 1. 定义集合.
set1 = {10, 2, 'c', 5, 'a', 6, 3, 'b', 10, 5, 'a'}
set2 = set()
set3 = {}           # 这个不是在定义集合, 而是在定义: 字典.

# 2. 打印变量的数据类型
print(type(set1))   # <class 'set'>
print(type(set2))   # <class 'set'>
print(type(set3))   # <class 'dict'>

# 3. 打印元素内容.
print(f'set1: {set1}')  # set1: {2, 3, 5, 6, 'a', 10, 'c', 'b'}
print(f'set2: {set2}')  # set2: set()
print(f'set3: {set3}')  # set3: {}
print('-' * 28)


# 需求: 对列表元素值进行去重.
list1 = [10, 20, 30, 20, 10, 30, 50]

# 去重
# 思路1: 定义新列表, 遍历原始列表获取每个元素, 然后判断是否在新列表中, 不存在就添加.
# 代码略, 自己写一下.

# 思路2: list -> set, 会自动去重 -> list
set_tmp = set(list1)
list1 = list(set_tmp)

# 打印去重后的结果.
print(f'list1: {list1}')