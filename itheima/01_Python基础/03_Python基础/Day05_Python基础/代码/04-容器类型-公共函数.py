"""
容器类型的公共操作-函数:
    len()                       获取长度
    del 或者 del()               删除
    max()                       获取最大值
    min()                       获取最小值
    range(start, end, step)     生成指定范围内的数据
    enumerate()                 基于可迭代类型(字符串, 列表, 元组等), 生成 下标 + 元素的方式, 即: ['a', 'b', 'c']  => [(0, 'a'), (1, 'b'), (2, 'c')]
"""

# 此处以列表作为演示, 其它雷同.
list1 = [10, 50, 20, 30, 66, 22]

# 演示: len()                       获取长度
print(len(list1))

# 演示: del 或者 del()               删除
del list1[1]
del(list1[1])      # 效果同上.
print(f'删除后的list1: {list1}')

# 演示: max()                       获取最大值
print(f'最大值: {max(list1)}')

# 演示: min()                       获取最小值
print(f'最大值: {min(list1)}')

# 演示: range(start, end, step)     生成指定范围内的数据
print(f'range生成数据: {range(1, 5, 2)}')           # range生成数据: range(1, 5, 2)
print(f'range生成数据: {list(range(1, 5, 2))}')     # range生成数据: [1, 3]

# 演示: enumerate()                 基于可迭代类型(字符串, 列表, 元组等), 生成 下标 + 元素的方式, 即: ['a', 'b', 'c']  => [(0, 'a'), (1, 'b'), (2, 'c')]
print(f'list1: {list1}')        # [10, 30, 66, 22]
print(enumerate(list1))     # 直接打印是: 枚举对象的地址值, 无意义, 我们来遍历它. <enumerate object at 0x000001DFB75F6640>

for i in enumerate(list1):
    print(i)        # 格式: (下标, 元素值), 且下标默认从 0 开始.

print('-' * 28)

for i in enumerate(list1, 5):
    print(i)        # 格式: (下标, 元素值), 且下标从 5 开始.
