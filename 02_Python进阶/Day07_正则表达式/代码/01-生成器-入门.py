"""
生成器介绍:
    概述:
        生成器就是用来生成数据的, 用一个, 生成1个, 这样可以节省大量的内存空间.
    大白话解释:
        生成器的推导式写法, 非常类似于以前我们用的 列表, 集合, 字典推导式, 只不过换成 小括号而已.
    实现方式:
        1. 推导式写法.
        2. yield关键字.
    问: 如何从生成器中获取到数据?
    答:
        方式1: next()函数, 逐个获取.
        方式2: 遍历生成器即可.
"""

# 案例1: 回顾之前学的 列表推导式.
# 需求: 生成 1 ~ 5的数字.
list1 = [i for i in range(1, 6)]
print(list1)            # [1, 2, 3, 4, 5]
print(type(list1))      # <class 'list'>

set1 = {i for i in range(1, 6)}
print(set1)             # {1, 2, 3, 4, 5}
print(type(set1))       # <class 'set'>
print('-' * 20)

# 案例2: 演示 生成器的 推导式写法.
# 需求: 生成 1 ~ 5的数字, 尝试用 "元组"推导式的写法写.
tuple1 = (i for i in range(1, 6))           # 这个不是元组推导式, 而是: 生成器(对象)
print(tuple1)            # 地址值: <generator object <genexpr> at 0x0000013BF7655890>
print(type(tuple1))      # <class 'generator'>
print('-' * 20)


# 案例3: 如何从生成器中获取到数据.
# 1. 自定义生成器, 获取 1 ~ 5的整数.
my_generator = (i for i in range(1, 6))

# 2. 从生成器中获取到元素.
# 方式1: next() 逐个获取
print(next(my_generator))   # 1
print(next(my_generator))   # 2
print('-' * 20)

# 方式2: 遍历生成器.
for i in my_generator:
    print(i)
