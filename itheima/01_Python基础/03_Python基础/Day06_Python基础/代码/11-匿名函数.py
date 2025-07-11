"""
匿名函数介绍:
    概述:
        没有名字的函数 就叫 匿名函数.
    格式:
        变量名 = lambda 形参列表 : 函数体(只能写一行代码, 且该行代码的结果会被自动返回)
    细节:
        1. Python的匿名函数 类似于 Java中的Lambda表达式.
        2. 匿名函数适用于简单的业务需求, 即: 函数体只有一行代码的函数.
        3. 匿名函数的应用场景:
            当对方法仅调用一次.
            匿名函数 可以作为 函数对象 进行传递.
"""

# 案例1: 匿名函数入门.
# 需求: 定义函数, 用于计算两个整数和.
# 场景1: 普通函数.
def get_sum(a, b):
    return a + b

print(get_sum(10, 20))


# 场景2: 匿名函数实现
my_get_sum = lambda a, b :  a + b
print(my_get_sum(11, 22))
print('-' * 28)


# 案例2: 定义函数, 接收2个整数, 分别计算两个整数的: 和, 差, 积, 商, 最大值, 最小值.
# 场景1: 普通函数.
def get_sum(a, b):
    return a + b            # 求和

def get_sub(a, b):
    return a - b            # 差

def get_mul(a, b):
    return a * b            # 积

# ...... 依次类推, 把其他的函数全部做出来即可, 但是这样就会导致 多个需求, 每个需求都对应1个函数, 函数太多了, 一方面开发效率下降,
# 另一方面不方便维护和管理, 光记忆函数名 就是1个非常庞大的工程, 如何解决呢?
# 可以通过 匿名函数的方式 解决.


# 场景2: 匿名函数 方式实现.
def my_func(a, b, fn):
    """
    自定义函数, 根据传入的规则, 来计算两个整数的结果.
    :param a: 要操作的第1个数据
    :param b: 要操作的第2个整数
    :param fn: 具体的 计算规则 函数.
    :return: 具体的计算结果.
    """
    return fn(a, b)


# 调用 my_func()函数.
# 加
# print(my_func(10, 3, get_sum))
print(my_func(10, 3, lambda a, b : a + b))

# 减
# print(my_func(10, 3, get_sub))
print(my_func(10, 3, lambda a, b: a - b))

# 乘
print(my_func(10, 3, lambda a, b: a * b))

# 除
print(my_func(10, 3, lambda a, b: a // b))

# 最大值
# print(my_func(10, 3, lambda a, b: max(a, b)))
print(my_func(10, 3, lambda a, b: a if a >= b else b))

# 最小值
print(my_func(10, 3, lambda a, b: a if a <= b else b))

# 平均值
print(my_func(10, 3, lambda a, b: (a + b) // 2))


