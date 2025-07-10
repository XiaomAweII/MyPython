"""
关于 函数传递 你需要掌握的两句话.
    1. 函数的返回值可以作为 (另一个函数的)参数进行传递.
    2. 函数可以作为 (另一个函数的)参数进行传递.
"""

# 演示 1. 函数的返回值可以作为 (另一个函数的)参数进行传递.
def fun1():
    return 100      # 返回1个结果 100

def fun2(num):      # 需要1个参数
    print(num)

# 调用fun1()函数, 获取返回值.
a = fun1()          # 相当于 a = 100

# 调用fun2()函数.
fun2(a)

# 合并版.
fun2(fun1())        # 函数的返回值可以作为 (另一个函数的)参数进行传递.
print('-' * 28)

# 演示 直接写函数名, 是函数对象.
def get_sum(a, b):
    return a + b

# 正常调用
sum = get_sum(10, 20)
print(sum)

# 直接写函数名 是 对象, 可以赋值给变量, 那个变量就有了和该函数一样的功能.
hg = get_sum        # 等价于:   def hg(a, b): return a + b
sum2 = hg(20, 30)
print(sum2)
print('-' * 28)


# 演示 2. 函数可以作为 (另一个函数的)参数进行传递.
def get_substract(a, b):
    return a - b

def calculate(a, b, fn):
    """
    自定义函数, 模拟计算器, 传入什么 函数(对象), 就做什么操作.
    :param a: 要操作的第1个整数
    :param b: 要操作的第2个整数
    :param fn: 具体的操作规则
    :return: 计算结果.
    """
    return fn(a, b)


print(calculate(10, 20, get_sum))
print(calculate(10, 20, get_substract))