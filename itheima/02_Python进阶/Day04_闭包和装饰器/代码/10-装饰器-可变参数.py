"""
使用装饰器的时候, 要注意:
    装饰器的 内部函数 格式要和 原函数(要被装饰的函数) 保持一致, 要么都是无参无返回, 要么都是无参有返回, 要么都是有参无返回, 要么都是有参有返回...
"""


# 案例: 演示装饰器 装饰 原函数(可变参数)
# 需求: 定义有可变参数的原函数 get_sum(), 用于计算 元组 及 字典的值的和. 在不改变该函数的基础上,  给这个函数添加友好提示. 请用所学, 模拟该知识点.
# 1. 定义装饰器.
def print_info(fn_name):
    def inner(*args, **kwargs):  # 内部函数 必须和 要被装饰的函数 格式一致.        有嵌套
        # 添加额外功能
        print('[友好提示] 正在努力计算中...')  # 有额外功能
        sum = fn_name(*args, **kwargs)  # 有引用.
        return sum  # 这个是返回 内部函数的 执行结果.

    return inner  # 有返回, 即: 返回的是 内部函数对象


# 2. 定义 原函数, 即: 要被装饰的函数.
@print_info
def get_sum(*args, **kwargs):
    # 定义求和变量.
    sum = 0
    # 求所有的 位置参数的和, 即: *args -> 元组.
    for i in args:
        sum += i
    # 求所有的 关键字参数的和, 即: **kwargs -> 字典
    for i in kwargs.values():
        sum += i
    # 返回求和结果.
    return sum


# 在main函数中调用.
if __name__ == '__main__':
    # 3. 装饰器, 写法2: 语法糖
    print(get_sum(1, 2, 3, a=4, b=5, c=6))
