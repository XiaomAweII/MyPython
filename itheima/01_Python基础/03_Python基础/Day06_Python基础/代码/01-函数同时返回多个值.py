"""
案例: 演示函数同时返回多个值.

需求: 自定义函数 calculate(), 接收两个整数, 然后返回它们的 加减乘除结果.
"""

# 1. 定义函数calculate(), 接收两个整数.
def calculate(a, b):
    """
    自定义函数, 模拟计算器, 计算两个整数的 加减乘除结果.
    :param a: 要操作的第1个整数
    :param b: 要操作的第2个整数
    :return: 加减乘除结果
    """
    # 具体的计算加减乘除结果的动作
    sum = a + b
    sub = a - b
    mul = a * b
    div = a // b
    # 2. 返回它们的 加减乘除结果.
    return sum, sub, mul, div       # 同时返回多个值, 默认会用 元组封装.
    # return [sum, sub, mul, div]       # 当然也可以手动把多个值封装成 列表或者集合, 然后返回.
   

# 3. 调用函数, 进行测试.
if __name__ == '__main__':
    result = calculate(10, 3)
    print(result)           # (13, 7, 30, 3)
    print(type(result))     # <class 'tuple'>