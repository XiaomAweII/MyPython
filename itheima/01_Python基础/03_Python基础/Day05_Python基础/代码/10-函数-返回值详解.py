"""
函数 返回值解释:
    概述:
        返回值指的是 函数操作完毕后, 需要返回给 调用者 1个什么结果.
    格式:
        在函数内部通过 return 返回值 这个格式即可返回.
    细节:
        在哪里调用函数, 就把返回值返回到哪里.
"""

# 接上个脚本, 即: 09-函数-参数解释.py  接着写.
# 写法3: 有参有返回值的函数.  即: 计算两个整数和, 并返回结果.
# 定义函数
def get_sum3(a, b):     # 形参
    """
    该函数用于计算两个整数和
    :param a: 求和计算的第1个整数
    :param b: 求和计算的第2个整数
    :return: 返回求和结果.
    """
    sum = a + b
    return sum


# 调用函数
sum = get_sum3(1, 2)      # 实参
print(f'求和结果为: {sum}')
print(f'平均值为: {sum // 2}')