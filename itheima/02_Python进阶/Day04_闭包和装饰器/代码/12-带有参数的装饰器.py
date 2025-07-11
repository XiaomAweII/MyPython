"""
带有参数的装饰器, 细节:
    1个装饰器 的参数, 只能有 1个
"""

# 需求: 定义1个技能装饰 减法运算, 又能装饰加法运算的装饰器, 请用所学, 实现该需求.
# 思路1: 为了演示 1个装饰器 只能有1个参数.
# 1. 定义装饰器, 实现: 传入 add()函数, 就提示 加法运算.   传入 substract()函数, 就提示: 减法运算.
def logging(flag):
    def print_info(fn_name):                # 这个才是我们的装饰器, 因为1个装饰器的参数只能有1个, 所以在其外边在包裹一层, 专门用于传入: 参数.
    # def print_info(fn_name, flag):        # 思路问题, 但是语法报错. 因为1个装饰器的参数只能有1个
        def inner(a, b):                    # 有嵌套
            if flag == '+':
                print('正在努力计算 加法 中...')    # 有额外功能
            elif flag == '-':
                print('正在努力计算 减法 中...')    # 有额外功能
            fn_name(a, b)                  # 有引用
        return inner                   # 有返回
    return print_info


# 思路2: 为了做题, 即: 实现这个需求. 考虑下, 能不能通过 fn_name的判断, 实现你的需求.
# def print_info(fn_name):
#     def inner(a, b):                    # 有嵌套
#         if fn_name.__name__ == 'add':
#             print('正在努力计算 加法 中...')    # 有额外功能
#         elif fn_name.__name__ == 'substract':
#             print('正在努力计算 减法 中...')    # 有额外功能
#         fn_name(a, b)                  # 有引用
#     return inner


# 2. 定义函数 add(), 表示: 加法运算.
@logging('+')
# @print_info
def add(a, b):
    result = a + b
    print(result)

# 3. 定义函数 substract(), 表示: 减法运算.
@logging('-')
# @print_info
def substract(a, b):
    result = a - b
    print(result)


# 在main函数中测试.
if __name__ == '__main__':
    # 4. 调用add(), 实现: 计算两个整数和.
    add(11, 33)
    print('-' * 20)

    # 5. 调用substract(), 实现: 计算两个整数差.
    substract(100, 20)