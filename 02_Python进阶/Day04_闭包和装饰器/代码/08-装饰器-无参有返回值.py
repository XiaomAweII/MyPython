"""
使用装饰器的时候, 要注意:
    装饰器的 内部函数 格式要和 原函数(要被装饰的函数) 保持一致, 要么都是无参无返回, 要么都是无参有返回, 要么都是有参无返回, 要么都是有参有返回...
"""

# 案例: 演示装饰器 装饰 原函数(无参有返回值)
# 需求: 定义无参有返回值的原函数 get_sum(), 用于计算两个整数和. 在不改变该函数的基础上,  给这个函数添加友好提示. 请用所学, 模拟该知识点.
# 1. 定义装饰器.
def print_info(fn_name):
    def inner():        # 内部函数 必须和 要被装饰的函数 格式一致.        有嵌套
        # 添加额外功能
        print('[友好提示] 正在努力计算中...')                        # 有额外功能
        sum = fn_name()                                          # 有引用.
        return sum                                               # 这个是返回 内部函数的 执行结果.
    return inner                                                 # 有返回, 即: 返回的是 内部函数对象


# 2. 定义 原函数, 即: 要被装饰的函数.
@print_info
def get_sum():
    a = 10
    b = 20
    sum = a + b
    return sum

# 在main函数中调用.
if __name__ == '__main__':
    # 3. 普通调用
    # print(get_sum())

    # 4. 装饰器, 写法1: 传统写法.        变量名 = 装饰器名(原函数名)
    # get_sum = print_info(get_sum)
    # sum = get_sum()
    # print(sum)
    # print(get_sum())
    print('-' * 20)

    # 5. 装饰器, 写法2: 语法糖
    print(get_sum())
