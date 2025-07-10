"""
Python中 函数的参数写法 主要有如下的四种:
    位置参数
    关键字参数
    默认参数(缺省参数)
    不定长参数

细节:
    1. 位置参数 和 关键字参数 是针对于 实参 来讲的.
    2. 缺省参数 和 不定长参数 是针对于 形参 来讲的.

不定长参数:
    概述:
        不定长参数也叫 可变参数, 即: 参数的个数是可以变化的.
    应用场景:
        适用于 实参的个数不确定的情况, 就可以把 形参定义成 可变参数.
    格式:
        *args           只能接收所有的 位置参数, 封装到: 元组中.
        **kwargs        只能接收所有的 关键字参数, 封装到: 字典中.
    细节:
        1. 关于实参, 位置参数在前, 关键字参数在后.
        2. 关于形参, 如果两种 可变参数都有, 则: *args 在前, **kwargs 在后.
        3. 关于形参, 如果既有 缺省参数 又有不定长参数, 则编写顺序为:  *args, 缺省参数, **kwargs

"""

# 需求1: 演示 不定长参数(可变参数)之 接收 位置参数.
def method01(*args):                      # 约定俗成, 变量名可以任意写, 但是建议写成 args
    print(f'接收到的所有参数为: {args}')     # 你把 args变量当做 元组来用即可.
    print(type(args))

# 需求2: 演示 不定长参数(可变参数)之 接收 关键字参数.
def method02(**kwargs):
    print(f'接收到的所有参数为: {kwargs}')
    print(type(kwargs))

# 需求3: 同时定义两种 参数.
#             不定长(可变)参数
def method03(*args, **kwargs):
    print(f'args: {args}')
    print(f'kwargs: {kwargs}')

# 需求4: 同时定义 缺省参数, 不定长参数.
#          不定长参数   缺省参数    不定长参数
def method04(*args, name='张三', **kwargs):
    print(f'name: {name}')
    print(f'args: {args}')
    print(f'kwargs: {kwargs}')



# main充当程序的 主入口.
if __name__ == '__main__':
    # 调用 method01()函数.
    method01(1, '张三', 23)
    # method01(1, '张三', age=23)       # 报错, *args 只能接收所有的 位置参数.
    print('-' * 28)

    # 调用 method02()函数.
    method02(name='张三', age=23, phone='13112345678')
    # method02('李四', age=24, phone='13112345678')     # 报错, **kwarg 只能接收所有的 关键字参数.
    print('-' * 28)

    # 调用 method03()函数
    #                位置参数           关键字参数
    method03(10, 20, 'aa', name='王五', age=25, address='杭州')
    print('-' * 28)

    # 调用 method04()函数
    #                位置参数             关键字参数
    method04(10, 20, 'aa', name='王五', age=25, address='杭州')

