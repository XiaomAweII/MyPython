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

    细节:
        只要def函数中, 看到了 yield关键字, 它就是生成器.
        yield关键字作用: 临时存储所有的数据, 并放到生成器中, 调用函数时, 会返回1个生成器对象.
"""
# 需求1: 自定义 get_list()函数, 实现: 返回1个 包含 1 ~ 5 整数的 列表.
def get_list():
    # return [i for i in range(1, 6)]       # 推导式写法.

    # 分解版写法.
    # 1. 定义1个列表.
    my_list = []
    # 2. 遍历, 获取到 1 ~ 5的整数.
    for i in range(1, 6):
        # 3. 把获取到的整数 添加到 列表中.
        my_list.append(i)
    # 4. 返回 列表
    return my_list


# 需求2: 自定义 get_generator()函数, 实现: 返回1个 包含 1 ~ 5 整数的 生成器.
def get_generator():
    # return (i for i in range(1, 6))         # 推导式写法, 返回: 生成器.

    # yield方式, 获取生成器对象.
    for i in range(1, 6):
        yield i     # 把每个i的值都放到生成器中, 函数结束后, 会返回: 生成器对象.
                    # yield作用: 1.创建生成器.   2.把i的每个值放到生成器中.   3.返回生成器

# 在main函数中测试
if __name__ == '__main__':
    # 1. 测试 get_list()函数.
    list1 = get_list()
    print(list1)        # [1, 2, 3, 4, 5]
    print(type(list1))  # <class 'list'>
    print('-' * 20)

    # 2. 测试 get_generator()函数
    my_g = get_generator()
    print(my_g)          # 地址值, 生成器对象的地址.
    print(type(my_g))    # <class 'generator'>
    print('-' * 20)

    # 3. 从生成器中获取数据.
    # 方式1: next()
    # print(next(my_g))   # 1
    # print(next(my_g))   # 2
    # print(next(my_g))   # 3
    # print(next(my_g))   # 4
    # print(next(my_g))   # 5
    # print(next(my_g))   # 报错, 因为生成器中已经没有数据了.

    # 方式2: 遍历 生成器.
    for i in my_g:
        print(i)

