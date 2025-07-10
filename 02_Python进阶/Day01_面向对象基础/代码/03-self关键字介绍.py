"""
self关键字介绍:
    概述:
        它代表本类当前对象的引用, 一般用于: 函数中, 即: 谁调用这个函数, self就代表谁(哪个对象).
    简单记忆:
        谁(本类对象)调用(函数), self就代表谁.

"""

# 需求1: 创建汽车类Car, 分别创建两个对象, 观察结果.
class Car:
    # 定义行为, 跑.
    def run(self):
        print('汽车会跑!')
        print(f'self: {self}')



# 测试代码一般写到 main方法中.
if __name__ == '__main__':
    # 创建对象.
    c1 = Car()
    c2 = Car()

    # 调用类的成员.
    c1.run()
    c2.run()
    print('-' * 20)

    # 打印对象名
    print(c1)   # <__main__.Car object at 0x000002244653D5E0>
    print(c2)   # <__main__.Car object at 0x000002244653D520>
