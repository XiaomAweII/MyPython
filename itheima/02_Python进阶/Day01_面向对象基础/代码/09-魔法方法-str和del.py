"""
案例: 演示魔法方法 init()之 无参数情况.

魔法方法解释:
    概述:
        在Python中, 有一类方法是用来 对Python类的功能做 加强(扩展)的, 且这一类方法都无需手动调用.
        在满足特定情况的场景下, 会被自动调用, 它们就称之为: 魔法方法.
    格式:
        __魔法方法名__()      注意, 这里是两个下划线.
    常用的魔法方法:
        __init__()      用于初始化对象的属性值的, 在创建对象的时候, 会被: 自动调用.           c1 = Car()
        __str__()       用于快速打印对象的各个属性值的, 在输出语句打印对象的时候会自动调用.     print(c1)
        __del__()       当删除对象的时候, 或者main函数执行完毕后, 会自动调用.
"""

# 需求: 通过外部给车这个对象设置 color属性, number属性值.
class Car:      # Python类的命名, 遵循: 大驼峰命名法. 例如: HelloWorld
    # 魔法方法 __init__()完成, 对 属性 的初始化操作.
    def __init__(self, color, number):
        self.color = color
        self.number = number

    # 魔法方法 __str__(), 输出语句打印对象时, 会自动调用, 一般用于打印对象的 各个属性值.
    def __str__(self):
        return f"汽车颜色: {self.color}, 轮胎数: {self.number}"


    # 魔法方法 __del__(), del删除对象的时候 或者 文件执行结束后, 会自动调用它, 用于: 释放资源.
    def __del__(self):
        print(f'{self} 对象被释放了!')


# main函数, 一般用于写: 测试代码
if __name__ == '__main__':
    c1 = Car('黑色', 6)              # 创建对象, 会自动调用 init()魔法方法

    # 手动打印 对象的属性值, 相对: 较麻烦.
    print(f'汽车颜色: {c1.color}, 轮胎数: {c1.number}')        # 红色, 4
    print(c1)       # 输出语句打印对象, 底层默认调用了 对象的 __str__()魔法方法, 一般我们都会通过 __str__()打印对象的各个属性值.
    print("-" * 20)

    c2 = Car('猛男粉', 8)
    print(c2)
    print("*" * 20)

    # del c1    从内存中, 删除对象c1
    # del c2
