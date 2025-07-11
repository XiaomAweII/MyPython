"""
property 关键字介绍:
    概述:
        它是用来修饰 函数的, 目的是: 简化代码开发.
    格式:
        1. property充当 装饰器的用法.
            @property               修饰的是 获取值的方法, 即: get_xxx方法
            @方法名.setter           修饰的是 设置值的方法, 即: set_xxx方法,  这里的方法名指的是: @property修饰的方法名
        2. property充当 类变量
            类变量名 = property(获取值的方法, 设置值的方法)
"""
# 需求: 定义学生类, 有1个私有的属性name, 对外提供公共的访问方式, 让外界访问它.
class Student(object):
    # 初始化属性
    def __init__(self):
        self.__name = ''        # 私有属性.

    # 获取值的方法
    # @property
    # def get_name(self):
    #     return self.__name
    #
    # # 设置值的方法
    # @get_name.setter
    # def set_name(self, name):
    #     self.__name = name

    # property 充当装饰器, 结合 get_xxx, set_xxx的终极写法.
    @property
    def name(self):
        return self.__name

    # 设置值的方法
    @name.setter
    def name(self, name):
        self.__name = name

# 在main中测试
if __name__ == '__main__':
    # 场景1: 私有属性, 然后通过 类提供的 公共的访问方式来访问.
    # # 创建对象
    # s = Student()
    # # 设置值
    # s.set_name('乔峰')
    # # 获取值
    # print(s.get_name())

    # 场景2: 私有属性, property充当 装饰器后的写法.
    # # 创建对象
    # s = Student()
    # # 设置值
    # s.set_name = '杨过'
    # # 获取值
    # print(s.get_name)

    # 场景3: 私有属性, property充当 装饰器后的 最终
    # 创建对象
    s = Student()
    # 设置值
    s.name = '杨过'
    # 获取值
    print(s.name)