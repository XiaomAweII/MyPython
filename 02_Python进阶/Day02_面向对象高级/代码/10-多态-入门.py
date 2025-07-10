"""
多态介绍:
    概述:
        多态指的是同一个事物在不同场景下表现出来的不同形态, 状态.
        Python中的多态指的是, 同一个函数, 传入不同的对象, 会实现不同的结果.
    多态的前提条件:
        1. 要有继承关系.
        2. 要有方法重写.
        3. 要有父类引用指向子类对象.            an:Animal = Dog()   an:Animal = Cat()      狗是动物, 猫是动物
    好处:
        提高代码的可维护性. 实现: 1个函数, 多种效果.
    应用场景:
        父类型充当函数形参的类型, 这样可以接受其任意的子类对象, 实现: 传入什么(子类)对象, 就调用其对应的功能.

    细节:
        有人说, Python中的多态其实是"伪多态", 因为 函数的形参类型, 并不能严格限制必须传入该类型或者其子类型的对象,
        其实传入其它类型的对象也是可以的.
"""


# 案例: 动物类案例.
# 1. 定义动物类, 有 speak()函数, 表示: 叫.
class Animal(object):
    def speak(self):
        print('动物会叫!')


# 2. 定义狗类, 继承自动物类, 重写: speak()函数.
class Dog(Animal):
    def speak(self):
        print('狗 汪汪汪 叫!')


# 3. 定义猫类, 继承自动物类, 重写: speak()函数.
class Cat(Animal):
    def speak(self):
        print('猫 喵喵喵 叫!')

# 扩展: 定义1个手机类.
class Phone(object):
    def speak(self):
        print('我像只鱼儿在你的荷塘, 只为寻找那...')


# 4. 定义函数 make_noise(动物类对象), 接收动物对象, 实现: 传入什么动物, 就怎么叫.
def make_noise(an: Animal):  # an:Animal 意思是: an "必须" 是Animal类的对象 或者 其子类对象.
    # 接收动物对象, 实现: 传入什么动物, 就有: 对应的叫声.
    an.speak()      # 猫对象.speak(),     狗对象.speak()


# 在main方法中测试.
if __name__ == '__main__':
    # 5. 分别创建猫类, 狗类对象.
    c = Cat()
    d = Dog()

    # 6. 调用 make_noise()函数, 实现: 多态.   传入狗对象, 就: 汪汪汪叫,  传入猫对象, 就: 喵喵喵叫.
    make_noise(c)       # 猫 喵喵喵 叫!
    make_noise(d)       # 猫 喵喵喵 叫!
    print('-' * 20)

    # 7. 演示 "伪多态"
    p = Phone()
    make_noise(p)

    # 扩展: 看变量属于什么类型.
    print(type(p))                  # type()函数, 看对象属于什么类型.
    print(isinstance(p, Phone))     # isinstance(对象名, 类型) 判断对象是否是对应的类型.
    print(isinstance(p, Animal))    # isinstance(对象名, 类型) 判断对象是否是对应的类型.
    print(isinstance(c, Animal))    # True