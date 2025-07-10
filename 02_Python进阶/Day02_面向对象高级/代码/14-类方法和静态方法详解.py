"""
类方法 和 静态方法详解:
    概述:
        类方法:
            1. 第1个参数必须是 当前类的对象, 一般用 cls当做变量名(即: class)
            2. 类方法必须通过 @classmethod 来修饰.
            3. 类方法是属于 类的方法, 能被该类下所有的对象所共享.
            4. 可以通过 类名. 或者 对象名. 的方式调用, 推荐: 前者.
        静态方法:
            1. 静态方法没有参数的硬性要求, 可以1个参数都不传.
            2. 静态方法必须通过 @staticmethod 来修饰.
            3. 类方法是属于 类的方法, 能被该类下所有的对象所共享.
            4. 可以通过 类名. 或者 对象名. 的方式调用, 推荐: 前者.

    区别:
        类方法 和 静态方法的区别: 要不要传参, 即: 第一个参数是写 还是 不写, 再简单点说: 是否需要使用 该类的对象, 用就定义成 类方法, 不用就定义成 静态方法.

    细节:

"""

# 需求: 定义学生类, 有 类属性 teacher_name, 对象属性: name.  类方法, 静态方法, 在测试类中进行测试.

# 1. 定义学生类, 有 类属性 teacher_name, 对象属性: name.  类方法, 静态方法
class Student(object):
    # 1.1 老师名字.
    teacher_name = '夯哥'  # 类属性, 每个对象所共享.

    # 1.2 name属性, 每个学生的名字都不一样, 所以定义成: 对象属性.
    def __init__(self):
        self.name = '张三'  # 对象属性.

    # 1.3 定义静态方法, 访问: teacher_name 这个类变量.
    @staticmethod
    def method01():
        print(Student.teacher_name)  # 类名. 的方式, 访问: 类变量.

    # 1.4 定义类方法, 访问: teacher_name 这个类变量.
    @classmethod
    def method02(cls):      # 这里的cls 是 class的意思.
        print(cls.teacher_name)  # 类名. 的方式, 访问: 类变量.


if __name__ == '__main__':
    # 2. 创建学生对象.
    s = Student()
    s.method01()        # 对象名. 的方式 访问静态方法, 可以, 但是不推荐.

    Student.method01()  # 类名. 的方式, 访问静态方法, 推荐.
    Student.method02()  # 类名. 的方式, 访问静态方法, 推荐.