"""
继承介绍:
    概述:
        现实中的继承指的是: 子承父业, 编程中的继承指的是: 子类从父类继承过来 属性 和 行为.
    格式:
        class 子类名(父类名):
            pass
    例如:
        class A(B):
            pass
    叫法:
        类A: 子类, 派生类, 扩展类.
        类B: 父类, 基类, 超类.
    好处:
        提高代码的复用性.
    细节:
        所有的类都直接或者间接继承自 object类, 它是所有类的父类, 基类.
"""
# 案例: 定义父类Father类, 属性: 性别=男,  行为: 散步.  定义子类Son, 继承父类, 创建对象, 并调用.
# 1. 定义Father类, 充当父类.
class Father(object):
    # 1.1 定义属性.
    def __init__(self):
        self.gender = '男'

    # 1.2 定义行为
    def walk(self):
        print('饭后走一走, 活到99!')

# 2. 定义子类Son类, 充当子类.
class Son(Father):  # 继承关系 Son -> Father -> object
    pass

# 在main方法中进行测试.
if __name__ == '__main__':
    # 3. 创建子类对象.
    s = Son()

    # 4. 尝试调用 从父类继承过来 的内容(属性 和 行为)
    print(s.gender)
    s.walk()