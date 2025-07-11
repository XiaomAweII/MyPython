"""
方法重写 解释:
    概述:
        子类中出现和父类一模一样的 属性 或者 行为(函数)时, 称为: 方法重写.
    应用场景:
        当子类需要从父类沿袭一些功能, 但是功能主体又有自己独有需求的时候, 就可以考虑使用方法重写了.
"""

# 故事3: 小明掌握了老师傅和黑马的技术后, 自己钻研出一套 自己的 独门配方摊煎饼果子法. 请用所学, 模拟这个知识点.
# 1. 定义师傅类Master.
class Master(object):
    # 1.1 属性, kongfu = '[古法摊煎饼果子技术]'
    def __init__(self):
        self.kongfu = '[古法摊煎饼果子技术]'

    # 1.2 行为, make_cake(), 表示: 摊煎饼.
    def make_cake(self):
        print(f'采用 {self.kongfu} 制作煎饼果子!')


# 2. 定义黑马学校类School.
class School(object):
    # 2.1 属性, kongfu = '[黑马AI摊煎饼果子技术]'
    def __init__(self):
        self.kongfu = '[黑马AI摊煎饼果子技术]'

    # 2.2 行为, make_cake(), 表示: 摊煎饼.
    def make_cake(self):
        print(f'采用 {self.kongfu} 制作煎饼果子!')


# 3. 定义徒弟类Prentice, 继承自 师傅类.
class Prentice(School, Master):     # 多继承, 同名属性和行为, 优先参考第1个父类, 即: 从左往右的顺序.
    # 3.1 属性
    def __init__(self):
        self.kongfu = '[独创(自研) 摊煎饼果子技术]'

    # 3.2 行为, make_cake(), 表示: 摊煎饼.
    def make_cake(self):
        print(f'采用 {self.kongfu} 制作煎饼果子!')


# 在main函数中测试.
if __name__ == '__main__':
    # 4. 创建徒弟类对象.
    p = Prentice()
    # 5. 打印 徒弟类对象 从父类继承过来的 属性.
    print(p.kongfu)
    # 6. 打印 徒弟类对象 从父类继承过来的 行为.
    p.make_cake()

    # 7. 查看MRO规则, method Resolution Order, 方法解析顺序.
    print(Prentice.__mro__)
    print(Prentice.mro())
