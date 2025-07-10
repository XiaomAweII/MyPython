# 单继承指的是: 类只能继承自另外的1个类, 从中继承过来 属性 和 行为.

# 故事1: 一个摊煎饼的老师傅, 在煎饼果子界摸爬滚打多年, 研发了一套精湛的摊煎饼的奇数. 他(老师傅)要传授这个套技术给徒弟. 请用所学, 模拟这个知识点.
# 1. 定义师傅类Master.
class Master(object):
    # 1.1 属性, kongfu = '[古法摊煎饼果子技术]'
    def __init__(self):
        self.kongfu = '[古法摊煎饼果子技术]'

    # 1.2 行为, make_cake(), 表示: 摊煎饼.
    def make_cake(self):
        print(f'采用 {self.kongfu} 制作煎饼果子!')


# 2. 定义徒弟类Prentice, 继承自 师傅类.
class Prentice(Master):
    pass


# 在main函数中测试.
if __name__ == '__main__':
    # 3. 创建徒弟类对象.
    p = Prentice()
    # 4. 通过徒弟类对象, 调用 从师傅类(父类) 中继承过来的 属性 和 行为.
    print(f'徒弟从师傅继承过来的属性: {p.kongfu}')
    p.make_cake()       # 徒弟从师傅继承过来的 行为.