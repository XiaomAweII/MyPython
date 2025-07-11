"""
案例: 减肥案例.

需求:
    小明同学当前的体重是 100KG, 每当他跑步一次, 体重减少0.5KG, 每当他大吃大喝一次, 体重增加2KG.
    请用所学的面向对象知识, 完成该题.

分析:
    类:  学生类, Student
        属性:
            当前体重, current_weight
        行为:
            跑步,     run()
            大吃大喝, eat()
"""

# 1. 定义学生类...
class Student():
    # 2. 在 init()魔法方法中, 完成 属性(当前体重)的初始化.
    # 场景1: 无参数.
    def __init__(self):
        self.current_weight = 100        # 单位: kg

    # 场景2: 有参数.
    # def __init__(self, weight):
    #     self.current_weight = weight        # 单位: kg

    # 3. 定义函数, 表示: 跑步, 执行一次, 体重减少0.5KG.
    def run(self):
        # 跑步, 执行一次, 体重减少0.5KG.
        self.current_weight = self.current_weight - 0.5
        # 打印当期体重.
        print(f'当前体重为: {self.current_weight} Kg!')


    # 4. 定义函数, 表示: 大吃大喝, 执行一次, 体重增加2KG.
    def eat(self):
        # 大吃大喝, 执行一次, 体重增加2KG.
        self.current_weight = self.current_weight + 2
        # 打印当期体重.
        print(f'当前体重为: {self.current_weight} Kg!')

    # 当然, 也可以通过魔法方法 str, 用于实现: 打印对象的时候, 直接打印其属性值.
    def __str__(self):
        return f'当前体重为: {self.current_weight} Kg!'

# 在main方法中, 完成测试.
if __name__ == '__main__':
    # 5. 创建对象.
    xm = Student()
    # xm = Student(100)

    # 6. 测试: 跑步的功能.
    xm.run()
    xm.run()
    xm.run()

    # 7. 测试 大吃大喝 的功能.
    xm.eat()
    xm.eat()

    # 8. 打印最终结果.
    # print(xm)
