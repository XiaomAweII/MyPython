"""
案例:
    演示 在类内部, 通过 self关键字, 访问类内部(自己的) 函数.

细节:
    在类中调用 类的行为(函数), 可以通过 self.的方式 调用.
"""

# 需求: 定义汽车类, 其有 run() 和 work()两个函数, run()表示跑的功能, 在work()函数中调用run()函数, 并在main方法中, 创建对象, 调用并测试.

# 1.定义汽车类
class Car():
    # 2. run()函数, 表示: 跑的行为.
    def run(self):
        print('我是 run 函数')
        print('汽车会跑!')

    # 3. 在work()函数中调用run()函数
    def work(self):
        print('我是 work 函数')
        self.run()      # 类内部, 可以通过 self. 的方式, 调用类的其它成员.


# 在main方法中测试.
if __name__ == '__main__':
    # 4. 创建对象.
    c1 = Car()
    c2 = Car()

    # 5. 调用 run()函数
    c1.run()
    c2.run()
    print('-' * 20)

    # 6. 调用 work()函数
    c1.work()
    c2.work()
