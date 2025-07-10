"""
函数的嵌套 解释:
    概述:
        函数的嵌套指的是 函数的嵌套调用, 并不是函数的嵌套定义.
        例如: 在 func_B 函数中 调用 func_A 这个函数, 就叫: 函数的嵌套调用.
    回顾:
        顺序结构: 代码会按照 从上往下, 从左往右, 依次逐行执行.
"""

# 1. 定义func_A()函数.
def func_A():
    print('----- func_A start -----')       # 2
    print('我是 funcA函数 ')                 # 3
    print('----- func_A end -----')         # 4


# 2. 定义func_B()函数.
def func_B():
    print('----- func_B start -----')       # 1
    # 3. 在这里调用 funcA()函数.
    func_A()
    print('----- func_B end -----')         # 5


# 4. 在main函数中, 调用 func_B()函数, 观察程序的打印结果.
# main函数是程序的主入口, 所有代码的执行都是从这里开始的.
# main方法可以省略不写(建议写), Python解释器底层执行的时候, 会自动帮我们加.
if __name__ == '__main__':
    func_B()
