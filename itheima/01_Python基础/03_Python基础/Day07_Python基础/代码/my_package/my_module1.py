# 这个是我们自定义的第 1个 模块.

__all__ = ['fun01', 'fun02']

# 函数 = 模块的功能, 相当于: 工具包中的工具
def get_sum(a, b):
    print('我是 my_module1 模块的 函数')
    print(__name__)
    return a + b


def fun01():
    print('我是 my_module1 模块的 函数')
    print('----- fun01 函数 -----')
    print(__name__)


def fun02():
    print('我是 my_module1 模块的 函数')
    print('----- fun02 函数 -----')
    print(__name__)


# 实际开发中, 定义好模块后, 一般会对模块的功能(函数)做测试.
# 如下的测试代码, 在 调用者中 也会被执行. 但真实的业务场景, 测试代码 在 调用者中是不能被执行的.
# 那, 如何解决这个问题呢?
# 答案: __name__ 属性 即可解决这个事儿, 它在当前模块中打印的结果是 __main__,
#      在调用者模块中打印的是 当前的模块名.

# # 测试代码
# print(get_sum(10, 20))
# fun01()
# fun02()

# 如果条件成立, 说明是在 当前模块中执行的.
if __name__ == '__main__':
    # 测试代码
    print(get_sum(10, 20))
    fun01()
    fun02()