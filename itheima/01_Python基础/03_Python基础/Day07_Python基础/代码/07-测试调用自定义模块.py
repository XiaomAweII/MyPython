"""
案例: 演示如何调用 自定义模块.

细节:
    1.  1个.py文件就可以看做是1个模块, 文件名 = 模块名, 所以: 文件名也要符合标识符的命名规范.
    2.  __name__属性, 当前模块中打印的结果是 __main__, 在调用者中打印的结果是: 调用的模块名
    3. 如果导入的多个模块中, 有同名函数, 默认会使用 最后导入的 模块的函数.
    4. __all__ 属性只针对于 from 模块名 import * 这种写法有效, 它只会导入 __all__记录的内容
"""

# 需求: 自定义 my_module1模块, 然后再其中定义一些函数.  在当前模块中 调用 my_module1模块的内容.
# import my_module1 as m1
# import my_module1 as m2
#
# # print(m1.get_sum(10, 20))
# # m1.fun01()
# # m1.fun02()
#
# m1.fun01()
# m2.fun01()


# 如果导入的多个模块中, 有同名函数, 默认会使用 最后导入的 模块的函数.
# from my_module1 import fun01
# from my_module2 import fun01
#
# fun01()


# __all__ 属性只针对于 from 模块名 import * 这种写法有效, 它只会导入 __all__记录的内容
from my_module1 import *

# print(get_sum(10, 20))  # 报错, 因为 __all__ 属性没有记录它.
fun01()
fun02()
