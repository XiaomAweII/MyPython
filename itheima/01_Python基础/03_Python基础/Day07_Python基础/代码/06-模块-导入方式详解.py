"""
模块介绍:
    概述:
        模块指的是 Module, 在Python中, 1个.py文件 = 1个模块.
        你可以把 模块理解为 工具包, 工具包中有很多的工具.  其实就是: 每个.py文件中都有很多的 函数, 这些函数都有不同的功能.
    大白话:
        学模块, 就是记忆一些 .py文件, 以及其中的一些 函数.
        例如: 随机数相关 用 random模块,   日期相关用 time模块,  文件路径相关用 os模块...

    模块的 导入方式:
        方式1: import 模块名                         后续通过  模块名.函数名() 的方式调用.   模块下所有函数 均可使用
        方式2: import 模块名 as 别名                 后续通过  别名.函数名() 的方式调用.     模块下所有函数 均可使用
        方式3: from 模块名 import 函数名            后续通过  函数名() 的方式直接调用.       只能使用该模块下 导入的函数.
        方式4: from 模块名 import 函数名 as 别名    后续通过  别名() 的方式直接调用.         只能使用该模块下 导入的函数.
        方式5: from 模块名 import *               后续通过  函数名() 的方式直接调用.       模块下所有函数 均可使用
"""

# 案例: 演示 导入模块的几种方式的 用法.
# 测试用例.   time模块下的 time()函数, sleep()函数.


# 演示 方式1: import 模块名                         后续通过  模块名.函数名() 的方式调用.   模块下所有函数 均可使用
# import time
#
# print('--- start ---')
# time.sleep(2)               # 让程序休眠 2秒.
# print(time.localtime())     # 获取系统的的本地时间
# print(time.time())          # 1719113316.0237932,  从时间原点(1970年1月1日 00:00:00 ~ 至今)的秒值.
# print('--- end ---')

# 演示 方式2: import 模块名 as 别名                 后续通过  别名.函数名() 的方式调用.     模块下所有函数 均可使用
# import time as t
#
# print('--- start ---')
# t.sleep(2)               # 让程序休眠 2秒.
# print(t.localtime())     # 获取系统的的本地时间
# print(t.time())          # 1719113316.0237932,  从时间原点(1970年1月1日 00:00:00 ~ 至今)的秒值.
# print('--- end ---')


# 演示 方式3: from 模块名 import 函数名            后续通过  函数名() 的方式直接调用.       只能使用该模块下 导入的函数.
# from time import sleep, localtime

# print('--- start ---')
# sleep(2)               # 让程序休眠 2秒.
# print(localtime())     # 获取系统的的本地时间
# # print(time())          # 报错, 因为没有导入, 所以使用不了.
# print('--- end ---')

# 演示 方式4: from 模块名 import 函数名 as 别名    后续通过  别名() 的方式直接调用.         只能使用该模块下 导入的函数.
# from time import sleep as sl, localtime as lt
#
# print('--- start ---')
# sl(2)               # 让程序休眠 2秒.
# print(lt())     # 获取系统的的本地时间
# # print(time())          # 报错, 因为没有导入, 所以使用不了.
# print('--- end ---')


# 演示 方式5: from 模块名 import *               后续通过  函数名() 的方式直接调用.       模块下所有函数 均可使用
from time import *

print('--- start ---')
sleep(2)               # 让程序休眠 2秒.
print(localtime())     # 获取系统的的本地时间
print(time())
print('--- end ---')

