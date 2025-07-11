"""
至此, 我们已经初始了 try.except的语法, 接下来我们来看看 它是如何 精准的捕获某个 指定异常的.

格式:
    try:
        可能出问题的代码
    except Exception as e:
        出问题后的解决方案

格式解释:
    Exception   所有异常的父类, 即: 它代表着所有的异常.
    e           类似于我们以前写的变量名, 这里它是: 异常对象名.

细节:
    还可以写成 except (异常1, 异常2) as e    这种写法是捕获多种异常.
"""

# 1. 读取了1个不存在的文件.
# src_f = open('1.txt', 'r')      # FileNotFoundError

# 2. 除零异常.
# print(10 // 0)                    # ZeroDivisionError

# 3. 变量未定义异常.
# print(name)                 # NameError


# 4. 演示 try.except 捕获指定异常.
# 场景1: 捕获 单个异常.
# try:
#     # print(name)   # NameError
#     src_f = open('1.txt', 'r')
# except NameError as e:              # 只能捕获 NameError 异常.
#     print('哎呀, 程序出问题了!')
#     # e就是异常对象, 代表着: 异常信息.  可以直接把它输出到控制台.
#     # print(e)


# 场景2: 捕获 多种异常.
# try:
#     # print(name)                      # NameError
#     # src_f = open('1.txt', 'r')       # FileNotFoundError
#     print(10 // 0)
# except (NameError, FileNotFoundError) as e:
#     # print('哎呀, 程序出问题了!')
#     # e就是异常对象, 代表着: 异常信息.  可以直接把它输出到控制台.
#     print(e)

# 场景3: 通用的 捕获异常的 方案.
try:
    print(name)                      # NameError
    # src_f = open('1.txt', 'r')       # FileNotFoundError
    # print(10 // 0)
except Exception as e:
    # print('哎呀, 程序出问题了!')
    # e就是异常对象, 代表着: 异常信息.  可以直接把它输出到控制台.
    print(e)

print('看看我执行了吗?')
