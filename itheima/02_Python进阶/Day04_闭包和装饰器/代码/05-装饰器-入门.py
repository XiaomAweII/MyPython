"""
装饰器介绍:
    概述:
        装饰器 = 闭包函数, 即: 装饰器 是 闭包函数的一种写法.
    目的/作用:
        在不改变原有函数的基础上, 对原有函数功能做 增强.
    前提条件:
        1. 有嵌套.
        2. 有引用.
        3. 有返回.
        4. 有额外功能.
    装饰器的用法:
        写法1: 传统写法.
            变量名 = 装饰器名(要被装饰的原函数名)
            变量名()           # 执行的就是, 装饰后的 原函数.

        写法2: 语法糖.       语法糖 = 语法格式简化版.
            在定义原函数的时候, 加上 @装饰器名即可, 之后就正常调用 该原函数即可.
    细节:
        内部函数的形式, 必须和 原函数(要被装饰的函数) 形式一致, 即: 要有参就都有参, 要有返回值就都有返回值.
        例如: 原函数(要被装饰的函数) 是 无参无返回值的, 那么 装饰器的内部函数 也必须是 无参无返回值的.
"""
# 需求: 发表评论前, 需要先登录.



# 1. 定义装饰器(闭包函数), 用于对 指定函数的功能 做增强.
def check_login(fn_name):
    # 定义内部函数, 用于对 指定的 fn_name 函数功能做增强.
    def inner():                    # 有嵌套
        print('登陆中...登陆成功...')  # 额外功能
        fn_name()                   # 有引用
    return inner                    # 有返回

# 2. 定义原函数(要被装饰的函数), 表示: 发表评论.
@check_login
def comment():
    print('发表评论...')

# 在main函数中, 测试调用.
if __name__ == '__main__':
    # 3. 调用comment()函数.
    # comment()

    # 4. 演示装饰器写法1: 变量名 = 装饰器名(要被装饰的原函数名)
    # comment = check_login(comment)      # comment 就是 装饰后的 函数.
    # comment()

    # 5. 演示装饰器写法2: 在定义原函数的时候, 加上 @装饰器名 即可, 之后就正常调用 该原函数即可.
    comment()