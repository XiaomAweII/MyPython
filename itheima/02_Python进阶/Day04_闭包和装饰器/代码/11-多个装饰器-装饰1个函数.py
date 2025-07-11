"""
多个装饰器 装饰 1个函数, 细节如下:
    1. 多个装饰器 装饰 1个函数, 装饰的顺序是 由内向外的.     即: 传统写法.
    2. 但是多个装饰器的执行顺序是, 由上往下的.              即: 语法糖写法.
"""

# 需求: 发表评论前, 需要 先登陆用户, 再进行 验证码验证.  在不改变原有函数基础上, 对功能做增强.

# 1. 定义装饰器, 加入: 登录的功能.
def check_user(fn_name):
    def inner():            # 有嵌套
        print('登录中...')   # 有额外功能
        fn_name()           # 有引用
    return inner            # 有返回


# 2. 定义装饰器, 加入: 校验 验证码 的功能.
def check_code(fn_name):
    def inner():                # 有嵌套
        print('校验验证码...')    # 有额外功能
        fn_name()               # 有引用
    return inner                # 有返回


# 3. 定义函数, 表示: 原函数(即: 要被装饰的函数), 即: 发表评论.
# @check_user
# @check_code
def comment():
    print('发表评论!')

# 在main函数中测试.
if __name__ == '__main__':
    # 4. 写法1: 装饰器的 传统写法         结论: 由内向外装饰.
    # 细节1: 多个装饰器 装饰 1个函数, 装饰的顺序是 由内向外的.     即: 传统写法.
    # 目前, 函数的关系, 由内向外分别是: comment() -> check_code() -> check_user()
    cc = check_code(comment)
    comment = check_user(cc)
    comment()
    print('-' * 20)

    # 5. 写法2: 装饰器的 语法糖写法.       结论: 由上往下执行.
    # 写法2: 但是多个装饰器的执行顺序是, 由上往下的.              即: 语法糖写法.
    comment()
