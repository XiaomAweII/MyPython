"""
正则表达式的规则:
    |           表示: 或者的意思.
    ()          表示: 括号内的内容是1个组
    \num        表示: 引用num分组的内容, 例如: \1 表示引用第1组的数据,  \2 表示引用第2组的数据

    扩展:
        (?P<分组名>)   设置分组
        (?P=分组名)    使用指定的分组
"""
import re

# 需求1: 列表中有一些水果, 喜欢吃: apple 和 pear, 请用正则验证, 下述的水果, 哪些是喜欢吃的, 哪些是不喜欢吃的.
# 1. 定义列表, 记录: 水果.
fruits = ['apple', 'banana', 'orange', 'pear']

# 2. 遍历, 获取到每一种水果.
for fruit in fruits:
    # fruit 就是列表中的 每一种水果, 校验即可.
    if re.match('apple|pear', fruit):
        # 进这里, 说明是: 喜欢吃的水果.
        print(f'喜欢吃: {fruit}')
    else:
        # 走这里, 说明 是不喜欢吃的水果.
        print(f'不喜欢吃: {fruit}')
print("-" * 20)


# 需求2: 匹配出 163, 126, qq等邮箱.
# 邮箱例子: liuli@163.com,   386021668@ .com
# 邮箱规则: 4~16位数字,字母,下划线  +   @标记符 +  邮箱域名 + .com 或者 .cn

# 1. 定义变量, 记录要被校验的邮箱.
email = '386021668@qq.com'
# email = 'liuli@163.com'
# email = 'liuli@163.xyz'

# 2. 具体的校验动作.
result = re.match('^[0-9a-zA-Z_]{4,16}@(163|126|qq)\.(com|cn)$', email) # 以左小括号为依据, 数数字, 是几就是第几组.

# 3. 打印匹配结果.
# if result != None:
if result:
    print(f'匹配到 所有的数据: {result.group()}')     # 获取所有匹配到的数据
    print(f'匹配到 第0组数据: {result.group(0)}')     # 获取匹配到的第0组数据, 效果同上, 即: 0组 = 原串
    print(f'匹配到 第1组数据: {result.group(1)}')     # 获取匹配到的第1组数据, 即: 域名 163
    print(f'匹配到 第2组数据: {result.group(2)}')     # 获取匹配到的第2组数据, 即: 后缀 com
else:
    print('未匹配到!')
print("-" * 20)
