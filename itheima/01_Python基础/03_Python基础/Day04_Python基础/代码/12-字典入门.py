"""
字典介绍:
    概述:
        它属于容器类型的一种, 存储的是 键值对数据, 它(字典)属于 可变类型.
    定义格式:
        dict1 = {键:值, 键:值......}
        dict2 = {}
        dict3 = dict()
    细节:
        1. 字典用 大括号 包裹.
        2. 字典存储的市 键值对形式的数据, 冒号左边的叫: 键, 右边的叫: 值.
        3. 键具有唯一性, 值可以重复.
"""

# 1. 定义变量, 表示: 字典.
dict1 = {'杨过':'小龙女', '郭靖':'黄蓉', '张无忌':'赵敏'}
dict2 = {}
dict3 = dict()

# 2. 打印字典内容.
print(f'dict1: {dict1}')
print(f'dict2: {dict2}')
print(f'dict3: {dict3}')

# 3. 打印字典的数据类型.
print(type(dict1))
print(type(dict2))
print(type(dict3))