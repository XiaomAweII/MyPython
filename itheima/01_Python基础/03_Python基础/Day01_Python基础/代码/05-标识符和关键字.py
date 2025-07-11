"""
标识符解释:
    概述:
        就是用来给 类, 函数, 变量等起名字的规则 和 规范.
    命名规则:
        1. 必须有英文字母, 数字, 下划线组成, 且 数字不能开头.
        2. 区分大小写.
        3. 最好做到见名知意, 虽然这个是规范, 但是你要当做规则用.
        4. 不能和关键字重名.
    常用的命名规范:
        大驼峰命名法, 也叫: 双峰驼命名法
            要求:
                每个单词的首字母都大写, 其它全部小写.
            例如:
                HelloWorld, MaxValue, ZhangSanAge...
        小驼峰命名法, 也叫: 单峰驼命名法
            要求:
                从第2个单词开始, 每个单词的首字母都大写, 其它全部小写.
            例如:
                helloWorld, maxValue, zhangSanAge...
        蛇形命名法,
            要求:
                单词间用 下划线 隔开.
            例如:
                MAX_VALUE, max_value, Zhang_San_Age...

        串行命名法,  Python不支持.
            要求:
                单词间用 中划线 隔开.
            例如:
                MAX-VALUE, max-value, Zhang-San-Age...

关键字:
    概述:
        被python赋予了特殊含义的单词.
    特点:
        常见的编辑器针对于关键字都会 高亮 显示.

    常见的关键字如下:
        'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
        'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally',
        'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal',
        'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'
"""

# 1. 演示: 不符合 见名知意 规范的 变量名.
age = '张三'
print(age)

# 2. 演示 Python中的关键字.
import keyword      # 导包, 后续讲解.
print(keyword.kwlist)   # key word list: 关键字列表


MaxValue = 100
minValue = 10
middle_value = 50

print(minValue, middle_value, MaxValue)