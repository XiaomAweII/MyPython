"""
字符串常用函数介绍:
    概述:
        你可以简单的把函数先理解为就是 别人写好的"工具", 我们直接调用就可以了, 传入 参数(原材料), 经过函数加工, 可以获取指定的 结果(返回值).
    常用函数:
        字符串变量名.find(子串, 起始索引, 结束索引)
            找子串在字符串中 第1次 出现的位置, 如果写开始和结束索引(包左不包右), 就在指定区间查找, 找不到就返回 -1
        字符串变量名.index(子串, 起始索引, 结束索引)
            效果同上, 只不过是, 找不到就 报错.

        字符串变量名.rfind(子串, 起始索引, 结束索引)
            效果类似于 find(), 只不过是找 子串在字符串中 最后1次出现的位置, 找不到就返回 -1.
        字符串变量名.rindex(子串, 起始索引, 结束索引)
            类似于 rfind(), 找不到就 报错.

    例如:
        字符串: hello and python and java and sql and scala
        索引:  0     6          17       26      34
"""

# 1. 定义字符串.
s1 = "hello and python and java and sql and scala"


# 2. 演示上述的函数.
print(s1.find('and'))                               # 6, 因为字符串由多个字符组成, 默认会返回 字符串第1个字符的 索引.
print(s1.find('and', 7, 30))      # 17
print(s1.find('and', 7, 19))      # -1, 包左不包右, 取不到索引19
print('-' * 28)

print(s1.index('and'))                               # 6, 因为字符串由多个字符组成, 默认会返回 字符串第1个字符的 索引.
print(s1.index('and', 7, 30))      # 17
# print(s1.index('and', 7, 19))      # 报错
print('-' * 28)


# 演示 rfind() 和 rindex(), 效果都一样, 区别就是 找不到的时候报不报错.
print(s1.rfind('and'))      # 34
print(s1.rfind('and', 10, 30))      # 26
print(s1.rfind('and', 10, 19))      # -1
print('-' * 28)

print(s1.rindex('and'))      # 34
print(s1.rindex('and', 10, 30))      # 26
# print(s1.rindex('and', 10, 19))      # 找不到, 就报错


