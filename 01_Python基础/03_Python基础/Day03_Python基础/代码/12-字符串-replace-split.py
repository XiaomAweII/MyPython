"""
字符串常用函数介绍:
    概述:
        你可以简单的把函数先理解为就是 别人写好的"工具", 我们直接调用就可以了, 传入 参数(原材料), 经过函数加工, 可以获取指定的 结果(返回值).
    常用函数:
        字符串变量名.replace(旧子串, 新子串, 替换次数)      用新串 来 替换 旧串, 可以设置替换几个(不写,   默认就替换所有)    返回新串.
        字符串变量名.split(切割符, 切割个数)               按照切割符切割字符串, 切割个数表示切几个.   返回: 列表.
"""
# 1. 定义字符串变量s1, 记录要操作的数据.
s1 = 'hello and python and sql and linux'


# 2. 演示 replace()函数.
s2 = s1.replace('and', 'or')            # 不写个数, 替换所有
s3 = s1.replace('and', 'or', 2) # 写了个数, 写几个替换几个.

# 3. 打印结果:
print(f's1: {s1}')      # s1是不可变类型, 内容不变.
print(f's2: {s2}')
print(f's3: {s3}')
print('-' * 28)


# 4. 演示 split()函数.
list1 = s1.split('and')
print(type(list1))      # <class 'list'> 列表, 也是容器类型的一种.
print(list1)            # ['hello ', ' python ', ' sql ', ' linux']


list2 = s1.split('and', 2)
print(type(list2))      # <class 'list'> 列表, 也是容器类型的一种.
print(list2)            # ['hello ', ' python ', ' sql and linux']
print('-' * 28)


# 扩展: len()函数, 获取字符串的长度的, 即: 字符串有几个字符.
print(len(s1))          # 34
print('-' * 28)


# 分隔符.join(字符串)  用分隔符隔开 字符串中的每个字符.
s4 = 'hello'
s5 = ','.join(s4)
print(s5)       # h,e,l,l,o


# 需求: 把 'hello' => ['h', 'e', 'l', 'l', 'o']
list3 = ','.join(s4).split(',')
print(list3)        # ['h', 'e', 'l', 'l', 'o']
