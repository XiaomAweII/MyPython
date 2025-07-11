#### 今日内容大纲介绍

* 生成器
  * yield关键字
  * 生成批次数据
* property关键字的用法
  * 修饰方法
  * 修饰类变量
* 正则表达式
  * 正则表达式规则
  * match()
  * search()
  * compile()
* 数据结构和算法入门
  * 时间复杂度公式
  * 时间复杂度效率

---

#### 1.生成器入门

```python
"""
生成器介绍:
    概述:
        它指的是 generator, 类似于以前学过的: 列表推导式, 集合推导式, 字典推导式...
    作用:
        降低资源消耗, 快速(批量)生成数据.
    实现方式:
        1. 推导式写法.
        2. yield写法.
    问题: 如何从生成器对象中获取数据?
    答案:
        1. for循环遍历
        2. next()函数, 逐个获取.
"""

# 案例1: 回顾之前的列表推导式, 集合推导式.
# 需求: 生成 1 ~ 5 的数据.
my_list = [i for i in range(1, 6)]
print(my_list, type(my_list))   # [1, 2, 3, 4, 5] <class 'list'>

my_set = {i for i in range(1, 6)}
print(my_set, type(my_set))     # {1, 2, 3, 4, 5} <class 'set'>


# 案例2: 演示 生成器写法1, 推导式写法
# 尝试写一下, "元组"推导式, 发现打印的结果不是元组, 而是对象, 因为这种写法叫: 生成器.
my_tuple = (i for i in range(1, 6))

print(my_tuple)             # <generator object <genexpr> at 0x0000024C90F056D0>    生成器对象
print(type(my_tuple))       # <class 'generator'>       生成器类型
print('-' * 31)

# 案例3: 如何从生成器对象中获取数据呢?
# 1. 定义生成器, 获取 1 ~ 5的数字.
my_generator = (i for i in range(1, 6))

# 2. 从生成器中获取数据.
# 格式1: for循环遍历
for i in my_generator:
    print(i)

# 格式2: next()函数, 逐个获取.
print(next(my_generator))       # 1
print(next(my_generator))       # 2
```



#### 2.yield关键字

```python
# 案例: 演示 yield关键字方式, 获取生成器.

# 需求: 自定义 get_generator()函数, 获取 包括: 1 ~ 5之间的整数 生成器.
# 1. 定义函数.
def get_generator():
    """
    用于演示 yield关键字的用法
    :return: 生成器对象.
    """
    # 思路1: 自定义列表, 添加指定元素, 并返回.
    # my_list = []
    # for i in range(1, 6):
    #     my_list.append(i)
    # return my_list

    # 思路2: yield写法, 即: 如下的代码, 效果同上.
    for i in range(1, 6):
        yield i     # yield会记录每个生成的数据, 然后逐个的放到生成器对象中, 最终返回生成器对象.


# 在main中测试.
if __name__ == '__main__':
    # 2. 调用函数, 获取生成器对象.
    my_generator = get_generator()

    # 3. 从生成器中获取每个元素.
    print(next(my_generator))   # 1
    print(next(my_generator))   # 2

    print('-' * 31)

    # 4. 遍历, 获取每个元素.
    for i in my_generator:
        print(i)
```



#### 3.生成批次数据

```python
"""
案例: 用生成器生成批次数据, 给后续的AI课程做铺垫, 因为在模型训练中, 数据都是分批次来 "喂" 的.

需求: 读取项目下的  jaychou_lyrics.txt文件(其中有5000多条 歌词数据), 按照8个 / 批次, 获取生成器, 并从中获取数据.
"""

import math

# 需求1: 铺垫知识,  math.ceil(数字):  获取指定数字的天花板数(向上取整), 即: 比这个数字大的所有整数中, 最小的哪个整数.
# print(math.ceil(5.1))       # 6
# print(math.ceil(5.6))       # 6
# print(math.ceil(5.0))       # 5



# 需求2: 获取生成器对象, 从文件中读数据数据, n条 / 批次
# 1. 定义函数 dataset_loader(batch_size), 表示: 数据生成器, 按照 batch_size条 分批.
def dataset_loader(batch_size):     # 假设: batch_size = 8
    """
    该函数用于获取生成器对象, 每条数据都是一批次的数据. 即: 生成器(8条, 8条, 8条...)
    :param batch_size: 每批次有多少条数据
    :return: 返回生成器对象.
    """
    # 1.1 读取文件, 获取到每条(每行)数据.
    with open("./jaychou_lyrics.txt", 'r', encoding='utf-8') as f:
        # 一次读取所有行, 每行封装成字符串, 整体放到列表中.
        data_lines = f.readlines()      # 结果: [第一行, 第二行, 第三行...]

    # 1.2 根据上述的数据, 计算出: 数据的总条数(总行数), 假设: 100行(条)
    line_count = len(data_lines)

    # 1.3 基于上述的总条数 和 batch_size(每批次的条数), 获取: 批次总数(即: 总共多少批)
    batch_count = math.ceil(line_count / batch_size)        # 例如: math.ceil(100 / 8) = 13

    # 1.4 具体的获取每批次数据的动作, 用 yield包裹, 放到生成器中, 并最终返回生成器(对象)即可.
    for i in range(batch_count):        # batch_count的值: 13,  i的值: 0, 1, 2, 3, 4, 5, .... 12
        # 1.5 yield会记录每批次数据, 封装到生成器中, 并返回(生成器对象)
        """
            推理过程:
                i = 0, 代表第1批次数据, 想要 第 1 条   ~~~~  第 8 条数据,    即:  data_lines[0:8]      
                i = 1, 代表第2批次数据, 想要 第 9 条   ~~~~  第 16 条数据,   即:  data_lines[8:16]      
                i = 2, 代表第3批次数据, 想要 第 17 条   ~~~~  第 24 条数据,  即:  data_lines[16:24]
                ......      
        """
        yield data_lines[i * batch_size: i * batch_size + batch_size]

# 在main中, 测试调用
if __name__ == '__main__':
    # 2. 获取生成器对象.
    my_generator = dataset_loader(13)

    # 3. 从生成器中获取第 1 批数据.
    # print(next(my_generator))
    # # 从第一批次中, 获取具体的每一条数据.
    # for line in next(my_generator):
    #     print(line, end='')
    #
    # print('-' * 31)
    #
    # # 从第二批次中, 获取具体的每一条数据.
    # for line in next(my_generator):
    #     print(line, end='')
    # print('-' * 31)


    # 4. 查看具体的每一批数据.
    for batch_data in my_generator:
        print(batch_data)
```

#### 4.property-充当装饰器用法

```python
"""
案例: 演示property关键字的 装饰器用法.

property解释:
    概述:
        它表示属性的意思, 可以用来修饰 方法, 修饰之后, 实现: 把 方法 当做 变量来使用.
    目的:
        简化开发.
    用法:
        格式1: 当做装饰器使用.
        格式2: 修饰类变量

    property充当装饰器用法, 格式:
        @property           # 修饰的是 获取值的函数.
        @方法名.setter       # 修饰的市 设置值的函数, 注意: 这里的方法名 要和 @property修饰的方法名保持一致.

    修饰类变量:
        类变量名 = property(get_xxx函数, set_xxx函数), 即: 参1是获取值的, 参2是设置值,  之后就可以直接使用该 类变量了.
"""

# 需求1: 定义Student类, 其中有个私有的属性 name, 定义公共的访问方式, 实现在外界访问该 私有变量.
# 1. 定义学生类.
class Student(object):
    # 1.1 定义私有变量(属性)
    def __init__(self):
        self.__name = '乔峰'      # 两个下划线_ 表示私有

    # # 1.2 定义get_xxx(), 获取: 姓名
    # @property
    # def get_name(self):
    #     return self.__name
    #
    # # 1.3 定义set_xxx(), 修改: 姓名
    # @get_name.setter
    # def set_name(self, name):
    #     self.__name = name

    # 1.2 定义get_xxx(), 获取: 姓名
    @property
    def name(self):
        return self.__name

    # 1.3 定义set_xxx(), 修改: 姓名
    @name.setter
    def name(self, name):
       self.__name = name


# 2. 在main函数中测试
if __name__ == '__main__':
    # 场景1: 以前学面向对象时的 普通写法.
    # # 2.1 创建学生类对象.
    # s = Student()
    # # 2.2 打印 name 属性值
    # print(s.get_name())
    # # 2.3 设置 name 属性值
    # s.set_name('萧峰')
    # # 2.4 打印 name 属性值
    # print(s.get_name())

    # 场景2: 演示: property充当装饰器后的用法.
    # # 2.1 创建学生类对象.
    # s = Student()
    # # 2.2 打印 name 属性值
    # print(s.get_name)
    # # 2.3 设置 name 属性值
    # s.set_name = '萧峰'
    # # 2.4 打印 name 属性值
    # print(s.get_name)

    # 场景3: 演示: property充当装饰器后的用法, 最终写法
    # 2.1 创建学生类对象.
    s = Student()
    # 2.2 打印 name 属性值
    print(s.name)
    # 2.3 设置 name 属性值
    s.name = '夯哥'
    # 2.4 打印 name 属性值
    print(s.name)
```

#### 5.property-修饰类变量

```python
"""
案例: 演示property关键字的 修饰类属性的用法.

property解释:
    概述:
        它表示属性的意思, 可以用来修饰 方法, 修饰之后, 实现: 把 方法 当做 变量来使用.
    目的:
        简化开发.
    用法:
        格式1: 当做装饰器使用.
        格式2: 修饰类变量

    property充当装饰器用法, 格式:
        @property           # 修饰的是 获取值的函数.
        @方法名.setter       # 修饰的市 设置值的函数, 注意: 这里的方法名 要和 @property修饰的方法名保持一致.

    修饰类变量:
        类变量名 = property(get_xxx函数, set_xxx函数), 即: 参1是获取值的, 参2是设置值,  之后就可以直接使用该 类变量了.
"""

# 需求1: 定义Student类, 其中有个私有的属性 name, 定义公共的访问方式, 实现在外界访问该 私有变量.
# 1. 定义学生类.
class Student(object):
    # 1.1 定义私有变量(属性)
    def __init__(self):
        self.__name = '乔峰'      # 两个下划线_ 表示私有

    # 1.2 定义get_xxx(), 获取: 姓名
    def get_name(self):
        return self.__name

    # 1.3 定义set_xxx(), 修改: 姓名
    def set_name(self, name):
       self.__name = name

    # 1.4. 通过property关键字, 实现: 把 get_name() 和 set_name()函数, 封装成 类属性.
    # 格式: 类变量名 = property(get_xxx函数, set_xxx函数), 即: 参1是获取值的, 参2是设置值,  之后就可以直接使用该 类变量了.
    name = property(get_name, set_name)

# 2. 在main函数中测试
if __name__ == '__main__':
    # 演示: property修饰 类属性的用法.
    # 2.1 创建学生类对象.
    s = Student()
    # 2.2 打印 name 属性值
    print(s.name)
    # 2.3 设置 name 属性值
    s.name = '夯哥'
    # 2.4 打印 name 属性值
    print(s.name)       # s1.name
```

#### 6.正则表达式介绍

```python
"""
正则表达式介绍:
    概述:
        全称是 Regular Expression, 正则表达式, 即: 正确的, 符合特定规则的式子.
    作用:
        校验, 匹配数据的.
    细节:
        1. 学正则就是学正则表达式的 规则, 不要背, 因为这么多年了, 校验邮箱, 校验手机...等一些列的规则前辈们都写出来, 网上一搜一堆.
        2. 我讲正则的目的: 能用我们学的规则, 看懂别人写的 (正则表达式)式子, 且会根据需求修改即可.
        3. 正则不独属于任意的一门语言, Java, Python...都支持, 且: 正则规则都是一样的, 不同的是 写法不一样.

    Python中 正则的使用步骤:
        1. 导包
            import re
        2. 正则校验.
            re.match()
            re.search()
            re.compile().sub()
        3. 获取匹配结果.
            result = re.group()
    我们要学习的正则规则如下:
        .
        \.
        [abc]
        [^abc]
        \d
        \D
        \s
        \S
        \w
        \W

        ^
        $

        ?
        +
        *
        {n}
        {n,}
        {n,m}

        |
        ()
        \num

        扩展:
            (?P<分组名>)       给分组起名字
            (?P=分组名)        使用指定分组的内容
"""
```

#### 7.正则表达式-match和search函数

```python
"""
正则表达式介绍:
    概述:
        全称是 Regular Expression, 正则表达式, 即: 正确的, 符合特定规则的式子.
    作用:
        校验, 匹配数据的.
    细节:
        1. 学正则就是学正则表达式的 规则, 不要背, 因为这么多年了, 校验邮箱, 校验手机...等一些列的规则前辈们都写出来, 网上一搜一堆.
        2. 我讲正则的目的: 能用我们学的规则, 看懂别人写的 (正则表达式)式子, 且会根据需求修改即可.
        3. 正则不独属于任意的一门语言, Java, Python...都支持, 且: 正则规则都是一样的, 不同的是 写法不一样.

    Python中 正则的使用步骤:
        1. 导包
            import re
        2. 正则校验.
            re.match(pattern=正则表达式, str, flag)       参1: 正则表达式,  参2: 要校验的字符串, 参3:可选项, 例如: 忽略大小写, 多行模式等.
            re.search(pattern=正则表达式, str, flag)
            re.compile(正则表达式).sub(用来替换的内容, 要被替换的内容)
        3. 获取匹配结果.
            result = re.group()
    上述函数 介绍:
        match:  匹配的意思, 从左往右, 逐个字符进行匹配, 不会跳过任意的1个字符, 要求: 全部匹配才行.
        search: 查找的意思, 从左往右, 从任意的某个字符开始, 只要能匹配上即可.
        compile: 用于替换的.

    我们要学习的正则规则如下:
        .           代表: 任意的1个字符
        \.          取消.的特殊含义, 就是一个普通的. 校验邮箱的时候用, 例如: zhangsan@163.com
        a           代表: 就是1个普通的字符a
        [abc]
        [^abc]
        \d
        \D
        \s
        \S
        \w
        \W

        ^
        $

        ?
        +
        *           数量词, 代表前边的内容, 至少出现 0次, 至多出现 n次
        {n}
        {n,}
        {n,m}

        |
        ()
        \num

        扩展:
            (?P<分组名>)       给分组起名字
            (?P=分组名)        使用指定分组的内容
"""
import re

# 需求2: 校验字符串格式是否是  任意1个字符 + it + 任意的1个字符
# result = re.match('.it.', '1ita')     # .it. 意思是: 任意1个字符 + it + 任意的1个字符
# result = re.match('.it.', 'ita')     # .it. 意思是: 任意1个字符 + it + 任意的1个字符

# 需求2: 校验字符串是否包含 it
# match: 从左往右, 逐个字符进行匹配, 不会跳过任意的1个字符, 要求: 全部匹配才行.
# result = re.match('.*it.*', 'sdit123sf')
# result = re.match('.*it.*', 'it123sf')
# result = re.match('.*it.*', 'sdit')

# it.*   it + 后续任意内容.
# result = re.match('it.*', 'sdit123sf'

# search: 查找的意思, 从左往右, 从任意的某个字符开始, 只要能匹配上即可.
result = re.search('it.*', 'sdit 123sf')


# 打印结果
if result:          # 只要result不是None, 就会走这里.
    # print(result)   # 匹配到的正则对象
    print(result.group())       # 从正则中, 获取具体 匹配到的内容.
else:
    # print(result)   # None
    print('未匹配!')
```

#### 8.compile函数

```python
"""
案例: 演示正则 替换.

涉及到的 re 模块下的函数:
    写法1:
        re.compile(正则表达式).sub(用来替换的内容, 要被替换的内容)

    写法2: 即, 上述格式的语法糖.
        re.sub(正则表达式, 用来替换的内容, 要被替换的内容)

     回顾: 字符串中的replace()函数, 也可以替换, 但是是全词匹配, 不支持正则.
        字符串.replace(旧内容, 新内容)
"""
import re

# 扩展: r'字符串'   取消字符串中\的转移的含义, 就是1个普通的\
# print('d:\\aa\\bb\\cc')     # d:\aa\bb\cc,  \在python中有特殊含义(转移符), 两个\表示一个\
# print(r'd:\\aa\\bb\\cc')    # d:\\aa\\bb\\cc
# print(r'd:\aa\bb\cc')    # d:\\aa\\bb\\cc

# 需求1: 把字符串中的"敏感词" 给 替换成 *
old_str = "车主说: 你的刹车片应该换了啊, 嘿嘿"

# 1. 自定义正则规则.
p = r'啊|阿|嘿|呵|哈|啦|嘻|桀'        # pattern: 模板(可以理解为: 正则规则),  字符串前加r表示, 取消转移.

# 2. 基于正则规则, 获取正则对象.
res = re.compile(pattern=p)     # pattern=具体的正则字符串

# 3. 对上述的字符串进行替换.
result = res.sub('A', old_str)  # 返回的是: 字符串.

# 4. 打印匹配结果.
if result:
    print(f'匹配到: {result}')
else:
    print('未匹配!')

print('-' * 31)

# 需求2: 上述代码的简化版, 语法糖实现
old_str = "故人西辞黄鹤楼, 烟花三月黄鹤楼, 中华虽然好抽, 但是不要上瘾, 抽烟只抽煊赫门, 一生只爱一个人!"
# 参1: 正则表达式.
# 参2: 用来替换的字符串.
# 参3: 要被替换的字符串.
result = re.sub('黄鹤楼|中华|煊赫门', '*', old_str)
print(result)
print('-' * 31)

# 需求3: 回顾字符串的 replace()
s1 = "烟花三月黄鹤楼, 黄鹤楼虽然好抽, 大中华, 抽着黄鹤楼!"

# 参1: 旧字符串(要被替换的)  参2: 新字符串(用来替换的)  参3: 替换几个, 不写就替换所有.
result = s1.replace('黄鹤楼', '*', 2)
# result = s1.replace('黄鹤楼|中华', '*', 2)  # 无效, 不支持正则.
print(result)
```

#### 9.正则-匹配单个字符

```python
"""
正则表达式介绍:
    概述:
        全称是 Regular Expression, 正则表达式, 即: 正确的, 符合特定规则的式子.
    作用:
        校验, 匹配数据的.
    细节:
        1. 学正则就是学正则表达式的 规则, 不要背, 因为这么多年了, 校验邮箱, 校验手机...等一些列的规则前辈们都写出来, 网上一搜一堆.
        2. 我讲正则的目的: 能用我们学的规则, 看懂别人写的 (正则表达式)式子, 且会根据需求修改即可.
        3. 正则不独属于任意的一门语言, Java, Python...都支持, 且: 正则规则都是一样的, 不同的是 写法不一样.

    Python中 正则的使用步骤:
        1. 导包
            import re
        2. 正则校验.
            re.match(pattern=正则表达式, str, flag)       参1: 正则表达式,  参2: 要校验的字符串, 参3:可选项, 例如: 忽略大小写, 多行模式等.
            re.search(pattern=正则表达式, str, flag)
            re.compile(正则表达式).sub(用来替换的内容, 要被替换的内容)
        3. 获取匹配结果.
            result = re.group()
    上述函数 介绍:
        match:  匹配的意思, 从左往右, 逐个字符进行匹配, 不会跳过任意的1个字符, 要求: 全部匹配才行.
        search: 查找的意思, 从左往右, 从任意的某个字符开始, 只要能匹配上即可.
        compile: 用于替换的.

    我们要学习的正则规则如下:
        .           代表: 任意的1个字符
        \.          取消.的特殊含义, 就是一个普通的. 校验邮箱的时候用, 例如: zhangsan@163.com
        a           代表: 就是1个普通的字符a
        [abc]       代表: a,b,c其中的任意1个字符
        [^abc]      代表: 除了a,b,c外, 任意的1个字符
        \d          代表: 所有的数字, 等价于 [0-9]
        \D          代表: 所有的非数字, 等价于 [^0-9]
        \s          代表: 空白字符, \n, 空格, \t等...
        \S          代表: 非空白字符, 即: 上述取反.
        \w          代表: 非特殊字符, 即: 字母, 数字, _ 下划线, 汉字
        \W          代表: 特殊字符, 即: 上述取反.

        ^
        $

        ?
        +
        *           数量词, 代表前边的内容, 至少出现 0次, 至多出现 n次
        {n}
        {n,}
        {n,m}

        |           或者的意思.
        ()
        \num

        扩展:
            (?P<分组名>)       给分组起名字
            (?P=分组名)        使用指定分组的内容
"""


# 导包
import re

# 需求: 获取字符串中 以数字开头的内容, 后续是啥无所谓.
# 演示: .           代表: 任意的1个字符(除了\n)
# result = re.match('it.', 'itA')
# result = re.match('it.', 'it\t')
# result = re.match('it.', 'it\n')    # 不匹配
# result = re.match('it.', 'i t1')    # 不匹配

# 演示: \.          取消.的特殊含义, 就是一个普通的. 校验邮箱的时候用, 例如: zhangsan@163.com
# 演示: a           代表: 就是1个普通的字符a
# result = re.match('.it\.', ' ait.')      # .it\.  任意1个字符 + it + .
# result = re.match('.it\.', ' aitb')      # 未匹配
# result = re.match('.it\.', ' ait.123')   # 未匹配

# 演示: [abc]       代表: a,b,c其中的任意1个字符
# result = re.match('[abc].*', 'asafs')
# result = re.match('[abc].*', 'bsafs')
# result = re.match('[abc].*', 'c123')
# result = re.match('[abc].*', 'd123')    # 未匹配

# 演示: [^abc]      代表: 除了a,b,c外, 任意的1个字符
# result = re.match('[^abc].*', 'asafs')  # 未匹配
# result = re.match('[^abc].*', 'bsafs')  # 未匹配
# result = re.match('[^abc].*', 'c123')   # 未匹配
# result = re.match('[^abc].*', 'd123')   # d123

# 演示: \d          代表: 所有的数字, 等价于 [0-9]
# result = re.match('\d.*', '1abc') # True
# result = re.match('\d.*', 'a1bc') # False

# 演示: \D          代表: 所有的非数字, 等价于 [^0-9]
# result = re.match('\D.*', '1abc')   # False
# result = re.match('\D.*', 'a1bc')   # True

# 演示: \s          代表: 空白字符, \n, 空格, \t等...
# result = re.match('it\s', 'it ')
# result = re.match('it\s', 'it\t')
# result = re.match('it\s', 'it\n')
# result = re.match('it\s', 'ita')

# 演示: \S          代表: 非空白字符, 即: 上述取反.
# result = re.match('it\S', 'it ')    # False, 未匹配
# result = re.match('it\S', 'it\t')   # False, 未匹配
# result = re.match('it\S', 'it\n')   # False, 未匹配
# result = re.match('it\S', 'ita')    # True

# 演示: \w          代表: 非特殊字符, 即: 字母, 数字, _ 下划线, 汉字
# result = re.match('it\w', 'it好')
# result = re.match('it\w', 'it_')
# result = re.match('it\w', 'it1')
# result = re.match('it\w', 'itxa')
# result = re.match('it\w', 'it+')    # 未匹配

# 演示: \W          代表: 特殊字符, 即: 上述取反.
result = re.match('it\W', 'it好')   # 未匹配
result = re.match('it\W', 'it_')    # 未匹配
result = re.match('it\W', 'it1')    # 未匹配
result = re.match('it\W', 'itxa')   # 未匹配
result = re.match('it\W', 'it+')


# 打印匹配结果
if result:
    info = result.group()       # 获取匹配到的内容
    print(f'匹配到: {info}')
else:
    print('未匹配!')

# 上述格式的: 简化版.
print(f'匹配到: {result.group()}' if result else '未匹配!')
```

#### 10.正则-匹配多个字符

```python
"""
正则表达式介绍:
    概述:
        全称是 Regular Expression, 正则表达式, 即: 正确的, 符合特定规则的式子.
    作用:
        校验, 匹配数据的.
    细节:
        1. 学正则就是学正则表达式的 规则, 不要背, 因为这么多年了, 校验邮箱, 校验手机...等一些列的规则前辈们都写出来, 网上一搜一堆.
        2. 我讲正则的目的: 能用我们学的规则, 看懂别人写的 (正则表达式)式子, 且会根据需求修改即可.
        3. 正则不独属于任意的一门语言, Java, Python...都支持, 且: 正则规则都是一样的, 不同的是 写法不一样.

    Python中 正则的使用步骤:
        1. 导包
            import re
        2. 正则校验.
            re.match(pattern=正则表达式, str, flag)       参1: 正则表达式,  参2: 要校验的字符串, 参3:可选项, 例如: 忽略大小写, 多行模式等.
            re.search(pattern=正则表达式, str, flag)
            re.compile(正则表达式).sub(用来替换的内容, 要被替换的内容)
        3. 获取匹配结果.
            result = re.group()
    上述函数 介绍:
        match:  匹配的意思, 从左往右, 逐个字符进行匹配, 不会跳过任意的1个字符, 要求: 全部匹配才行.
        search: 查找的意思, 从左往右, 从任意的某个字符开始, 只要能匹配上即可.
        compile: 用于替换的.

    我们要学习的正则规则如下:
        .           代表: 任意的1个字符
        \.          取消.的特殊含义, 就是一个普通的. 校验邮箱的时候用, 例如: zhangsan@163.com
        a           代表: 就是1个普通的字符a
        [abc]       代表: a,b,c其中的任意1个字符
        [^abc]      代表: 除了a,b,c外, 任意的1个字符
        \d          代表: 所有的数字, 等价于 [0-9]
        \D          代表: 所有的非数字, 等价于 [^0-9]
        \s          代表: 空白字符, \n, 空格, \t等...
        \S          代表: 非空白字符, 即: 上述取反.
        \w          代表: 非特殊字符, 即: 字母, 数字, _ 下划线, 汉字
        \W          代表: 特殊字符, 即: 上述取反.

        ^
        $

        ?           数量词, 至少0次,至多1次
        +           数量词, 至少1次, 至多n次
        *           数量词, 代表前边的内容, 至少出现 0次, 至多出现 n次
        {n}         恰好n次, 多一次少一次都不行.
        {n,}        至少n次, 至多无所谓
        {n,m}       至少n次, 至多m次, 包括n和m

        |           或者的意思.
        ()
        \num

        扩展:
            (?P<分组名>)       给分组起名字
            (?P=分组名)        使用指定分组的内容
"""

import re

# 演示正则数量词, ?           数量词, 至少0次,至多1次
# result = re.match('it.?', 'ita')        # .?   任意的0个或者1个字符
# result = re.match('it.?', 'it\n')       # .?   任意的0个或者1个字符
# result = re.match('it.?', 'i t\n')      # 未匹配

# 演示正则数量词, +           数量词, 至少1次, 至多n次
result = re.match('it.+', 'it\n')       # 未匹配
result = re.match('it.+', 'it中')       # .+ 代表至少1个 至多任意个 任意的字符
result = re.match('it.+', 'it中, 真中, 非常中!')       # .+ 代表至少1个 至多任意个 任意的字符

result = re.match('it[abc]+', 'itaaabbc')   # [abc]+ 任意的1个或者多个 a,b,c组成的字符串
result = re.match('it[abc]+', 'itaaaa')   # [abc]+ 任意的1个或者多个 a,b,c组成的字符串
result = re.match('it[abc]+', 'it1aaaa')   # 未匹配


# 演示正则数量词, *           数量词, 代表前边的内容, 至少出现 0次, 至多出现 n次
result = re.match('it[abc]*', 'itaaabbc')   # [abc]+ 任意的1个或者多个 a,b,c组成的字符串
result = re.match('it[abc]*', 'itaaaa')   # [abc]+ 任意的1个或者多个 a,b,c组成的字符串
result = re.match('it[abc]*', 'i t1aaaa')   # 未匹配

# 演示正则数量词, {n}         恰好n次, 多一次少一次都不行.
# 前边恰好3个数字, 后续是啥无所谓.
result = re.match('[0-9][0-9][0-9].*', '623abc!@#')
result = re.match('[0-9]{3}.*', '623abc!@#')     # 效果同上
result = re.match('\d{3}.*', '623abc!@#')        # 效果同上
result = re.match('\d{3}.*', '6239abc!@#')        # 效果同上

# 演示正则数量词, {n,}        至少n次, 至多无所谓
# 前边至少2个整数, 后续是啥无所谓.
result = re.match('\d{2,}.*', '62123abc!@#')
result = re.match('\d{2,}.*', '6!@#')       # 未匹配

# 演示正则数量词, {n,m}       至少n次, 至多m次, 包括n和m
# 前边至少2个, 至多5个 整数或者字母或者下划线, 后续是啥无所谓.
result = re.match('[0-9a-zA-Z_]{2,5}.*', '62abc!@#')
result = re.match('[0-9a-zA-Z_]{2,5}.*', '__你12!@#')
result = re.match('[0-9a-zA-Z_]{2,5}.*', '6!2abc@#')     # 未匹配
result = re.match('[0-9a-zA-Z_]{2,5}.*', '_你12!@#')     # 未匹配


# 打印结果
print(f'匹配到: {result.group()}' if result else '未匹配!')
```

#### 11.正则-校验开头和结尾

```python
"""
正则表达式介绍:
    概述:
        全称是 Regular Expression, 正则表达式, 即: 正确的, 符合特定规则的式子.
    作用:
        校验, 匹配数据的.
    细节:
        1. 学正则就是学正则表达式的 规则, 不要背, 因为这么多年了, 校验邮箱, 校验手机...等一些列的规则前辈们都写出来, 网上一搜一堆.
        2. 我讲正则的目的: 能用我们学的规则, 看懂别人写的 (正则表达式)式子, 且会根据需求修改即可.
        3. 正则不独属于任意的一门语言, Java, Python...都支持, 且: 正则规则都是一样的, 不同的是 写法不一样.

    Python中 正则的使用步骤:
        1. 导包
            import re
        2. 正则校验.
            re.match(pattern=正则表达式, str, flag)       参1: 正则表达式,  参2: 要校验的字符串, 参3:可选项, 例如: 忽略大小写, 多行模式等.
            re.search(pattern=正则表达式, str, flag)
            re.compile(正则表达式).sub(用来替换的内容, 要被替换的内容)
        3. 获取匹配结果.
            result = re.group()
    上述函数 介绍:
        match:  匹配的意思, 从左往右, 逐个字符进行匹配, 不会跳过任意的1个字符, 要求: 全部匹配才行.
        search: 查找的意思, 从左往右, 从任意的某个字符开始, 只要能匹配上即可.
        compile: 用于替换的.

    我们要学习的正则规则如下:
        .           代表: 任意的1个字符
        \.          取消.的特殊含义, 就是一个普通的. 校验邮箱的时候用, 例如: zhangsan@163.com
        a           代表: 就是1个普通的字符a
        [abc]       代表: a,b,c其中的任意1个字符
        [^abc]      代表: 除了a,b,c外, 任意的1个字符
        \d          代表: 所有的数字, 等价于 [0-9]
        \D          代表: 所有的非数字, 等价于 [^0-9]
        \s          代表: 空白字符, \n, 空格, \t等...
        \S          代表: 非空白字符, 即: 上述取反.
        \w          代表: 非特殊字符, 即: 字母, 数字, _ 下划线, 汉字
        \W          代表: 特殊字符, 即: 上述取反.

        ^           代表: 开头      '^[^abc].*'
        $           代表: 结尾

        ?           数量词, 至少0次,至多1次
        +           数量词, 至少1次, 至多n次
        *           数量词, 代表前边的内容, 至少出现 0次, 至多出现 n次
        {n}         恰好n次, 多一次少一次都不行.
        {n,}        至少n次, 至多无所谓
        {n,m}       至少n次, 至多m次, 包括n和m

        |           或者的意思.
        ()
        \num

        扩展:
            (?P<分组名>)       给分组起名字
            (?P=分组名)        使用指定分组的内容
"""

import re

# 演示  ^           代表: 开头
result = re.match('\d.*', '1abc')         # 必须以数字开头
result = re.search('\d.*', 'a1bc')       # 必须以数字开头
result = re.search('^\d.*', 'a1bc')       # 未匹配

# 演示  $           代表: 结尾
result = re.match('.*[a-zA-Z]', '123abc')
result = re.match('.*[a-zA-Z]', '123abc1')
result = re.match('.*[a-zA-Z]$', '123abc1')  # 未匹配



# 打印结果
print(f'匹配到: {result.group()}' if result else '未匹配!')
```

#### 12.正则-或者

```python
"""
正则规则:
    |           或者的意思.
    ()
    \num

    扩展:
        (?P<分组名>)       给分组起名字
        (?P=分组名)        使用指定分组的内容
"""
import re

# 需求: 演示正则规则之 |

# 1. 定义列表, 记录水果.
fruit = ['apple', 'banana', 'orange', 'pear']

# 2. 遍历, 获取到每一种水果.
for value in fruit:
    # 3. 判断当前水果是否是喜欢吃的(苹果, 梨)水果, 并打印.
    result = re.match('apple|pear', value)
    # 4. 打印结果.
    if result:
        # print(f'喜欢吃: {value}')
        print(f'喜欢吃: {result.group()}')
    else:
        print(f'不喜欢吃: {value}')
```

#### 13.正则-校验邮箱

```python
# 案例: 演示正则校验邮箱.

"""
正则规则:
    |           或者的意思.
    ()          代表分组.
    \num

    扩展:
        (?P<分组名>)       给分组起名字
        (?P=分组名)        使用指定分组的内容
"""
import re

# 需求: 匹配出163, 126, qq等邮箱, 格式为:  4 ~ 20位任意字母, 数字, _  + @标记符 + 域名163,126,qq + .com

# 校验邮箱, 格式为:  4 ~ 20位任意字母, 数字, _  + @标记符 + 域名163,126,qq + . + 后缀(任意2~3个字符)
result = re.match('[a-zA-Z0-9_]{4,20}@(163|126|qq)\.com', 'hello@163.com')

# 打印结果
if result:
    print(f'匹配到: {result.group()}')         # hello@163.com
    print(f'group(0), 即: 0组, 结果为: {result.group(0)}')   # 同上, hello@163.com
    print(f'group(0), 即: 0组, 结果为: {result.group(1)}')   # 同上, 163
else:
    print('未匹配到!')

```

#### 14.正则-获取指定分组内容

```python
# 案例: 演示正则-获取指定分组内容.

"""
正则规则:
    |           或者的意思.
    ()          代表分组.
    \num

    扩展:
        (?P<分组名>)       给分组起名字
        (?P=分组名)        使用指定分组的内容
"""
import re

# 需求: 从 qq:qq号 这个格式的字符串中, 提取出qq 和 qq号

result = re.match('(qq):(\d{5,11})', 'qq:12306')


if result:
    print(f'匹配到: {result.group()}')       # qq:12306
    print(f'group(0), {result.group(0)}')   # qq:12306
    print(f'group(1), {result.group(1)}')   # qq
    print(f'group(2), {result.group(2)}')   # 12306
else:
    print('未匹配到!')

```

#### 15.正则-校验html标签

```python
"""
案例: 演示正则校验 html标签.

正则规则:
    |           或者的意思.
    ()          代表分组.
    \num        \分组编号, 表示引入某组的内容.

    扩展:
        (?P<分组名>)       给分组起名字
        (?P=分组名)        使用指定分组的内容
"""
import re

# 需求1: 演示正则校验html标签, 单级标签.
# 方式1: 直接编写, 复制同样内容即可.
result = re.match('<[a-zA-Z]{1,4}>.*</[a-zA-Z]{1,4}>', '<html>hh</html>')

# 方式2: 上述方式, 存在相同的正则规则, 我们可以用 分组的思维来优化它.
# 细节: \在python中有转义的意思, 所以 两个\表示一个\, 要么写 \\, 要么用 r'' 取消转义.
result = re.match('<([a-zA-Z]{1,4})>.*</\\1>', '<html>hh</html>')
result = re.match(r'<([a-zA-Z]{1,4})>.*</\1>', '<html>hh</html>')


# 需求2: 演示正则校验html标签, 多级标签.
#                              分组1            分组2
result = re.match(r'<([a-zA-Z]{1,4})><(h[1-6])>.*</\2></\1>', '<html><h3>hh</h3></html>')

# 扩展, 给分组起别名 及 使用指定的分组.
#  (?P<分组名>)       给分组起名字
#  (?P=分组名)        使用指定分组的内容
result = re.match(r'<(?P<A>[a-zA-Z]{1,4})><(?P<B>h[1-6])>.*</(?P=B)></(?P=A)>', '<html><h6>hh</h6></html>')

# 需求: 演示正则校验html标签.
if result:
    print(f'匹配到: {result.group()}')       # qq:12306
else:
    print('未匹配到!')
```

