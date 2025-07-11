#### 今日内容大纲介绍

* 函数参数详解
  * 针对于实参
    * 位置参数
    * 关键字参数
  * 针对于形参
    * 缺省(默认)参数
    * 不定长参数
      * *args -> 接收数据, 封装成: 元组.
      * **kwargs -> 接收数据, 封装成: 字典
* 组包和拆包
* 扩展: 交换变量值
* 可变类型 和 不可变类型介绍
* 文件相关操作
  * os模块(Operating System)
  * 读文件数据
  * 写数据到文件
  * 扩展: with...open()...语句

---

#### 1.函数同时返回多个值

```python
"""
案例: 演示函数同时返回多个值.

需求: 自定义函数 calculate(), 接收两个整数, 然后返回它们的 加减乘除结果.
"""

# 1. 定义函数calculate(), 接收两个整数.
def calculate(a, b):
    """
    自定义函数, 模拟计算器, 计算两个整数的 加减乘除结果.
    :param a: 要操作的第1个整数
    :param b: 要操作的第2个整数
    :return: 加减乘除结果
    """
    # 具体的计算加减乘除结果的动作
    sum = a + b
    sub = a - b
    mul = a * b
    div = a // b
    # 2. 返回它们的 加减乘除结果.
    return sum, sub, mul, div       # 同时返回多个值, 默认会用 元组封装.
    # return [sum, sub, mul, div]       # 当然也可以手动把多个值封装成 列表或者集合, 然后返回.
   

# 3. 调用函数, 进行测试.
if __name__ == '__main__':
    result = calculate(10, 3)
    print(result)           # (13, 7, 30, 3)
    print(type(result))     # <class 'tuple'>
```



#### 2.函数参数-位置参数

```python
"""
Python中 函数的参数写法 主要有如下的四种:
    位置参数
    关键字参数
    默认参数(缺省参数)
    不定长参数

细节:
    1. 位置参数 和 关键字参数 是针对于 实参 来讲的.
    2. 缺省参数 和 不定长参数 是针对于 形参 来讲的.

位置参数:
    我们之前写的实参写法 就是位置参数, 即: 实参的 个数 和 顺序 必须和 形参保持一致.
"""

# 1. 定义函数 user_info(), 接收三个参数, 打印自己的信息: name, age, address
def user_info(name, age, address):
    print(f'我叫: {name}, 今年 {age} 岁了, 我住在: {address}')


# main函数是程序的 主入口, 所有的代码都是从这里开始执行的.
if __name__ == '__main__':
    # 2. 调用函数 user_info()
    user_info('张三', 23, '北京')
    user_info('张三', '北京', 23)   # 不报错, 但是结果不是我们要的.

```



#### 3.函数参数-关键字参数

```python
"""
Python中 函数的参数写法 主要有如下的四种:
    位置参数
    关键字参数
    默认参数(缺省参数)
    不定长参数

细节:
    1. 位置参数 和 关键字参数 是针对于 实参 来讲的.
    2. 缺省参数 和 不定长参数 是针对于 形参 来讲的.

关键字参数:
    我们之前写的实参写法 就是位置参数, 即: 实参的 个数 和 顺序 必须和 形参保持一致.
    如果 实参 和 形参的顺序不一致, 结果不一定是我们想要的, 所以为了更灵活的调用函数, 引入了: 关键字参数.

    格式:
        采用 键 = 值 的形式来传递 实参.
    细节:
        如果有多个参数(实参), 则: 位置参数在前, 关键字参数在后(且多个关键字参数之间, 没有顺序要求)
"""


# 1. 定义函数 user_info(), 接收三个参数, 打印自己的信息: name, age, address
def user_info(name, age, address):
    print(f'我叫: {name}, 今年 {age} 岁了, 我住在: {address}')


# main函数是程序的 主入口, 所有的代码都是从这里开始执行的.
if __name__ == '__main__':
    # 2. 调用函数 user_info()
    # 位置参数
    user_info('张三', 23, '北京')
    user_info('张三', '北京', 23)  # 位置参数, 故意传错位置. 不报错, 但是结果不是我们要的.
    print('-' * 28)

    # 关键字参数, 多个关键字参数之间无顺序要求.
    user_info(address='广州', name='李四', age=24)

    # 如果既有关键字参数, 又有位置参数. 则: 位置参数在前, 关键字参数在后.
    # user_info(address='广州', name='李四', 25)      # 报错, 顺序不对.

    #           位置参数              关键字参数
    user_info('王五', address='深圳', age=25)

```



#### 4.函数参数-缺省参数

```python
"""
Python中 函数的参数写法 主要有如下的四种:
    位置参数
    关键字参数
    默认参数(缺省参数)
    不定长参数

细节:
    1. 位置参数 和 关键字参数 是针对于 实参 来讲的.
    2. 缺省参数 和 不定长参数 是针对于 形参 来讲的.

缺省参数:
    概述:
        缺省参数也叫 默认参数, 即: 定义在 函数的形参列表的 最后.

    格式:
        采用 键 = 值 的形式来定义, 且必须放 形参列表的最后.
    细节:
        1. 调用函数时, 没有给缺省参数赋值, 则: 用缺省参数的默认值.
        2. 调用函数式, 如果给缺省参数赋值, 则: 用赋的新值.
"""


# 1. 定义函数 user_info(), 接收三个参数, 打印自己的信息: name, age, address
# def user_info(name, address='北京', age):     # 报错, 缺省参数必须在 形参列表的最后.
def user_info(name, age, address='三亚'):
    print(f'我叫: {name}, 今年 {age} 岁了, 我住在: {address}')

# main函数是程序的 主入口, 所有的代码都是从这里开始执行的.
if __name__ == '__main__':
    # 2. 调用函数 user_info()
    # 位置参数
    user_info('张三', 23, '北京')
    user_info('张三', '北京', 23)  # 位置参数, 故意传错位置. 不报错, 但是结果不是我们要的.
    print('-' * 28)

    # 关键字参数, 多个关键字参数之间无顺序要求.
    user_info(address='广州', name='李四', age=24)

    # 如果既有关键字参数, 又有位置参数. 则: 位置参数在前, 关键字参数在后.
    # user_info(address='广州', name='李四', 25)      # 报错, 顺序不对.

    #           位置参数              关键字参数
    user_info('王五', address='深圳', age=25)
    print('-' * 28)

    # 不给默认参数传值.
    user_info('王五', age=25)

```



#### 5.函数参数-不定长参数

```python
"""
Python中 函数的参数写法 主要有如下的四种:
    位置参数
    关键字参数
    默认参数(缺省参数)
    不定长参数

细节:
    1. 位置参数 和 关键字参数 是针对于 实参 来讲的.
    2. 缺省参数 和 不定长参数 是针对于 形参 来讲的.

不定长参数:
    概述:
        不定长参数也叫 可变参数, 即: 参数的个数是可以变化的.
    应用场景:
        适用于 实参的个数不确定的情况, 就可以把 形参定义成 可变参数.
    格式:
        *args           只能接收所有的 位置参数, 封装到: 元组中.
        **kwargs        只能接收所有的 关键字参数, 封装到: 字典中.
    细节:
        1. 关于实参, 位置参数在前, 关键字参数在后.
        2. 关于形参, 如果两种 可变参数都有, 则: *args 在前, **kwargs 在后.
        3. 关于形参, 如果既有 缺省参数 又有不定长参数, 则编写顺序为:  *args, 缺省参数, **kwargs

"""

# 需求1: 演示 不定长参数(可变参数)之 接收 位置参数.
def method01(*args):                      # 约定俗成, 变量名可以任意写, 但是建议写成 args
    print(f'接收到的所有参数为: {args}')     # 你把 args变量当做 元组来用即可.
    print(type(args))

# 需求2: 演示 不定长参数(可变参数)之 接收 关键字参数.
def method02(**kwargs):
    print(f'接收到的所有参数为: {kwargs}')
    print(type(kwargs))

# 需求3: 同时定义两种 参数.
#             不定长(可变)参数
def method03(*args, **kwargs):
    print(f'args: {args}')
    print(f'kwargs: {kwargs}')

# 需求4: 同时定义 缺省参数, 不定长参数.
#          不定长参数   缺省参数    不定长参数
def method04(*args, name='张三', **kwargs):
    print(f'name: {name}')
    print(f'args: {args}')
    print(f'kwargs: {kwargs}')



# main充当程序的 主入口.
if __name__ == '__main__':
    # 调用 method01()函数.
    method01(1, '张三', 23)
    # method01(1, '张三', age=23)       # 报错, *args 只能接收所有的 位置参数.
    print('-' * 28)

    # 调用 method02()函数.
    method02(name='张三', age=23, phone='13112345678')
    # method02('李四', age=24, phone='13112345678')     # 报错, **kwarg 只能接收所有的 关键字参数.
    print('-' * 28)

    # 调用 method03()函数
    #                位置参数           关键字参数
    method03(10, 20, 'aa', name='王五', age=25, address='杭州')
    print('-' * 28)

    # 调用 method04()函数
    #                位置参数             关键字参数
    method04(10, 20, 'aa', name='王五', age=25, address='杭州')


```



#### 6.扩展-不定长参数的应用

```python
# 需求: 定义get_sum()函数, 分别实现计算求 任意个整数和, 例如: 2个整数和, 3个整数和, 4个整数和...

# 分析: 因为get_sum()具体要接受多少个实参(整数), 也不确定, 所以推荐使用: 可变参数.
def get_sum2(*args):
    # args: 接收所有的 位置参数, 并封装成: 元组.  这里你直接把 args当元组用即可.
    # 例如:  args = (10, 20, 30)
    # sum = 0
    # for i in args:
    #     sum += i
    # return sum

    # 扩展: 可以直接用 sum()函数, 计算元组的 元素和.
    return sum(args)


# 调用函数
if __name__ == '__main__':
    result1 = get_sum2(10, 20)
    print(f'result1: {result1}')

    result2 = get_sum2(10, 20, 30)
    print(f'result2: {result2}')

    result3 = get_sum2(10, 20, 30, 50)
    print(f'result3: {result3}')

    # 细节: 可变参数最少可以传入 0个参数, 至多可以传入 无数个参数.
    result4 = get_sum2()
    print(f'result4: {result4}')


```



#### 7.组包和拆包

```python
"""
组包和拆包解释:
    概述:
        组包 和 拆包 是Python中的一种独有写法.
    格式:
        把 多个值 => 1个变量 的过程, 称之为: 组包.
        把 1个(容器)变量 => 多个变量值 的过程, 称之为: 拆包.
    应用场景:
        1. 一次性获取到 元组, 列表, 字典中的每个数据.
        2. 交换变量.
"""

# 组包.   多 => 1
list1 = [11, 22, 33, 44, 55]
tuple1 = ('aa', 'bb', 'cc')
dict1 = {'name': '张三', 'age': 23}


# 拆包.
# 从 列表 拆包
a, b, c, d, e = list1
print(a, b, c, d, e)

# 从 元组 拆包
x, y, z = tuple1
print(x, y, z)

# 从 字典 拆包, 只能获取 键 的值.
k1, k2 = dict1
print(k1, k2)

```



#### 8.交换变量值

```python
# 需求1: 交换 字符串 变量值.
s1 = "刘亦菲"
s2 = "胡歌"

# 方式1: 采用 临时变量(第三方变量)
# tmp = s1
# s1 = s2
# s2 = tmp

# 方式2: 拆包
s1, s2 = s2, s1
print(f's1: {s1},  s2: {s2}')
print('-' * 31)

# 需求2: 交换 数字 变量值.
a = 10
b = 20

# 方式1: 采用 临时变量.
# tmp = a
# a = b
# b = tmp

# 方式2: 拆包
a, b = b, a

# 方式3: 算术运算符.
# a = a + b   # a = 30, b = 20
# b = a - b   # a = 30, b = 10
# a = a - b   # a = 20, b = 10

# 方式4: 位运算符.
# 规律: 1个数字 被 另一个数字 位移或两次, 该数字值不变.   10 ^ 5 ^ 5 = 10
# a = a ^ b   # a = 10 ^ 20,                      b = 20
# b = a ^ b   # a = 10 ^ 20,                      b = a ^ b = 10 ^ 20 ^ 20 = 10
# a = a ^ b   # a = a ^ b = 10 ^ 20 ^ 10 = 20,    b = 10

print(f'a: {a},  b: {b}')
```



#### 9.引用详解

```python
"""
引用 介绍:
    概述:
        Python中的引用 指的是 地址值,即: 变量在内存中的位置(地址).
    格式:
        id(变量名)         可以查看变量在内存中的地址.
    细节:
        1. Python中只有 引用传递, 即: 我们以前看到的所有赋值动作, 都是把 地址值拷贝(赋值)过去.
        2. 区分 可变 和 不可变类型的依据: 在不改变地址值的情况下, 是否可以修改变量的内容, 可以: 可变类型, 不可以: 不可变类型.
            可变:     列表, 字典, 集合.
            不可变:   int, float, bool, str, 元组
        3. 观察如下的代码, 分析程序结果:
            形参是可变类型:    形参的可变直接影响实参.
            形参是不可变类型:  形参的改变对实参没有任何影响.
"""

# 案例1: 查看变量的地址值.
a = 10
b = a
c = a

print(id(a))    # 140714022086720, 说明: 10在内存中就1份, abc分别指向它.
print(id(b))    # 140714022086720, 说明: 10在内存中就1份, abc分别指向它.
print(id(c))    # 140714022086720, 说明: 10在内存中就1份, abc分别指向它.
print('-' * 28)

# 案例2: 演示可变 和 不可变类型.
# 场景1: 可变类型, 即: 在不改变地址值的情况下, 可以修改里边的元素内容.
list1 = [1, 2, 3]
print(f'list1 修改前元素值: {list1}')
print(f'list1 修改前地址值: {id(list1)}')     # 2525712470976
# 修改list1的元素.
list1.append(100)
list1[1] = 200
print(f'list1 修改后元素值: {list1}')
print(f'list1 修改后地址值: {id(list1)}')     # 2525712470976
print('-' * 28)

# 场景2: 不可变类型, 即: 在不改变地址值的情况下, 不可以修改里边的元素内容.
s1 = 'abc'
print(f's1 修改前元素值: {s1}')
print(f's1 修改前地址值: {id(s1)}')     # 1374973319088
# 修改s1的元素.
s1 = 'xyz'
print(f's1 修改前元素值: {s1}')
print(f's1 修改前地址值: {id(s1)}')     # 1374973726448
print('-' * 28)
```



#### 10.扩展-引用相关面试题

```python
"""
观察如下的代码, 分析程序结果, 得到的结论如下.
    形参是可变类型:    形参的改变直接影响实参.
    形参是不可变类型:  形参的改变对实参没有任何影响.
"""

# 定义函数, 接收 参数, 一会传入: 整数(int), 不可变类型.
def change(num):
    num = 200

# 定义函数, 接收 参数, 一会传入: 列表(list), 可变类型.
def change2(list1):
    list1[1] = 28

if __name__ == '__main__':
    # 演示: 不可变类型 函数的调用.
    a = 100
    print(f'调用 change 函数前, a: {a}')     # 100
    change(a)
    print(f'调用 change 函数前, a: {a}')     # 100

    # 演示: 可变类型 函数的调用.
    list1 = [1, 2, 3, 4, 5]
    print(f'调用 change 函数前, list1: {list1}')  # 1, 2, 3, 4, 5
    change2(list1)
    print(f'调用 change 函数后, list1: {list1}')  # 1, 28, 3, 4, 5



```



#### 11-匿名函数

```python
"""
匿名函数介绍:
    概述:
        没有名字的函数 就叫 匿名函数.
    格式:
        变量名 = lambda 形参列表 : 函数体(只能写一行代码, 且该行代码的结果会被自动返回)
    细节:
        1. Python的匿名函数 类似于 Java中的Lambda表达式.
        2. 匿名函数适用于简单的业务需求, 即: 函数体只有一行代码的函数.
        3. 匿名函数的应用场景:
            当对方法仅调用一次.
            匿名函数 可以作为 函数对象 进行传递.
"""

# 案例1: 匿名函数入门.
# 需求: 定义函数, 用于计算两个整数和.
# 场景1: 普通函数.
def get_sum(a, b):
    return a + b

print(get_sum(10, 20))


# 场景2: 匿名函数实现
my_get_sum = lambda a, b :  a + b
print(my_get_sum(11, 22))
print('-' * 28)


# 案例2: 定义函数, 接收2个整数, 分别计算两个整数的: 和, 差, 积, 商, 最大值, 最小值.
# 场景1: 普通函数.
def get_sum(a, b):
    return a + b            # 求和

def get_sub(a, b):
    return a - b            # 差

def get_mul(a, b):
    return a * b            # 积

# ...... 依次类推, 把其他的函数全部做出来即可, 但是这样就会导致 多个需求, 每个需求都对应1个函数, 函数太多了, 一方面开发效率下降,
# 另一方面不方便维护和管理, 光记忆函数名 就是1个非常庞大的工程, 如何解决呢?
# 可以通过 匿名函数的方式 解决.


# 场景2: 匿名函数 方式实现.
def my_func(a, b, fn):
    """
    自定义函数, 根据传入的规则, 来计算两个整数的结果.
    :param a: 要操作的第1个数据
    :param b: 要操作的第2个整数
    :param fn: 具体的 计算规则 函数.
    :return: 具体的计算结果.
    """
    return fn(a, b)


# 调用 my_func()函数.
# 加
# print(my_func(10, 3, get_sum))
print(my_func(10, 3, lambda a, b : a + b))

# 减
# print(my_func(10, 3, get_sub))
print(my_func(10, 3, lambda a, b: a - b))

# 乘
print(my_func(10, 3, lambda a, b: a * b))

# 除
print(my_func(10, 3, lambda a, b: a // b))

# 最大值
# print(my_func(10, 3, lambda a, b: max(a, b)))
print(my_func(10, 3, lambda a, b: a if a >= b else b))

# 最小值
print(my_func(10, 3, lambda a, b: a if a <= b else b))

# 平均值
print(my_func(10, 3, lambda a, b: (a + b) // 2))



```



#### 12.文件操作-读

```python
"""
文件 介绍:
    概述:
        无论是 windows, Linux, Mac系统, 都是采用 文件 来管理数据的, 它们都是 文件管理系统.
        之所以用文件来管理数据, 原因是因为: 内存中的数据是临时存储的, 电脑管理了, 数据就丢失了.
        文件: 可以实现 永久 存储数据.
    文件的类型:
        文本文档, 图片类型, 视频类型, 音频类型......
    文件的操作步骤:
        1. 打开文件.
        2. 读写操作.
        3. 关闭文件.

    打开文件 涉及到的API(Application Programming Interface, 应用程序编程接口), 就是: 别人写的 函数.
        文件对象名 = open('文件路径', '打开模式', 码表)        # 参3为可选项, 针对于 中文有效.
    读取文件信息:
        read(num)       一次读取num个字节的数据, 不写就一次性读取所有的数据.
        readline()      一次读取一行.
        readlines()     一次性读取读完所有行, 且会把每行数据封装到 1个列表中.
    关闭文件:
        文件对象名.close()

细节:
    1. Python中, 路径的写法, 要么用 \\,  要么用 /, 要么用 r'一个\就行', 即: r'\' 会取消 \的转移含义, 当做1个普通字符来用.
    2. 相对路径默认是相对于 当前项目的路径来写的, 即: 你直接写 1.txt, 想到于是  /当前项目路径/1.txt
"""

# 1. 打开文件(file).
# 场景1: 绝对路径.
# f = open(r'D:\workspace\ai_28_basic_bj\pythonProject\day06\data\a.txt', 'r')           # r'' 取消 \的 转义函数.
# f = open('D:\\workspace\\ai_28_basic_bj\\pythonProject\\day06\\data\\a.txt', 'r')      # \\ 代表 1个 \
# f = open('D:/workspace/ai_28_basic_bj/pythonProject/day06/data/a.txt', 'r')              # 或者可以写成 /


# 场景2: 相对路径写法.  默认是相对于当前项目的路径来讲的
f = open('./data/a.txt', 'r')           # ./ 代表当前目录
# print(f'文件对象名: {f}')

# 2. 读写操作.
# read(num)       一次读取num个字节的数据, 不写就一次性读取所有的数据.
# print(f.read())      # 一次性读取所有的数据.
# print(f.read(3))     # 一次读取3个字节, 包括: \n 也占1个字节
# print(f.read(5))     # 一次读取5个字节, 包括: \n 也占1个字节

# readline()      一次读取一行.
# print(f.readline())     # 'abc\n'
# print(f.readline())     # 'defg\n'
# print(f.readline())     # 'xyz'
# print(f.readline())     # 空

# readlines()     一次性读取读完所有行, 且会把每行数据封装到 1个列表中.
print(f.readlines())    # ['abc\n', 'defg\n', 'xyz']

# 3. 关闭文件.
f.close()

```



#### 13.文件操作-读-中文

```python
"""
中文 解释:
    计算机底层存储, 操作, 运算数据, 都是采用数据的 二进制(补码)形式, 所以中文, 特殊符号, 数字, 英文字母底层都是要转成二进制的.
    后来科学家就提出了 码表的概念, 用来描述 字符 及其 对应的数字的关系.  例如: 'a' => 97,  'A' => 65, '0' => 48...
    最早的码表 ASCII码表记录的就是: 英文字母, 数字, 特殊符号及其对应的 数字的关系.
    后来随着计算机的普及, 各个国家都有了各个国家的 码表, 咱们国内使用最多的主要是 GBK系列, 1个中文 占 2个字节.
    后来有个组织就统计全世界各个国家的码表, 制定了一张"万国码", 也叫"统一码", 这就是: Unicode系列的码表, 例如: utf-8, utf-16, utf-32...

    总结:
        国内主要用 GBK码表, 1个中文占 2个 字节.
        国际通用码表 UTF-8, 1个中文占 3个字节.
        无论是什么码表, 英文字母, 数字, 特殊符号都只占1个字节.
        只要以后你遇到了乱码的情况, 不用想, 原因只会有1个: 编解码不一致.
"""

# 1. 打开文件.
# f = open('./data/a.txt', 'r')                 # r是字符形式读, 没写码表, 默认是按照: gbk 读.
# f = open('./data/a.txt', 'r', encoding='gbk')   # 效果同上
f = open('./data/a.txt', 'r', encoding='utf-8')   # 按照 utf-8 码表解析

# 以二进制形式来读
# f = open('./data/a.txt', 'rb')

# 2. 读取文件内容.
print(f.read())

# 3. 关闭文件.
f.close()
```



#### 14.文件操作-写

```python
"""
往文件中写信息:
    write(数据)       往文件中写数据.
    writelines()     一次写多行.

细节:
    1. 注意写数据的模式,  w: write,覆盖写入.  a: append,追加写入.
    2. 读的时候, 如果 数据源文件不存在, 会报错.  No Such File Or Directory...
    3. 写的时候, 如果 目的地文件不存在, 会自动创建.
"""

# 需求1: 演示 写数据到文件.
# # 1. 打开文件.
# # f = open('./data/b.txt', 'w', encoding='utf-8')     # 覆盖写入
# f = open('./data/b.txt', 'a', encoding='utf-8')     # 覆盖写入
#
# # 2. 往文件中写数据.
# f.write('hello world!\n')
# f.write('好好学习, 天天向上!')
#
# # 3. 关闭文件.
# f.close()


# 需求2: 演示 拷贝 a.txt文件数据 到 b.txt
# 1. 封装数据源文件, 用于: 读取它的数据.
src_f = open('./data/a.txt', 'r', encoding='utf-8')
# src_f = open('./data/a.txt', 'rb')      # 字节形式 读写的时候, 不能指定码表.

# 2. 封装目的地文件, 用于: 往其中写数据.
dest_f = open('./data/b.txt', 'w', encoding='utf-8')
# dest_f = open('./data/b.txt', 'wb')

# 方式1: 一次性读取所有, 并一次性写入所有.
# data = src_f.read()
# dest_f.write(data)

# 方式2: 一次性读取所有(行), 并一次性写入所有行.
# data = src_f.readlines()    # [行, 行, 行...]
# dest_f.writelines(data)

# 方式3: 循环读取, 读一行, 写一行.  或者 一次性读写指定的字节数, 一般是: 1024的整数倍.
while True:
    # 一次性读取 8192个字节, 8 * 1024 = 8192个字节 = 8KB
    data = src_f.read(8192)
    # 核心细节: 如果读不到数据了, 即读取到文件末尾了, 结束读取.
    # if len(data) == 0:        # 可以判断长度
    if data == '':              # 也可以判断是否为空
        break
    # 把读取到的数据, 写到目的地文件.
    dest_f.write(data)

# 3. 关闭文件.
src_f.close()
dest_f.close()

```



#### 15.综合案例-文件备份

```python
# 需求: 键盘录入 当前目录下任意的1个文件名, 然后对该文件进行备份, 备份文件名格式为: 原文件名[备份].原后缀名, 例如:  test.txt => test[备份].txt

# 1. 提示用户录入要备份的 文件名, 并接收.       绕口令.txt,  abc.mp3.txt,   .txt
old_name = input('请录入要备份的文件名: ')

# 2. 找到 最后1个 . 的索引.
index = old_name.rfind('.')

# 3. 判断文件名是否合法.
# 3.1 如果合法, 就拷贝.
if index > 0:
    # 4. 根据原文件名, 拼接 新文件名.           绕口令.txt  => 绕口令[备份].txt
    new_name = old_name[:index] + '[备份]' + old_name[index:]
    # print(new_name)

    # 5. 正常的读写操作.
    # 5.1 打开 数据源文件.
    # old_f = open(old_name, 'r', encoding='utf-8')   # 码表性质只能拷贝 纯文本文件.
    old_f = open(old_name, 'rb')        # 二进制形式读写, 无需指定码表, 通用版读写.
    # 5.2 打开 目的地文件.
    # new_f = open(new_name, 'w', encoding='utf-8')
    new_f = open(new_name, 'wb')
    # 5.3 循环 拷贝.
    while True:
        # 5.4 每次读取 8192个 字节的数据, 存储到 data变量中.
        data = old_f.read(8192)
        # 5.5 判断, 如果没有读取到内容, 说明文件内容读取完毕, 结束拷贝.
        if len(data) <= 0:
            break
        # 5.6 走到这里, 说明读取到内容, 将读取到的数据写出到 目的地文件中.
        new_f.write(data)
    # 5.7 释放资源.
    old_f.close()
    new_f.close()
    print('备份成功!')

else:
    # 3.2 如果不合法, 就提示, 然后程序结束.
    print('您录入的路径有误, 程序结束!')
```



#### 16-扩展-with-open语法

```python
"""
扩展: with-open语句:
    它主要是针对于 文件操作的, 即: 你再也不用手动 close()释放资源了, 该语句会在 语句体执行完毕后, 自动释放资源.

    格式:
        with open('路径', '模式', '码表') as 别名,  open('路径', '模式', '码表') as 别名:
            语句体

    特点:
        语句体执行结束后, with后边定义的变量, 会自动被释放.
"""

# 1. 打开 数据源 和 目的地文件.
with open('./data/a.txt', 'rb') as src_f, open('./data/b.txt', 'wb') as dest_f:
    # 2. 具体的 拷贝动作.
    # 2.1 循环拷贝.
    while True:
        # 2.2 一次读取8192个字节.
        data = src_f.read(8192)
        # 2.3 读完了, 就不读了.
        if len(data) <= 0:
        # if data == '':
            break
        # 2.4 走到这里, 说明读到了, 把读取到的数据写出到目的地文件.
        dest_f.write(data)


# 先了解, 明儿详解.
import os
print(os.getcwd())      # current work directory, 当前的工作空间.  我们写的 相对路径的 ./ 就是它的执行结果.
```

