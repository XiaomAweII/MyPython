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
