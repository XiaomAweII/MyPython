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
