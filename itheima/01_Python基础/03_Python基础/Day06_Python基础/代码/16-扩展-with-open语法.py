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