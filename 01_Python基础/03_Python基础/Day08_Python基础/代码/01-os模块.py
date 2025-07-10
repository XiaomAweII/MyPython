"""
os 模块介绍:
    概述:
        全称叫: Operating System, 系统模块, 主要是操作 文件夹, 文件, 路径等的.
        属于第3方的包, 所以我们使用的时候需要导包.
    常用函数:
        getcwd()    获取当前的工作空间目录(即: 你写相对路径时, 参考的路径). current work directory: 当前工作目录
        chdir()     改变工作空间路径.  change directory
        rmdir()     删除文件夹, 必须是空文件夹. remove directory
        mkdir()     制作文件夹.   make directory
        rename()    改名, 文件名 或者 文件夹名均可.
        listdir()   获取指定目录下 所有的子级文件或者文件夹(注意: 不包括子级的子级)
        ......
"""

# 导包
import os

# 演示 os 模块的函数
# getcwd()    获取当前的工作空间目录(即: 你写相对路径时, 参考的路径). current work directory: 当前工作目录
print(os.getcwd())

# chdir()     改变工作空间路径.  change directory
# os.chdir("d:/")
# print(os.getcwd())

# mkdir()     制作文件夹.   make directory
# os.mkdir('aa')      # 创建aa文件夹, 如果存在就报错, 不存在就创建.

# rmdir()     删除文件夹, 必须是空文件夹. remove directory
# os.rmdir('aa')

# rename()    改名, 文件名 或者 文件夹名均可.
# os.rename('aa', 'bb')
# os.rename('1.txt', 'hg.txt')

# listdir()   获取指定目录下 所有的子级文件或者文件夹(注意: 不包括子级的子级)
# file_list = os.listdir('./')
file_list = os.listdir('d:/')
print(file_list)


# 读取文件数据.
# f = open('1.txt', 'r', encoding='utf-8')
# print(f.read())
# f.close()
