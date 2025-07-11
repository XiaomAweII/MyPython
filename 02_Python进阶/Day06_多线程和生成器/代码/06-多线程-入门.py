"""
多任务 相关介绍:
    概述:
        之间我们写的多任务的代码都是 单线程的, 即: 从上往下执行, 如果前边没有执行完毕, 后续的代码不会被执行.
        为了提高多任务的执行效率, 我们可以用 多进程 或者 多线程的思维来解决.
    多任务的实现方式:
        1. 多进程.
        2. 多线程.

    回顾: 之前学的 进程的 概念.
        进程:
            CPU分配资源的基本单位(给微信多少资源, 给QQ多少资源), 进程 = 可执行程序, 例如: *.exe
            每个进程都至少有 1个 线程.
        线程:
            CPU调度资源的最小单位, 也是 进程的执行路径(执行单元).
            大白话: 进程 = 车,  线程 = 车道

    多线程的实现步骤:
        1. 导包
            import threading
        2. 创建线程对象, 关联函数.
            t1 = threading.Thread(target=目标函数)
        3. 开启线程.
            t1.start()
"""

# 需求: 使用多线程模拟一边写代码, 一边听音乐.

# 导包
import threading, time


# 1. 定义函数, 模拟: 写代码
def coding():
    for i in range(10):
        print('正在敲代码...')
        time.sleep(0.1)


# 2. 定义函数, 模拟: 听音乐.
def music():
    for i in range(10):
        print('正在听音乐.......')
        time.sleep(0.1)


# main函数中测试
if __name__ == '__main__':
    # 3. 创建线程对象, 关联上述的: 两个函数.
    t1 = threading.Thread(target=coding)
    t2 = threading.Thread(target=music)

    # 4. 启动线程.
    t1.start()
    t2.start()
