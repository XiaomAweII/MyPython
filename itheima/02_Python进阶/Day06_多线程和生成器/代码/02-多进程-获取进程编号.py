"""
案例: 演示 如何获取 进程的编号.

细节:
    1. 1个进程拥有1个唯一的 进程id, 当该进程被关闭的时候, 进程id也会同步释放. 即: 进程id是可以重复使用的.
    2. 知道了进程id, 就可以锁定到唯一的进程, 方便我们管理和维护, 以及梳理 子进程 和 父进程之间的关系.
    3. 获取当前进程的id, 有两种方式:
        方式1: os模块的 getpid() 函数.
        方式2: multiprocessing模块的 pid属性.
    4. 获取当前进程的 父id, 方式如下:
        os模块的 getppid()函数,     parent Process, 父进程.
"""

# 需求: 使用多进程模拟小明一边编写num行代码, 一边听 count首音乐功能实现.

# 导包
import multiprocessing
import time
import os

# 1. 定义函数, 表示: 敲代码.
def coding(name, num):
    for i in range(num):
        print(f'{name} 正在敲第 {i} 行代码.')
        time.sleep(0.1)
        # 打印当前进程的pid          current_process(), 获取当前正在执行的进程对象.
        print(f'p1进程的id为: {os.getpid()}, {multiprocessing.current_process().pid}, 它的父进程id为: {os.getppid()}')

# 2. 定义函数, 表示: 听音乐.
def music(name, count):
    for i in range(count):
        print(f'{name} 正在听第 {i} 首音乐.......')
        time.sleep(0.1)
        print(f'p2进程的id为: {os.getpid()}, {multiprocessing.current_process().pid}, 它的父进程id为: {os.getppid()}')

# 在main方法中测试
if __name__ == '__main__':
    # 3. 创建两个进程对象, 分别关联: 上述的两个函数.
    # args方式传参, 实参的个数 和 数据类型, 顺序 必须和 进程关联的函数的形参列表 一致.
    p1 = multiprocessing.Process(target=coding, name='刘亦菲', args=('小明', 10))
    #  kwargs方式传参, 实参的个数 和 数据类型 必须和 进程关联的函数的形参列表 一致, 顺序无所谓.
    p2 = multiprocessing.Process(target=music, name='胡歌', kwargs={'count': 7, 'name': '小王'})

    # 4. 打印进程的名字.
    print(f'p1进程的名字: {p1.name}')  # 刘亦菲
    print(f'p2进程的名字: {p2.name}')  # 胡歌

    # 5. 启动进程, 观察结果.
    p1.start()
    p2.start()

    # 6. 打印 main 进程的id
    print(f'main进程的id为: {os.getpid()}, {multiprocessing.current_process().pid}, 它的父进程id为: {os.getppid()}')

