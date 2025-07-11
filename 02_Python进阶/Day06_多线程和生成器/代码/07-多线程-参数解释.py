"""
案例: 演示 带参数的 多线程代码.

进程涉及到的参数如下:
    target      关联的是 当前进程要执行的函数.
    name        设置当前进程的名字
    args        以元组的形式, 给 当前进程关联的 函数 传参.
    kwargs      以字典的形式, 给 当前进程关联的 函数 传参.

细节:
    1. args方式传参, 实参的个数 和 数据类型, 顺序 必须和 进程关联的函数的形参列表 一致.
    2. kwargs方式传参, 实参的个数 和 数据类型 必须和 进程关联的函数的形参列表 一致, 顺序无所谓.
    3. 线程的默认命名规则是: Thread-编号, 编号从1开始, 例如: Thread-1, Thread-2...
"""

# 需求: 使用多进程模拟小明一边编写num行代码, 一边听 count首音乐功能实现.

# 导包
import threading, time

# 1. 定义函数, 模拟: 写代码
def coding(name, num):
    for i in range(num):
        print(f'{name} 正在敲第 {i} 行代码...')
        time.sleep(0.1)


# 2. 定义函数, 模拟: 听音乐.
def music(name, count):
    for i in range(count):
        print(f'{name}正在听第 {i} 首音乐.......')
        time.sleep(0.1)


# main函数中测试
if __name__ == '__main__':
    # 3. 创建线程对象, 关联上述的: 两个函数.
    t1 = threading.Thread(target=coding, name='乔峰', args=('小明', 10))
    print(f't1线程的名字: {t1.name}')

    t2 = threading.Thread(target=music, name='阿朱', kwargs={'count':5, 'name':'小王'})
    print(f't2线程的名字: {t2.name}')

    # 4. 启动线程.
    t1.start()
    t2.start()
