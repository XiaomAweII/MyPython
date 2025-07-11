"""
案例: 演示 带参数的 多进程代码.

进程涉及到的参数如下:
    target      关联的是 当前进程要执行的函数.
    name        设置当前进程的名字
    args        以元组的形式, 给 当前进程关联的 函数 传参.
    kwargs      以字典的形式, 给 当前进程关联的 函数 传参.

细节:
    1. args方式传参, 实参的个数 和 数据类型, 顺序 必须和 进程关联的函数的形参列表 一致.
    2. kwargs方式传参, 实参的个数 和 数据类型 必须和 进程关联的函数的形参列表 一致, 顺序无所谓.
    3. 进程的默认命名规则是: Process-编号, 编号从1开始, 例如: Process-1, Process-2...
"""

# 需求: 使用多进程模拟小明一边编写num行代码, 一边听 count首音乐功能实现.

# 导包
import multiprocessing
import time

# 1. 定义函数, 表示: 敲代码.
def coding(name, num):
    for i in range(num):
        print(f'{name} 正在敲第 {i} 行代码.')
        time.sleep(0.1)

# 2. 定义函数, 表示: 听音乐.
def music(name, count):
    for i in range(count):
        print(f'{name} 正在听第 {i} 首音乐.......')
        time.sleep(0.1)

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
