"""
案例: 演示 线程共享 全局变量.

细节:
    1. 进程之间数据是相互隔离的, (同1个进程的)线程之间 数据是可以共享的.
    2. 多线程, 并发, 操作同一数据, 有可能引发安全问题, 需要用到 线程同步 来解决.
"""

# 需求: 定义1个全局变量, my_list = [], 创建两个子线程分别给列表添加元素, 及从列表中提取元素. 请用所学, 模拟实现该需求.
# 导包
import threading, time

# 1. 定义全局变量 my_list
my_list = []

# 2. 核心代码: 多线程,并发,操作同一数据, 有可能引发安全问题, 需要通过 线程同步 思路来解决.

# 3. 定义函数, write_data(), 往其中添加数据.
def write_data():
    for i in range(1, 66):
        my_list.append(i)
        print(f'add: {i}')
    print(f'write_data函数: {my_list}')

# 4. 定义函数, read_data(), 从中读取数据.
def read_data():
    # 为了让效果更明显, 加入休眠线程.
    time.sleep(3)
    print(f'read_data函数: {my_list}')

# main方法
if __name__ == '__main__':
    # 5. 创建两个线程对象, 分别关联上述的两个函数.
    t1 = threading.Thread(target=write_data)
    t2 = threading.Thread(target=read_data)

    # 6. 启动线程.
    t1.start()
    t2.start()

