"""
多进程(process):
    可以指定每个进程的任务, 多个进程之间可以并发, 也可以并行执行.

多进程实现步骤:
    1. 导包.
        import multiprocessing
    2. 创建进程对象, 关联: 要执行的任务(函数).
        p1 = multiprocessing.Process(target=目标函数名)
    3. 开启进程.
        p1.start()
"""

# 需求: 使用多进程来模拟一边写代码, 一边听音乐的功能.

# 导包
import time
import multiprocessing

# 1. 定义函数, 表示: 写代码.
def coding():
    for i in range(10):
        print(f'敲代码...{i}')
        time.sleep(0.2)     # 休眠0.2秒

# 2. 定义函数, 表示: 听音乐.
def music():
    for i in range(10):
        print(f'听音乐.......{i}', end='\n')
        time.sleep(0.2)     # 休眠0.2秒

# 在main函数中测试
if __name__ == '__main__':
    # 3. 创建进程对象, 分别关联 上述的两个函数.
    p1 = multiprocessing.Process(target=coding)
    p2 = multiprocessing.Process(target=music)

    # 4. 开启进程
    p1.start()
    p2.start()

    # 5. 来个main函数的 for循环.
    for i in range(5):
        print(f'main---------{i}')
        time.sleep(0.1)


