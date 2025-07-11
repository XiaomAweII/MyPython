"""
单进程:
    我们目前写的代码都是单进程的, 即: 前边的代码没有执行结束, 后边的代码不会被执行.
"""

# 导包
import time

# 1. 定义函数, 表示: 写代码.
def coding():
    for i in range(10):
        print(f'敲代码...{i}')
        time.sleep(0.2)     # 休眠0.2秒

# 2. 定义函数, 表示: 听音乐.
def music():
    for i in range(10):
        print(f'听音乐.......{i}')
        time.sleep(0.2)     # 休眠0.2秒

# 在main函数中测试
if __name__ == '__main__':
    # 调用函数.
    coding()
    music()