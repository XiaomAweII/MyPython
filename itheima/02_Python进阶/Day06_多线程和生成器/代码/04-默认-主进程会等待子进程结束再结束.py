"""
案例: 演示 默认情况下, 主进程会等待子进程结束再结束.

目的:
    引出下个知识点, 如何实现, 让主进程结束的时候, 它的子进程也同步结束.
"""

# 需求: 创建1个子进程, 执行完大概需要2秒.  而主进程执行完需要1秒. 实现该需求, 观察结果.
# 导包
import multiprocessing, time


# 1. 定义函数work(), 表示: 子进程关联的函数.
def work():
    for i in range(10):
        print(f'work {i}...')
        time.sleep(0.2)  # 总执行时长: 0.2秒 * 10 = 2秒


# 在main方法中测试.
if __name__ == '__main__':
    # 2. 创建(子)进程, 关联: work()函数, 执行需要 2秒.
    p1 = multiprocessing.Process(target=work)
    # 3. 启动 子进程.
    p1.start()

    # 4. 休眠1秒, 表示: 主进程执行需要 1 秒.
    time.sleep(1)
    # 5. 打印: 主进程执行结束.
    print('main进程(主进程)执行结束!')
