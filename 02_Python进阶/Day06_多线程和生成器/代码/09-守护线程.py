"""
案例: 演示 默认情况下, 主线程会等待子线程结束再结束.

问: 如何实现 主线程结束, 子线程也同步结束呢?
答:
    设置 子线程为 守护线程即可.
方式:
    1. 通过 创建线程的时候,  daemon属性实现.
    2. 通过 线程对象名.setDaemon() 函数实现.
"""

# 需求: 创建1个子线程, 执行完大概需要2秒.  而主线程执行完需要1秒. 实现该需求, 观察结果.
# 导包
import threading, time

# 1. 定义函数work(), 表示: 子进程关联的函数.
def work():
    for i in range(10):
        print(f'work {i}...')
        time.sleep(0.2)  # 总执行时长: 0.2秒 * 10 = 2秒


# 在main方法中测试.
if __name__ == '__main__':
    # 2. 创建(子)线程, 关联: work()函数, 执行需要 2秒.
    # 方式1: daemon属性实现
    # t1 = threading.Thread(target=work, daemon=True)

    # 方式2: 通过 setDaemon()函数实现.
    t1 = threading.Thread(target=work)
    t1.setDaemon(True)

    # 3. 启动 子进程.
    t1.start()

    # 4. 休眠1秒, 表示: 主进程执行需要 1 秒.
    time.sleep(1)
    # 5. 打印: 主进程执行结束.
    print('main线程(主线程)执行结束!')
