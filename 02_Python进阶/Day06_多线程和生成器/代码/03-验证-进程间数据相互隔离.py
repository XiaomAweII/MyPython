"""
案例: 演示 进程之间 数据是相互隔离的.

细节:
    1. 进程之间数据是相互隔离的, 不能共享.
    2. 子进程相当于是父进程的副本, 即: 把父进程的内容会拷贝一遍, 单独执行.  注意: main外资源.
    3. 案例: 定义1个列表, 1个进程添加数据, 另1个进程查看数据, 看是否能查到数据即可.
"""

# 需求: 在不同进程中修改列表 my_list = [], 并新增元素, 观察结果.
import multiprocessing, time

# 1. 定义列表, 表示: "共享"资源.
my_list = []
print('看看我执行了几次!')      # p1, p2, main 一共三次.

# 2. 定义函数 write_data(), 往 列表中添加元素.
def write_data():
    for i in range(1, 6):
        # 具体的添加数据到列表的动作.
        my_list.append(i)
        # 为了让效果更明显, 我们打印下 添加的过程.
        print(f'add: {i}')
    # 添加完毕后, 打印下 列表的 元素.
    print(f'write_data函数结果: {my_list}')


# 3. 定义函数 read_data(), 从列表中读取数据.
def read_data():
    # 为了保证write_data()函数一定执行结束, 我们这里加个休眠线程.
    time.sleep(3)
    print(f'read_data函数结果: {my_list}')


# main函数
if __name__ == '__main__':
    # 4. 创建两个进程, 分别关联: 两个函数.
    p1 = multiprocessing.Process(target=write_data)
    p2 = multiprocessing.Process(target=read_data)

    # 5. 执行进程.
    p1.start()      # write_data函数结果: [1, 2, 3, 4, 5]
    p2.start()      # read_data函数结果: []

    # print('看看我执行了几次!')