"""
观察如下的代码, 分析程序结果, 得到的结论如下.
    形参是可变类型:    形参的改变直接影响实参.
    形参是不可变类型:  形参的改变对实参没有任何影响.
"""

# 定义函数, 接收 参数, 一会传入: 整数(int), 不可变类型.
def change(num):
    num = 200

# 定义函数, 接收 参数, 一会传入: 列表(list), 可变类型.
def change2(list1):
    list1[1] = 28

if __name__ == '__main__':
    # 演示: 不可变类型 函数的调用.
    a = 100
    print(f'调用 change 函数前, a: {a}')     # 100
    change(a)
    print(f'调用 change 函数前, a: {a}')     # 100

    # 演示: 可变类型 函数的调用.
    list1 = [1, 2, 3, 4, 5]
    print(f'调用 change 函数前, list1: {list1}')  # 1, 2, 3, 4, 5
    change2(list1)
    print(f'调用 change 函数后, list1: {list1}')  # 1, 28, 3, 4, 5


