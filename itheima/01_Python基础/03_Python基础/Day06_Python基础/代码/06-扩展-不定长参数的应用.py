# 需求: 定义get_sum()函数, 分别实现计算求 任意个整数和, 例如: 2个整数和, 3个整数和, 4个整数和...

# 分析: 因为get_sum()具体要接受多少个实参(整数), 也不确定, 所以推荐使用: 可变参数.
def get_sum2(*args):
    # args: 接收所有的 位置参数, 并封装成: 元组.  这里你直接把 args当元组用即可.
    # 例如:  args = (10, 20, 30)
    # sum = 0
    # for i in args:
    #     sum += i
    # return sum

    # 扩展: 可以直接用 sum()函数, 计算元组的 元素和.
    return sum(args)


# 调用函数
if __name__ == '__main__':
    result1 = get_sum2(10, 20)
    print(f'result1: {result1}')

    result2 = get_sum2(10, 20, 30)
    print(f'result2: {result2}')

    result3 = get_sum2(10, 20, 30, 50)
    print(f'result3: {result3}')

    # 细节: 可变参数最少可以传入 0个参数, 至多可以传入 无数个参数.
    result4 = get_sum2()
    print(f'result4: {result4}')

