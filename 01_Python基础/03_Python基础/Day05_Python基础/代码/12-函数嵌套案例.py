# 需求1: 定义函数print_line()用于打印 一行图形, 定义函数 print_lines(), 底层调用 print_line()函数, 实现打印多行图形.

# 1. 定义函数print_line()用于打印 一行图形.
def print_line(cols):       # columns: 列.
    """
    定义函数, 用于打印1行图形.
    :param cols: 该行的列数.
    :return: 无
    """
    for i in range(cols):
        print('-', end='')
    print()                 # 打印完毕后, 记得换行


# 2. 定义函数 print_lines(), 底层调用 print_line()函数, 实现打印多行图形.
def print_lines(rows, cols):
    """
    该函数用于打印 多行图形.
    :param rows: 行数
    :param cols: 每行的列数
    :return: 无
    """
    for i in range(rows):
        # 具体的打印每一行图形的操作, 直接调用 print_line()函数即可.
        print_line(cols)


# 3. 调用print_lines()函数, 查看图形.
print_lines(3, 5)




# 需求2: 定义函数 get_sum(), 用于计算三个整数的和.  然后再 get_avg()函数中, 调用get_sum()函数, 并计算它们(三个整数)的平均值.
# 1. 定义函数 get_sum(), 用于计算三个整数的和
def get_sum(a, b, c):
    """
    求和函数.
    :param a: 求和操作的第1个整数
    :param b: 第2个整数
    :param c: 第3个整数
    :return: 3个整数的 和
    """
    sum = a + b + c
    return sum

# 2. get_avg()函数中, 调用get_sum()函数, 并计算它们(三个整数)的平均值.
def get_avg(a, b, c):
    """
    求 平均值 函数
    :param a: 要操作的第1个整数
    :param b: 第2个整数
    :param c: 第3个整数
    :return: 3个整数的平均值.
    """
    sum = get_sum(a, b, c)
    avg = sum // 3
    return avg

# 3. 打印结果.
avg = get_avg(10, 20, 30)
print(f'平均值为: {avg}')