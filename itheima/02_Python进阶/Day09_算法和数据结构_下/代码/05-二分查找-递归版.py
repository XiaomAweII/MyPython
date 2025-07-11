"""
二分查找 算法解释:
    概述:
        它是一种高效的查找类的算法, 也叫: 折半查找.
    前提:
        要查找的列表, 必须是: 有序的.
    原理:
        找到中间值, 如果要查找的值 和 中间值一致, 就返回True
        如果比中间值小, 就去 中值前(中间值的左边)  查找.
        如果比中间值大, 就去 中值后(中间值的右边)  查找.
"""


# 1. 定义函数 binary_search(my_list, item), 表示: 二分查找
def binary_search(my_list, item):
    """
    递归版, 二分查找
    :param my_list: 记录的元素, 注意: 有序的.
    :param item: 要被查找的元素
    :return: 查找结果, True: 找到了, False: 没有找到.
    """
    # 1.1 获取列表的长度.
    n = len(my_list)

    # 1.2 判断列表的长度如果 小于等于 0, 直接返回False
    if n <= 0:
        return False

    # 1.3 获取中间值的索引.
    mid = n // 2

    # 1.4 具体的比较过程.
    if item == my_list[mid]:
        return True  # 找到了
    elif item < my_list[mid]:
        # 如果要查找的值 比中间值小, 就去 中值前(中间值的左边)  查找.
        return binary_search(my_list[:mid], item)
    else:
        # 如果比中间值大, 就去 中值后(中间值的右边)  查找.
        return binary_search(my_list[mid + 1:], item)

    # 1.5 走到这里, 说明肯定没有找到.
    # return False


# 在main方法中测试.
if __name__ == '__main__':
    # 2. 定义列表.
    my_list = [8, 22, 31, 32, 44, 55, 56, 70, 78]
    # 3. 调用上述的函数, 查找元素, 并打印结果.
    print(binary_search(my_list, 31))
