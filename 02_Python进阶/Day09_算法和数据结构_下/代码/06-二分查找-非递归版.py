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
    时间复杂度:
        最优: O(1)
        最坏: O(logn)
"""


# 1. 定义函数 binary_search(my_list, item), 表示: 二分查找
def binary_search(my_list, item):
    """
    递归版, 二分查找
    :param my_list: 记录的元素, 注意: 有序的.
    :param item: 要被查找的元素
    :return: 查找结果, True: 找到了, False: 没有找到.
    """
    # 1.1 定义两个变量, start, end, 分别表示: 查找的范围(即: 起始索引 和 结束索引)
    start = 0
    end = len(my_list) - 1
    # 1.2 具体的查找过程.
    while start <= end:
        # 1.2 计算中间索引.
        mid = (start + end) // 2
        if item == my_list[mid]:
            return True  # 找到了
        elif item < my_list[mid]:
            end = mid - 1  # 去 中值前 查找
        else:
            start = mid + 1

    # 走到这里, 还没找到, 返回False
    return False


# 在main方法中测试.
if __name__ == '__main__':
    # 2. 定义列表.
    my_list = [8, 22, 31, 32, 44, 55, 56, 70, 78]
    # 3. 调用上述的函数, 查找元素, 并打印结果.
    print(binary_search(my_list, 39))
