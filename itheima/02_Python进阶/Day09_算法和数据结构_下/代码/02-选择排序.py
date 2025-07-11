"""
选择排序:
    概述/原理:
        每轮比较, 都找到最小值所在的索引, 然后和 最小索引进行交换即可.
        大白话: 选择排序就是把符合条件的元素给选择出来, 进行排序.
    推理过程:  假设列表长度为 5
        比较的轮数       每轮比较的次数     谁(索引)和谁(索引)比较             外循环(i), 内循环(j)
            0               4               0和1, 0和2, 0和3, 0和4          0        1 ~ 4
            1               3               1和2, 1和3, 1和4                1        2 ~ 4
            2               2               2和3, 2和4                     2         3 ~ 4
            3               1               3和4                           3         4
    核心:
        1. 比较的总轮数.      列表长度 - 1
        2. 每轮比较的次数.    i + 1  ~ 列表长度 - 1
        3. 谁和谁比较.       my_list[j] 和 my_list[min_index]
    时间复杂度:
        最优时间复杂度: O(n)
        最坏时间复杂度: O(n²)
    选择排序属于 不稳定排序算法.
"""


# 1. 定义函数 select_sort(my_list), 表示: 选择排序.
def select_sort(my_list):
    """
    通过选择排序的思路, 对 列表元素 进行排序.
    :param my_list: 要进行排序的列表
    :return:
    """
    # 1.1 获取列表的长度.
    n = len(my_list)
    # 1.2 开始每轮 每次的比较
    for i in range(n - 1):  # 外循环, 控制: 比较的轮数. 假设n为5, 则 i的值: 0, 1, 2, 3
        # 核心细节: 定义变量 min_index, 用于记录 (每轮)最小值的 索引.
        min_index = i       # 假设每轮的第1个值, 为: 最小值.
        for j in range(i + 1, n):  # 内循环, 控制: 每轮比较的次数.
            # 1.3 具体的比较过程: 如果当前元素 比 min_index 记录的元素还要小, 就用 min_index 记录住该元素的索引.
            if my_list[j] < my_list[min_index]:
                min_index = j
        # 1.4 走到这里, 说明1轮比较完毕, 我们要看 min_index的值有无发生改变, 改变了, 就是找到了 本轮的最小值.
        if min_index != i:
            my_list[i], my_list[min_index] = my_list[min_index], my_list[i]


# 在main函数中, 测试
if __name__ == '__main__':
    # 2. 定义列表, 记录要排序的数字.
    my_list = [5, 3, 4, 7, 2]
    # my_list = [2, 3, 4, 5, 7]
    print(f'排序前: {my_list}')

    # 3. 具体的排序动作.
    select_sort(my_list)
    print(f'排序后: {my_list}')
