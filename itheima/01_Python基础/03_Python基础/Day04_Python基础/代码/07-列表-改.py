"""
列表 修改相关的 函数如下:
    列表名[索引] = 值             根据索引, 修改其对应的元素
    列表名.reverse()             反转列表元素.
    列表名.sort(reverse=True)    对列表元素值进行排序, reverse=False 升序,  reverse=True 降序
"""

# 1. 定义列表.
list1 = [10, 20, 30, 10, 30]


# 2. 演示列表 修改相关的操作.
# list1[2] = 300      # 修改索引值为2的元素为: 300
# list1[20] = 666     # 报错, 索引越界.

# 反转列表元素.
# list1.reverse()       # [30, 10, 30, 20, 10]

# 对列表元素内容进行排序.
# list1.sort()                # 默认: 升序
# list1.sort(reverse=False)   # 效果同上, 升序
list1.sort(reverse=True)      # 降序

# 3. 打印列表元素内容.
print(f'list1: {list1}')