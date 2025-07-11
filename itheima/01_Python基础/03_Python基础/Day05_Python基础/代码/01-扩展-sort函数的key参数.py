# 案例: 演示下 sort()函数的用法.

# 需求1: 数值列表, 元素排序.
# 1. 定义列表.
list1 = [11, 33, 22, 55]

# 2. 对列表元素值进行排序.
list1.sort()  # 升序
list1.sort(reverse=False)  # 升序
list1.sort(reverse=True)  # 降序

# 3. 打印列表元素内容.
print(f'list1: {list1}')
print("-" * 28)

# 需求2: 字符串列表, 元素排序.
list2 = ['bc', 'abc', 'xyz1', 'h']

# 对list2列表的元素排序.
# list2.sort()        # 默认按照 字典顺序排列, 即: ['abc', 'bc', 'h', 'xyz1']

# 要求: 按照 字母(单词) 的长度进行排序.
list2.sort(key=len)  # 会先用len()函数计算每个字符串的长度, 然后按照结果(长度)排序
list2.sort(key=len, reverse=True)  # 会先用len()函数计算每个字符串的长度, 然后按照结果(长度)排序

print(f'list2: {list2}')
print("-" * 28)

# 需求3: 对 列表嵌套元组, 内容排序.
list3 = [(1, 3), (2, 2), (5, 1), (3, 9)]

# 该函数接收1个元组, 然后返回该元组的 第2个元素(即: 索引为1的元素)
def get_data(t1):
    return t1[1]

# 要求: 按照 元素的第2个元素值 进行排序.
list3.sort(key=get_data)        # key参数接收的是1个函数, 该函数会做用到列表中的每个元素.
list3.sort(key=get_data, reverse=True)        # key参数接收的是1个函数, 该函数会做用到列表中的每个元素.

print(f'list3: {list3}')
