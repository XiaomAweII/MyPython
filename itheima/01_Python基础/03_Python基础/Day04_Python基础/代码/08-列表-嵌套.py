"""
列表嵌套介绍:
    概述:
        所谓的列表嵌套指的是: 列表的元素还是1个列表, 这种写法就称之为: 列表嵌套.
    格式:
        列表名 = [列表1, 列表2, 列表3...]
    如何获取元素?
        方式1: 遍历.
        方式2: 列表名[外部列表的索引][内部列表的索引]
"""

# 1. 定义列表, 记录: 姓名.
name_list = [['刘怡铭', '陈正', '蔡徐坤'], ['王心凌', '王祖贤', '张曼玉'], ['李白', '杜甫', '李贺']]


# 2. 获取指定的元素.
print(name_list[1])         # ['王心凌', '王祖贤', '张曼玉']
print(name_list[1][0])      # '王心凌'
print(name_list[2][1])      # '杜甫'
print('-' * 28)

# 3. 遍历.
for i in range(len(name_list)):
    # i 代表的是 二维列表的每个元素的 索引.
    child_list = name_list[i]           # ['刘怡铭', '陈正', '蔡徐坤']...
    # 因为 child_list 还是1个列表, 所以我们接着遍历.
    for j in range(len(child_list)):
        print(child_list[j])
    print('-' * 28)

print('*' * 28)

for child_list in name_list:
    for name in child_list:
        print(name)
    print('#' * 28)