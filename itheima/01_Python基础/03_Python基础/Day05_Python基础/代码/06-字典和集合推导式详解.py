# 演示 集合推导式.

# 需求1: 生成 0 ~ 9 的偶数 集合.
set1 = {i for i in range(10) if i % 2 == 0}
print(set1)

# 需求2: 创建1个集合, 数据为下方列表的 2次方.
# 目的: 集合元素具有 唯一性, 会自动去重.
list1 = [1, 1, 2]
set2 = {i ** 2 for i in list1}
print(set2)
print('-' * 28)

# 演示 字典推导式, 回顾字典写法: dict1 = {'name':'张三', 'age':23}
# 需求3: 创建1个字典, key是 1 ~ 5的数字, value是该数字的2次方, 例如: {1:2, 2:4, 3:9, 4:16, 5:25}
dict1 = {i: i ** 2 for i in range(1, 6)}
print(dict1)
print('-' * 28)

# 需求4: 把下述的两个列表, 拼接成1个字典.
# 细节: 两个列表的元素个数(长度) 要 一致.
list1 = ['name', 'age', 'gender']
list2 = ['Tom', 20, 'male']

dict2 = {list1[i]: list2[i] for i in range(len(list1))}
print(dict2)
