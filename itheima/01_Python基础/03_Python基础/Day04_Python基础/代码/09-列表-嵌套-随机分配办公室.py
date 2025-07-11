"""
需求:
    已知有三个教室, 格式为: class_list = [[], [], []]
    只有有8名教师, 格式为: name_list = [1, 2, 3...]
    请用所学实现, 把8名教师随机分配到上述的3个教室中.
涉及到的知识点:
    列表嵌套, 随机数, 循环, append()...
"""
# 0.导包
import random

# 1. 定义列表, 记录: 教室.
class_list = [[], [], []]
# 2. 定义列表, 记录: 老师.
name_list = ['乔峰', '虚竹', '段誉', '杨过', '郭靖', '张三丰', '岳不群', '令狐冲']
# name_list = [1, 2, 3, 4, 5, 6, 7]

# 3. 开始随机分配, 遍历 老师列表, 获取到每一个老师(的名字)
for name in name_list:
    # 4. 核心细节: 随机生成教室的编号, 这个教室, 就是当前教师要去的教室.  添加即可.
    class_id = random.randint(0, 2)     # 包左包右.
    class_list[class_id].append(name)

# 5. 循环结束后, 教室分配完毕, 打印结果即可.
# 方式1: 直接输出.
# print(class_list)

# 方式2: 遍历输出.
for class_info in class_list:   # 遍历: 获取到每个教室的信息
    for name in class_info:     # 遍历: 获取到教室中的每个 老师的信息
        print(name)
    print('-' * 28)