"""
扩展1: dict属性
    概述:
        它是类的内置属性, 类似于: 魔法方法一样, 是可以被直接调用的.
    格式:
        对象名.__dict__
    作用:
        把对象的各个属性信息, 封装成 字典形式,  属性名做键, 属性值作为值.
"""

# 导包
from student import Student

# 在main方法中测试.
if __name__ == '__main__':
    # 1. 创建3个学生信息
    s1 = Student('乔峰', '男', 33, '131', '帮主')
    s2 = Student('阿朱', '女', 26, '151', '帮主夫人')
    s3 = Student('虚竹', '男', 29, '186', '和尚')

    # 需求1: 把 s1 对象 封装成字典形式.
    # 方式1: 手动封装.
    dict1 = {'name': s1.name, 'gender': s1.gender, 'age': s1.age, 'mobile': s1.mobile, 'des': s1.des}
    print(dict1)

    # 方式2: __dict__ 内置属性.
    dict2 = s1.__dict__
    print(dict2)
    print('-' * 31)

    # 需求2: 把 [学生对象, 学生对象, 学生对象]     =>      [{学生信息}, {学生信息}, {学生信息}]
    student_list = [s1, s2, s3]
    # 方式1: 分解版.
    list_data = []
    for stu in student_list:
        list_data.append(stu.__dict__)
    print(list_data)

    # 方式2: 列表推导式
    list_data2 = [stu.__dict__ for stu in student_list]
    print(list_data2)