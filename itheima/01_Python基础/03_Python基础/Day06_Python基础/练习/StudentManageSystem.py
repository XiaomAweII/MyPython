# todo 猜数字, 石头 获得金币
"""
数据的结构:学生信息用字典存储
{id:["name","tel"],}
"""
stu = dict() # 学生字典


def add_info():
    """
    当用户选择1的时候, 实现操作: 添加学生(学生编号, 学生姓名, 手机号).
    :return: 打印添加成功
    """

    # TODO 要有异常处理, 违法字符, 已经存在, 要先查找
    id = int(input("请输入学生编号: "))
    name = input("请输入学生姓名: ")
    tel = int(input("请输入手机号: "))

    if id in stu: #
        print("id重复")
    else: # 添加学生
        stu[id] = [name, tel]
        print("成功添加")


def delete_info():
    """
    当用户选择2的时候, 实现操作: 删除学生(根据编号删除)
    :return:
    """

    id = int(input("请输入要删除的学生编号: "))
    # 先查找
    if id not in stu:
        print("查无此学生")
    else:
        # 删除
        del stu[id]
        print("删除成功")


def update_info():
    # 当用户选择3的时候, 实现操作: 修改学生信息(只能改姓名, 手机号)
    id = int(input("请输入要更新的学生编号: "))
    # 先查找
    if id not in stu:
        print("查无此学生")
    else:
        name = input("请输入学生姓名: ")
        tel = int(input("请输入手机号: "))
        stu[id] = [name, tel]
        print("更新成功")


def search_info():
    name = input("请输入学生姓名: ")
    # 先查找
    for k, v in stu:
        if v[0] == name:
            print(f"{id}:{name}:{tel}")
    else:
        print("查无此学生")

def search_all():
    if len(stu) == 0:
        print("无学生")
    else:
        for k, v in stu.items():
            print(f"{k}:{v[0]}:{v[1]}")
        print("查询结束")

def main():
    while True:
        input_num = int(input("""
        先打印提示界面(1-6的数字), 让用户选择他/她要进行的操作.
        当用户选择1的时候, 实现操作: 添加学生(学生编号, 学生姓名, 手机号).
        当用户选择2的时候, 实现操作: 删除学生(根据编号删除)
        当用户选择3的时候, 实现操作: 修改学生信息(只能改姓名, 手机号)
        当用户选择4的时候, 实现操作: 查询单个学生信息(根据姓名查)
        当用户选择5的时候, 实现操作: 查询所有学生信息
        当用户选择6的时候, 实现操作: 退出系统
        请输入你的选择: 
        """))

        if input_num == 6:
            # 当用户选择6的时候, 实现操作: 退出系统
            break
        elif input_num == 1:
            # 当用户选择1的时候, 实现操作: 添加学生(学生编号, 学生姓名, 手机号).
            add_info()
        elif input_num == 2:
            # 当用户选择2的时候, 实现操作: 删除学生(根据编号删除)
            delete_info()
        elif input_num == 3:
            # 当用户选择3的时候, 实现操作: 修改学生信息(只能改姓名, 手机号)
            update_info()
        elif input_num == 4:
            # 当用户选择4的时候, 实现操作: 查询单个学生信息(根据姓名查)
            search_info()
        else:
            # 当用户选择5的时候, 实现操作: 查询所有学生信息
            search_all()


if __name__ == "__main__":
    main()