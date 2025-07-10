"""
需求:
    1. 先打印提示界面(1-6的数字), 让用户选择他/她要进行的操作.
    2. 当用户选择1的时候, 实现操作: 添加学生(学生编号, 学生姓名, 手机号).
    3. 当用户选择2的时候, 实现操作: 删除学生(根据编号删除)
    4. 当用户选择3的时候, 实现操作: 修改学生信息(只能改姓名, 手机号)
    5. 当用户选择4的时候, 实现操作: 查询单个学生信息(根据姓名查)
    6. 当用户选择5的时候, 实现操作: 查询所有学生信息
    7. 当用户选择6的时候, 实现操作: 退出系统

目的
  把之前所需的知识点: if, for, 函数等知识点 结合到一起, 做一个综合案例.

思路
    1. 定义函数 print_info(), 打印提示信息.
    2. 自定义while True循环逻辑, 实现: 用户录入什么数据, 就进行相应的操作
        注意: 处理一下非法值.
    3. 自定义函数 add_info(), 实现: 添加学生
        编号必须唯一
    4. 自定义函数 delete_info(), 实现: 删除学生
        根据编号删除(唯一)
        根据姓名删除(可重复)
    5. 自定义函数 update_info(), 实现: 修改学生信息
        根据编号修改, 只能修改: 姓名, 手机号.
    6. 自定义函数 search_info(), 实现: 查询某个学生信息.
        根据 姓名 查询(可重复)
        根据 学号 查询(唯一)
    7. 自定义函数 search_all(), 实现: 查询所有学生的信息.
    8. 在main函数中, 完成: 程序的入口启动动作.

升级版思路:

Day09下午答辩:
    1. 以组的形式答辩, 2 ~ 3人上台宣讲(PPT形式), 思路: 项目名, 小组成员及职责, 具体的项目截图(效果图), 项目亮点, 遇到的Bug及解决方案...
    2. 宣讲结束后, 其他组成员, 可以进行提问.
    3. 投票.
    4. 项目的"延伸"思路:
        基本版: 学生管理系统 + 列表嵌套字典
        升级版: 学生管理系统 + 文件
        进阶版: 学生管理系统 + 数据库
        终极版: 黑马**游戏, 两套系统 A: (管理员系统(就是学生管理系统), 可以管理用户, 增删改查), B:游戏系统(登陆, 注册, 玩儿游戏, 石头剪刀布, 猜数字, 打印图形, 约瑟夫环)..
"""

# 1. 定义函数 print_info(), 打印提示信息.
def print_info():
    print('1.添加学生')
    print('2.删除学生')
    print('3.修改学生信息')
    print('4.查询单个学生信息')
    print('5.查询所有学生信息')
    print('6.退出系统')


# 2. 自定义while True循环逻辑, 实现: 用户录入什么数据, 就进行相应的操作. 注意: 处理一下非法值.
def student_manager():
    # 自定义while True循环逻辑.
    while True:
        # 2.1 打印提示界面
        print_info()

        # 2.2 接收用户录入的 编号, 注意: 不要转成整数.
        input_num = input('请录入您要操作的编号: ')

        # 2.3 判断用户选择的 选项, 并进行相应的操作.
        if input_num == '1':
            # print('添加学生')
            add_info()          # 调用函数, 实现 添加学生信息.
        elif input_num == '2':
            # print('删除学生')
            # delete_info_by_id()  # 根据 id(学号) 删除学生信息
            delete_info_by_name()  # 根据 name(姓名) 删除学生信息
        elif input_num == '3':
            # print('修改学生信息')
            update_info()
        elif input_num == '4':
            # print('查询单个学生信息')
            # search_info_by_id()   # 根据 id(学号) 查询学生信息
            search_info_by_name()   # 根据 name(姓名) 查询学生信息
        elif input_num == '5':
            # print('查询所有学生信息')
            search_all()
        elif input_num == '6':
            print('退出系统, 期待下次再见!')
            break  # 记得结束循环.
        else:
            print('录入有误, 请重新录入!\n')

# 3. 自定义函数 add_info(), 实现: 添加学生,   要求: 编号必须唯一
# 1个学生信息, 格式为: 学号(id), 姓名(name), 手机号(tel)
# 3.1 定义列表 user_info, 用来记录(存储) 所有学生的信息, 格式为: 列表嵌套字典.
user_info = [
    # {'id': 'hm01', 'name': '李白', 'tel': '111'},
    # {'id': 'hm02', 'name': '韩信', 'tel': '222'},
    # {'id': 'hm03', 'name': '达摩', 'tel': '333'},
    # {'id': 'hm03', 'name': '达摩', 'tel': '333'},
    # {'id': 'hm03', 'name': '达摩', 'tel': '333'},
    # {'id': 'hm03', 'name': '达摩', 'tel': '333'}
]

# 3.2 定义 add_info()函数, 实现添加学生信息.
def add_info():
    # 3.3 提示用户录入 要添加的学生的 学号, 并接收.
    new_id = input('请录入要添加的学生学号: ')
    # 3.4 判断, 要添加的学号是否存在, 如果存在, 就提示, 并结束添加.
    for stu in user_info:
        # stu的格式:  {'id': 'hm01', 'name': '李白', 'tel': '111'}
        if new_id == stu['id']:
            # 走这里, 说明学号重复.
            print(f'学号 {new_id} 已存在, 请校验后重新添加!\n')
            return      # 结束函数.

    # 3.5 走到这里, 说明学号是 唯一的, 就提示用户录入 要添加的学生的 姓名 和 手机号 并接收.
    new_name = input('请录入要添加的学生姓名: ')
    new_tel = input('请录入要添加的学生手机号: ')

    # 3.6 将用户录入的学生信息, 封装成: 字典形式.
    new_info = {'id': new_id, 'name': new_name, 'tel': new_tel}

    # 3.7 把上述的学生信息(字典形式), 添加到: 学生列表中, 至此, 添加学生信息结束.
    user_info.append(new_info)
    print(f'学号 {new_id} 学生信息添加成功!\n')


# 4. 自定义函数 delete_info(), 实现: 删除学生
# 场景1: 根据编号删除(唯一)
def delete_info_by_id():
    # 4.1 提示用户录入要删除的学号, 并接收.
    del_id = input('请录入要删除的学号: ')
    # 4.2 遍历 学生列表, 获取到每个学生信息, 然后判断是否 有该学号的信息.
    for stu in user_info:
        # 4.3 如果有, 就删除该学生信息, 并提示. 至此, 删除结束.
        if stu['id'] == del_id:
            user_info.remove(stu)   # 具体删除学生信息的动作.
            print(f'学号为 {del_id} 的学生信息已成功删除!\n')
            break
    else:
        # 4.4 走到这里, 说明该学号不存在, 提示即可.
        print('该学号不存在, 请校验后重新删除!\n')


# 场景2: 根据姓名删除(可重复)
def delete_info_by_name():
    # 4.0 核心: 定义标记变量 flag, 表示: 是否删除学生. 默认是: False(没删除), True: 删除.
    flag = False

    # 4.1 提示用户录入要删除的学生姓名, 并接收.
    del_name = input('请录入要删除的学生姓名: ')
    # 4.2 遍历 学生列表, 获取到每个学生信息, 然后判断是否 有该 姓名 的信息.
    i = 0
    while i < len(user_info):
        stu = user_info[i]      # stu就代表着 某个学生信息.
        # 4.3 如果有, 就删除该学生信息, 并提示. 至此, 删除结束.
        if stu['name'] == del_name:
            user_info.remove(stu)   # 具体删除学生信息的动作.
            # 删除学生信息后, 后续元素会往前提一位, 数据会变化. 索引 -= 1即可.
            i -= 1
            # 修改标记变量的值
            flag = True
        # 无论是否删除, 都要开始判断下个学生信息了.
        i += 1

    # 4.4 走到这里, 判断是否删除学生信息, 并提示.
    if flag == False:
        print('该 姓名 不存在, 请校验后重新删除!\n')
    else:
        print(f'姓名为 {del_name} 的学生信息已成功删除!\n')


# 5. 自定义函数 update_info(), 实现: 修改学生信息.  要求: 根据编号修改, 只能修改: 姓名, 手机号.
def update_info():
    # 5.1 提示用户录入 要修改的学生的 学号, 并接收.
    update_id = input('请录入要修改的学生学号: ')
    # 5.2 判断, 要修改的学号是否存在, 如果存在, 就修改.
    for stu in user_info:
        # stu的格式:  {'id': 'hm01', 'name': '李白', 'tel': '111'}
        if update_id == stu['id']:
            # 走这里, 说明 学号存在.
            # 5.3 提示用户录入 要修改的学生的 姓名 和 手机号 并接收.
            update_name = input('请录入要添加的学生姓名: ')
            update_tel = input('请录入要添加的学生手机号: ')
            # 5.4 具体的修改学生信息的动作.
            stu['name'] = update_name
            stu['tel'] = update_tel
            print(f'学号 {update_id} 学生信息修改成功!\n')
            # 5.5 因为学号具有唯一性, 只要修改了, break即可.
            break
    else:
        print(f'学号 {update_id} 不存在, 请校验后重新修改!\n')
        return      # 结束函数.


# 6. 自定义函数 search_info(), 实现: 查询某个学生信息.
# 场景1: 根据 学号 查询(唯一)
def search_info_by_id():
    # 6.1 提示用户录入 要查询的学生的 学号, 并接收.
    search_id = input('请录入要查询的学生学号: ')
    # 6.2 判断, 要查询的学号是否存在, 如果存在, 就打印该学号的信息.
    for stu in user_info:
        # stu的格式:  {'id': 'hm01', 'name': '李白', 'tel': '111'}
        if search_id == stu['id']:
            # 6.3 走这里, 说明 学号存在, 就打印该学号的信息
            print(f'id = {stu["id"]}, name = {stu["name"]}, tel = {stu["tel"]}\n')
            # 6.4 因为学号具有唯一性, 只要打印了, break即可.
            break
    else:
        # 6.5 走到这里, 学号不存在, 提示即可.
        print(f'学号 {search_id} 不存在, 请校验后重新查询!\n')


# 场景2: 根据 姓名 查询(可重复)
def search_info_by_name():
    # 核心: 标记变量
    flag = False        # 假设: False 没找到,  True: 找到了.
    # 6.1 提示用户录入 要查询的学生的 姓名, 并接收.
    search_name = input('请录入要查询的学生姓名: ')
    # 6.2 判断, 要查询的 姓名 是否存在, 如果存在, 就打印该 姓名 的信息.
    for stu in user_info:
        # stu的格式:  {'id': 'hm01', 'name': '李白', 'tel': '111'}
        if search_name == stu['name']:
            # 6.3 走这里, 说明 姓名存在, 就打印该姓名的信息
            print(f'id = {stu["id"]}, name = {stu["name"]}, tel = {stu["tel"]}')
            flag = True

    # 6.5 判断, 如果没有找到该姓名的信息, 就 提示即可.
    # if not flag:
    if flag == False:
        print(f'姓名 {search_name} 不存在, 请校验后重新查询!\n')
    else:
        # 找到了, 我们加个空行, 好看点.
        print()


# 7. 自定义函数 search_all(), 实现: 查询所有学生的信息.
def search_all():
    # 7.1 判断是否有学生信息, 如果没有, 直接提示, 然后结束即可.
    if len(user_info) == 0:
        print('暂无学生信息, 请添加后重新查询!\n')
    else:
        # 7.2 走这里, 说明 查到了学生信息. 遍历打印即可.
        for stu in user_info:
            # print(stu)
            # 字典根据键获取值有两种方式:  字典名.get(键)   或者  字典名[键]
            print(f'id = {stu.get("id")}, name = {stu["name"]}, tel = {stu["tel"]}')
        # 7.3 为了格式好看, 记得加个换行.
        print()


# 8. main函数, 作为程序的主入口.
if __name__ == '__main__':
    # 启动学生管理系统.
    student_manager()

