# 该文件是 学生管理系统文件, 主要完成: 学生管理系统的 主要业务逻辑的.

# 导包
import time
from student import Student


# cms: Content Management System, 内容管理系统
class StudentCms(object):
    # 1. 初始化属性, student_list = [学生对象, 学生对象...], 记录所有的 学生信息.
    def __init__(self):
        # s1 = Student('乔峰', '男', 33, '131', '帮主')
        # s2 = Student('阿朱', '女', 26, '151', '帮主夫人')
        # s3 = Student('虚竹', '男', 29, '186', '和尚')
        self.student_list = []  # 用来记录所有学生信息的.

    # 2. 定义函数 show_view(), 用于展示: 提示界面.
    # 因为该函数仅仅是用来做打印和提示的, 函数内部不涉及到 self对象的使用等, 所以可以定义成: 静态方法.
    @staticmethod
    def show_view():
        print('*' * 31)
        print('欢迎来到学生管理系统V2.0界面:')
        print('\t1.添加学员')
        print('\t2.修改学员')
        print('\t3.删除学员')
        print('\t4.查询某个学员')
        print('\t5.显示所有学员')
        print('\t6.保存信息')
        print('\t0.退出系统')
        print('*' * 31)

    # 3. 定义函数 add_student(), 表示 添加学生.
    def add_student(self):
        # 3.1 提示用户录入学生的信息, 并接收.
        name = input('请录入要添加的学生的 姓名: ')
        gender = input('请录入要添加的学生的 性别: ')
        age = int(input('请录入要添加的学生的 年龄: '))  # 记得把年龄换成 整数.
        mobile = input('请录入要添加的学生的 手机号: ')
        des = input('请录入要添加的学生的 描述信息: ')
        # 3.2 把上述的信息, 封装成学生对象.
        stu = Student(name, gender, age, mobile, des)
        # 3.3 把上述封装好的学生对象, 添加到 学生列表中.
        self.student_list.append(stu)
        # 3.4 打印 提示信息即可.
        print(f'添加学生信息成功!\n')

    # 4. 定义函数 update_student(), 表示 修改学员
    def update_student(self):
        # 4.1 提示用户录入要修改的学生的 姓名.
        update_name = input('请录入要修改的学生的姓名: ')
        # 4.2 遍历学生列表, 获取到每个学生信息.
        for stu in self.student_list:
            # 4.3 判断当前学生姓名 是否和 要修改的学生姓名一致.
            if stu.name == update_name:
                # 4.4 如果一致, 就提示用重新录入: 她/他的 性别, 年龄, 手机号, 描述信息等.
                stu.gender = input('请录入修改后的学生的 性别: ')
                stu.age = int(input('请录入修改后的学生的 年龄: '))  # 记得把年龄换成 整数.
                stu.mobile = input('请录入修改后的学生的 手机号: ')
                stu.des = input('请录入修改后的学生的 描述信息: ')
                # 提示: 修改成功.
                print('修改学生信息成功!\n')
                # 核心细节: 修改后, 记得 break
                break
        # 4.5 如果循环结束, 还没有匹配到, 就提示: 查无此人即可.
        else:
            print('查无此人, 请校验后重新修改! \n')

    # 5. 定义函数 del_student(), 表示 删除学员
    def del_student(self):
        # 5.1 提示用户录入要删除的学生的 姓名.
        del_name = input('请录入要删除的学生的姓名: ')
        # 5.2 遍历学生列表, 获取到每个学生信息.
        for stu in self.student_list:
            # 5.3 判断当前学生姓名 是否和 要删除的学生姓名一致.
            if stu.name == del_name:
                # 5.4 如果一致, 就 删除当前学生对象.
                self.student_list.remove(stu)
                # 提示: 删除成功.
                print('删除学生信息成功!\n')
                # 核心细节: 删除后, 记得 break
                break
        # 5.5 如果循环结束, 还没有匹配到, 就提示: 查无此人即可.
        else:
            print('查无此人, 请校验后重新删除! \n')

    # 6. 定义函数 search_one_student(), 表示 查询某个学员
    def search_one_student(self):
        # 6.1 提示用户录入要查询的学生的 姓名.
        search_name = input('请录入要查询的学生的姓名: ')
        # 6.2 遍历学生列表, 获取到每个学生信息.
        for stu in self.student_list:
            # 6.3 判断当前学生姓名 是否和 要查询的学生姓名一致.
            if stu.name == search_name:
                # 6.4 如果一致, 就 打印当前学生对象.
                print(stu, end='\n\n')
                # 核心细节: 查询后, 记得 break
                break
        # 6.5 如果循环结束, 还没有匹配到, 就提示: 查无此人即可.
        else:
            print('查无此人, 请校验后重新查询! \n')

    # 7. 定义函数 search_all_student(), 表示 显示所有学员
    def search_all_student(self):
        # 7.1 判断, 学生列表中 是否有学生信息, 有则打印, 无则提示.
        if len(self.student_list) > 0:
            # 7.2 走这里, 说明有 学生信息, 遍历即可.
            for stu in self.student_list:
                print(stu)
            print()  # 为了好看, 我们可以加个换行.
        else:
            # 7.3 走这里, 暂无学生信息, 提示即可.
            print('暂无学生信息, 请添加后查询! \n')

    # 8. 定义函数 save_student(), 表示 保存信息 到 文件中.
    def save_student(self):
        # 流程: [学生对象, 学生对象...] => [{学生信息}, {学生信息}...] => "[{学生信息}, {学生信息}...]" => 写到student_data.txt文件中
        # 8.1 把 列表存储学生对象 转成 列表存储字典的形式, 即: [学生对象, 学生对象...] => [{学生信息}, {学生信息}...]
        # student_dict = [stu.__dict__ for stu in self.student_list]
        # print(student_dict)

        # 8.2 把上述的 列表嵌套字典, 转成: 字符串形式.       [{学生信息}, {学生信息}...] => "[{学生信息}, {学生信息}...]"
        # student_data = str(student_dict)

        # 合并版.
        student_data = str([stu.__dict__ for stu in self.student_list])

        # 8.3 把上述的 字符串(学生信息) 写到目的地 文件中.
        with open('./student_data.txt', 'w', encoding='utf-8') as dest_f:
            # 一次性写入信息即可.
            dest_f.write(student_data)
            # 提示即可.
            print('学生信息保存成功!\n')

    # 9. 定义函数 load_student(), 表示 从文件中 加载学生信息到 student_list 列表中.
    def load_student(self):
        # 9.1 尝试 从文件中读取 所有的学生信息.
        try:
            with open('./student_data.txt', 'r', encoding='utf-8') as src_f:
                # 9.2 一次性从文件中加载 所有学生信息
                student_data = src_f.read()
                # 9.3 判断是否有数据, 没有就设置初值为: "[]"
                if len(student_data) <= 0:
                    student_data = "[]"
                # 9.4 流程: "[{学生信息}, {学生信息}...]"  => [{学生信息}, {学生信息}...] => [学生对象, 学生对象...]
                list_data = eval(student_data)          # eval() 去掉两端的引号, 是啥就是啥
                # list_data = list(student_data)        # 这个转不了, 因为是: 列表嵌套字典的形式.
                # 9.5 列表推导式, 实现: [{学生信息}, {学生信息}...] => [学生对象, 学生对象...]
                self.student_list = [Student(stu_dict['name'], stu_dict['gender'], stu_dict['age'], stu_dict['mobile'], stu_dict['des']) for stu_dict in list_data]
                # print(self.student_list)
        except:
            # 9.6 走这里, 说明文件不存在, 创建即可.
            dest_f = open('./student_data.txt', 'w', encoding='utf-8')
            dest_f.close()

            # 或者直接写 pass

    # 10. 定义函数 start(), 表示: 程序的入口, 在这里完成具体的逻辑.
    def start(self):
        # 10.1 从文件中加载学生信息 到 student_list 列表中.
        self.load_student()

        # 10.2 通过while True死循环, 完成对应的业务逻辑操作.
        while True:
            # 10.3 为了效果更明显, 模拟系统启动用户等待, 加入休眠线程.
            time.sleep(0.5)
            # 10.4. 打印 (选择)功能界面.
            # self.show_view()        # 静态方法调用方式1: 对象名. 的方式调用, 可以, 但是不推荐.
            StudentCms.show_view()    # 静态方法调用方式2: 类名. 的方式调用, 推荐写法.
            # 10.5 提示用户录入他/她要操作的选项编号, 并接收.
            input_num = input('请录入您要操作的编号: ')
            # 10.6 根据用户录入的选项, 进行对应的操作.
            if input_num == '1':
                # print('添加学生! \n')
                self.add_student()
            elif input_num == '2':
                # print('修改学生! \n')
                self.update_student()
            elif input_num == '3':
                # print('删除学生! \n')
                self.del_student()
            elif input_num == '4':
                # print('查询单个学生! \n')
                self.search_one_student()
            elif input_num == '5':
                # print('显示所有学生! \n')
                self.search_all_student()
            elif input_num == '6':
                # print('保存学生信息! \n')
                self.save_student()
            elif input_num == '0':
                # 退出系统时, 需要二次确认.
                # 询问用户是否要退出, 并接收.
                result = input('您确定要退出吗? (Y/N or 其它): ')
                if result.upper() == 'Y':
                    print('谢谢您的使用, 期待下次再会! \n')
                    break
            else:
                print('录入有误, 新功能正在加急研发中, 请耐心等待!...\n')


# 测试代码
if __name__ == '__main__':
    # 创建对象
    stu_cms = StudentCms()
    # 启动程序, 测试.
    stu_cms.start()

    # 测试 提示界面.
    # stu_cms.show_view()
    # 测试 学生列表.
    # print(f'目前有的学生信息: {stu_cms.student_list}')
