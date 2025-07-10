# 这个模块, 表示程序(学生管理系统-面向对象版)的主入口, 所有的代码都是从这里开始执行的.


# 导包.
from student_cms import StudentCms


# 启动程序即可.
if __name__ == '__main__':
    # 创建 学生管理系统文件的 对象.
    stu_cms = StudentCms()
    # 启动程序.
    stu_cms.start()