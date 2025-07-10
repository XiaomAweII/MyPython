# 自定义的 学生类, 描述学生信息的, 属性: 姓名, 性别, 年龄, 联系方式, 描述信息.
class Student(object):
    # 初始化属性信息
    def __init__(self, name, gender, age, mobile, des):
        """
        魔法方法 init, 初始化学生的信息.
        :param name: 学生姓名
        :param gender: 性别
        :param age: 年龄
        :param mobile: 手机号
        :param des: description, 描述信息
        """
        self.name = name
        self.gender = gender
        self.age = age
        self.mobile = mobile
        self.des = des

    # 打印学生对象的 各个属性信息.
    def __str__(self):
        return f'姓名:{self.name}, 性别:{self.gender}, 年龄:{self.age}, 手机号:{self.mobile}, 描述信息:{self.des}'


# 测试代码, 访问在 调用者中 执行当期的测试内容.
if __name__ == '__main__':
    # 测试学生类.
    s = Student('乔峰', '男', 39, '13112345678', '威武, 刚猛, 喝酒')
    print(s)
