"""
回顾:
    with 语法功能之所以那么强大, 底层用到的就是: 上下文管理器.
    即: with管理的就是上下文管理器对象.

上下文管理器介绍:
    1个类, 实现了 __enter__() 和 __exit__()这两个方法, 它就是: 上下文管理器.
    其中:
        __enter__() 会在 with 语句前执行, 一般用于: 初始化对象.
        __exit__() 会在 with语句后执行, 一般用于: 释放资源.
"""

# 需求: 自定义代码实现 上下文管理器对象, 完成 文件的 读, 写操作.

# 1 自定义MyFile类, 表示: 上下文管理器类.
class MyFile(object):
    # 2. 初始化自定义的 上下文管理器类的 属性.
    def __init__(self, file_name, file_model):
        self.file_name = file_name      # 要操作的文件名(路径)
        self.file_model = file_model    # 要操作文件的方式, 例如: r, w, a
        self.file_obj = None            # 初始化文件对象为空.


    # 3. 上文管理器, 即: __enter__() 会在 with语句前, 自动执行. 一般用于初始化: 对象.
    def __enter__(self):
        # 获取具体的文件对象.
        self.file_obj = open(self.file_name, self.file_model, encoding='utf-8')

        # 返回 文件对象.
        return self.file_obj

        # 返回 自定义的 上下文管理器对象.
        # return self

    # 4. 下文管理器, 即: __exit__(), 会在with语句后, 自动执行.
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 释放资源.
        self.file_obj.close()
        # 打印提示语句.
        print('文件对象已被释放!')


# 在main方法中测试
if __name__ == '__main__':
    # 场景1: 上下文管理器类 MyFile, __enter__()返回的是:  上下文管理器对象本身.
    # # 1. 创建文件对象.
    # with MyFile('./1.txt', 'w') as self_obj:
    #     # 2. 获取具体的 文件对象, 往文件中写数据.
    #     self_obj.file_obj.write('我是上下文管理器对象!')


    # 场景2: 上下文管理器类 MyFile, __enter__()返回的是: 文件对象.
    # 1. 创建文件对象.
    with MyFile('./1.txt', 'w') as file_obj:
        # 2. 获取具体的 文件对象, 往文件中写数据.
        file_obj.write('好好学习, 明儿学完正则, 我们就开始学 算法和数据结构!')
