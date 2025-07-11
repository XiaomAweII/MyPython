#### 今日内容大纲介绍

* 异常处理-try-except
* 模块详解
* 包详解
* 综合案例-学生管理系统**(非常重要)**

---

#### 1.异常演示

```python
"""
异常介绍:
    概述:
        在Python中, 我们把程序出现的所有的 非正常情况, 统称为: 异常. 俗称叫: Bug
    常见的异常:
        FileNotFoundError
        除零错误
        ......
    异常的默认处理方式:
        程序会将异常的 类型, 产生原因, 异常出现的位置 打印到控制台上.
        并终止程序的执行.

"""

# 1. 读取了1个不存在的文件.
# src_f = open('1.txt', 'r')      # FileNotFoundError

# 2. 除零异常.
print(10 // 0)                    # ZeroDivisionError
print('看看我执行了吗?')
```



#### 2.捕获异常-入门

```python
"""
捕获异常:
    概述:
        捕获异常这种方式处理完异常之后, 程序会 自动往下 继续执行.
    (基本)格式:
        try:
            可能出问题的代码
        except:
            出问题后的解决方案
    执行流程:
        1. 先执行 try中的内容, 看有无问题. 有问题会立即跳转到 except中执行.
        2. 如果try中内容无问题, 程序会跳过 except, 继续向下执行.
"""

# 需求: 演示 try.except写法.
try:
    print("try 1")
    # 1. 读取了1个不存在的文件.
    src_f = open('1.txt', 'r')      # FileNotFoundError
    print("try 2")
except:
    print('文件不存在, 请校验后重新操作!')

# 2. 除零异常.
# print(10 // 0)                    # ZeroDivisionError
print('看看我执行了吗?')
```



#### 3.捕获异常-Exception

```python
"""
至此, 我们已经初始了 try.except的语法, 接下来我们来看看 它是如何 精准的捕获某个 指定异常的.

格式:
    try:
        可能出问题的代码
    except Exception as e:
        出问题后的解决方案

格式解释:
    Exception   所有异常的父类, 即: 它代表着所有的异常.
    e           类似于我们以前写的变量名, 这里它是: 异常对象名.

细节:
    还可以写成 except (异常1, 异常2) as e    这种写法是捕获多种异常.
"""

# 1. 读取了1个不存在的文件.
# src_f = open('1.txt', 'r')      # FileNotFoundError

# 2. 除零异常.
# print(10 // 0)                    # ZeroDivisionError

# 3. 变量未定义异常.
# print(name)                 # NameError


# 4. 演示 try.except 捕获指定异常.
# 场景1: 捕获 单个异常.
# try:
#     # print(name)   # NameError
#     src_f = open('1.txt', 'r')
# except NameError as e:              # 只能捕获 NameError 异常.
#     print('哎呀, 程序出问题了!')
#     # e就是异常对象, 代表着: 异常信息.  可以直接把它输出到控制台.
#     # print(e)


# 场景2: 捕获 多种异常.
# try:
#     # print(name)                      # NameError
#     # src_f = open('1.txt', 'r')       # FileNotFoundError
#     print(10 // 0)
# except (NameError, FileNotFoundError) as e:
#     # print('哎呀, 程序出问题了!')
#     # e就是异常对象, 代表着: 异常信息.  可以直接把它输出到控制台.
#     print(e)

# 场景3: 通用的 捕获异常的 方案.
try:
    print(name)                      # NameError
    # src_f = open('1.txt', 'r')       # FileNotFoundError
    # print(10 // 0)
except Exception as e:
    # print('哎呀, 程序出问题了!')
    # e就是异常对象, 代表着: 异常信息.  可以直接把它输出到控制台.
    print(e)

print('看看我执行了吗?')

```



#### 4.捕获异常-完整格式

```python
"""
捕获异常, 完整格式如下:
    try:
        里边写的是可能出问题的代码
    except [Exception as e]:
        出现问题后的 解决方案
    else:
        只要try中内容无问题, 就会执行这里的内容.
        只要try中有问题, 就会跳过这里的代码.
    finally:
        无论try是否有问题, 都会走这里的内容, 一般用于释放资源.
"""

# 演示 捕获异常 完整格式.
try:
    print('try 1')
    print(10 // 2)
    print('try 2')
except Exception as e:
    print(f'程序出问题了, {e}')
else:
    print('我是else, 看看我执行了吗?')
finally:
    print('我是finally, 看看我执行了吗?')

print('-' * 28)

# 案例: 拷贝文件, 加入异常处理.
try:
    src_f = open('1.txt', 'rb')
    dest_f = open('aa/bb/cc/2.txt', 'wb')
except Exception as e:
    print(e)
else:
    while True:
        data = src_f.read(1024)
        if len(data) <= 0:
            break
        dest_f.write(data)
finally:
    try:
        src_f.close()
    except Exception as e:
        print(e)

    try:
        dest_f.close()
    except Exception as e:
        print(e)

print('-' * 28)

try:
    with open('1.txt', 'rb') as src_f, open('aa/bb/cc/2.txt', 'wb') as dest_f:
        while True:
            data = src_f.read(1024)
            if len(data) <= 0:
                break
            dest_f.write(data)
except Exception as e:
    print(e)
```



#### 5.异常的传递性

```python
"""
异常是具有传递性的, 函数内的异常 会传递给该函数的 调用者, 逐级传递, 直至这个异常被处理, 或者传递到main还不处理, 程序就会报错.
"""


# 案例: 演示异常的传递性.
def fun01():
    print('----- fun01 start-----')
    print(10 // 0)
    print('----- fun01 end-----')


def fun02():
    print('----- fun02 start-----')

    # 调用 fun01()
    fun01()
    # try:
    #     fun01()
    # except:
    #     print('除数不能为0')

    print('----- fun02 end-----')


def fun03():
    print('----- fun03 start-----')
    # 调用 fun02()
    fun02()
    print('----- fun03 end-----')


if __name__ == '__main__':
    try:
        fun03()
    except:
        print('除数不能为0')


```



#### 6.模块-导入方式详解

```python
"""
模块介绍:
    概述:
        模块指的是 Module, 在Python中, 1个.py文件 = 1个模块.
        你可以把 模块理解为 工具包, 工具包中有很多的工具.  其实就是: 每个.py文件中都有很多的 函数, 这些函数都有不同的功能.
    大白话:
        学模块, 就是记忆一些 .py文件, 以及其中的一些 函数.
        例如: 随机数相关 用 random模块,   日期相关用 time模块,  文件路径相关用 os模块...

    模块的 导入方式:
        方式1: import 模块名                         后续通过  模块名.函数名() 的方式调用.   模块下所有函数 均可使用
        方式2: import 模块名 as 别名                 后续通过  别名.函数名() 的方式调用.     模块下所有函数 均可使用
        方式3: from 模块名 import 函数名            后续通过  函数名() 的方式直接调用.       只能使用该模块下 导入的函数.
        方式4: from 模块名 import 函数名 as 别名    后续通过  别名() 的方式直接调用.         只能使用该模块下 导入的函数.
        方式5: from 模块名 import *               后续通过  函数名() 的方式直接调用.       模块下所有函数 均可使用
"""

# 案例: 演示 导入模块的几种方式的 用法.
# 测试用例.   time模块下的 time()函数, sleep()函数.


# 演示 方式1: import 模块名                         后续通过  模块名.函数名() 的方式调用.   模块下所有函数 均可使用
# import time
#
# print('--- start ---')
# time.sleep(2)               # 让程序休眠 2秒.
# print(time.localtime())     # 获取系统的的本地时间
# print(time.time())          # 1719113316.0237932,  从时间原点(1970年1月1日 00:00:00 ~ 至今)的秒值.
# print('--- end ---')

# 演示 方式2: import 模块名 as 别名                 后续通过  别名.函数名() 的方式调用.     模块下所有函数 均可使用
# import time as t
#
# print('--- start ---')
# t.sleep(2)               # 让程序休眠 2秒.
# print(t.localtime())     # 获取系统的的本地时间
# print(t.time())          # 1719113316.0237932,  从时间原点(1970年1月1日 00:00:00 ~ 至今)的秒值.
# print('--- end ---')


# 演示 方式3: from 模块名 import 函数名            后续通过  函数名() 的方式直接调用.       只能使用该模块下 导入的函数.
# from time import sleep, localtime

# print('--- start ---')
# sleep(2)               # 让程序休眠 2秒.
# print(localtime())     # 获取系统的的本地时间
# # print(time())          # 报错, 因为没有导入, 所以使用不了.
# print('--- end ---')

# 演示 方式4: from 模块名 import 函数名 as 别名    后续通过  别名() 的方式直接调用.         只能使用该模块下 导入的函数.
# from time import sleep as sl, localtime as lt
#
# print('--- start ---')
# sl(2)               # 让程序休眠 2秒.
# print(lt())     # 获取系统的的本地时间
# # print(time())          # 报错, 因为没有导入, 所以使用不了.
# print('--- end ---')


# 演示 方式5: from 模块名 import *               后续通过  函数名() 的方式直接调用.       模块下所有函数 均可使用
from time import *

print('--- start ---')
sleep(2)               # 让程序休眠 2秒.
print(localtime())     # 获取系统的的本地时间
print(time())
print('--- end ---')


```



#### 7.测试调用自定义模块

* my_module1 模块的代码

  ```python
  # 这个是我们自定义的第 1个 模块.
  
  # 如果不写all属性, 表示: 导入模块中 所有的函数.
  __all__ = ['fun01', 'fun02']
  
  # 函数 = 模块的功能, 相当于: 工具包中的工具
  def get_sum(a, b):
      print('我是 my_module1 模块的 函数')
      print(__name__)
      return a + b
  
  
  def fun01():
      print('我是 my_module1 模块的 函数')
      print('----- fun01 函数 -----')
      print(__name__)
  
  
  def fun02():
      print('我是 my_module1 模块的 函数')
      print('----- fun02 函数 -----')
      print(__name__)
  
  
  # 实际开发中, 定义好模块后, 一般会对模块的功能(函数)做测试.
  # 如下的测试代码, 在 调用者中 也会被执行. 但真实的业务场景, 测试代码 在 调用者中是不能被执行的.
  # 那, 如何解决这个问题呢?
  # 答案: __name__ 属性 即可解决这个事儿, 它在当前模块中打印的结果是 __main__,
  #      在调用者模块中打印的是 当前的模块名.
  
  # # 测试代码
  # print(get_sum(10, 20))
  # fun01()
  # fun02()
  
  # 如果条件成立, 说明是在 当前模块中执行的.
  if __name__ == '__main__':
      # 测试代码
      print(get_sum(10, 20))
      fun01()
      fun02()
  ```

* my_module2模块的代码

  ```python
  # 这个是我们自定义的第 2个 模块.
  
  # 函数 = 模块的功能, 相当于: 工具包中的工具
  def get_sum(a, b):
      print('我是 my_module2 模块的 函数')
      print(__name__)
      return a + b
  
  
  def fun01():
      print('我是 my_module2 模块的 函数')
      print('----- fun01 函数 -----')
      print(__name__)
  
  
  def fun02():
      print('我是 my_module2 模块的 函数')
      print('----- fun02 函数 -----')
      print(__name__)
  
  
  
  # 如果条件成立, 说明是在 当前模块中执行的.
  if __name__ == '__main__':
      # 测试代码
      print(get_sum(11, 22))
      fun01()
      fun02()
  ```

* 测试模块的代码

  ```python
  """
  案例: 演示如何调用 自定义模块.
  
  细节:
      1.  1个.py文件就可以看做是1个模块, 文件名 = 模块名, 所以: 文件名也要符合标识符的命名规范.
      2.  __name__属性, 当前模块中打印的结果是 __main__, 在调用者中打印的结果是: 调用的模块名
      3. 如果导入的多个模块中, 有同名函数, 默认会使用 最后导入的 模块的函数.
      4. __all__ 属性只针对于 from 模块名 import * 这种写法有效, 它只会导入 __all__记录的内容
  """
  
  # 需求: 自定义 my_module1模块, 然后再其中定义一些函数.  在当前模块中 调用 my_module1模块的内容.
  # import my_module1 as m1
  # import my_module1 as m2
  #
  # # print(m1.get_sum(10, 20))
  # # m1.fun01()
  # # m1.fun02()
  #
  # m1.fun01()
  # m2.fun01()
  
  
  # 如果导入的多个模块中, 有同名函数, 默认会使用 最后导入的 模块的函数.
  # from my_module1 import fun01
  # from my_module2 import fun01
  #
  # fun01()
  
  
  # __all__ 属性只针对于 from 模块名 import * 这种写法有效, 它只会导入 __all__记录的内容
  from my_module1 import *
  
  # print(get_sum(10, 20))  # 报错, 因为 __all__ 属性没有记录它.
  fun01()
  fun02()
  
  ```


#### 8.测试-包

* 结构图

  ![1719140235061](assets/1719140235061.png)

* init文件的代码

  ```python
  
  # all属性 如果不写, 默认是: 包中所有的 模块, 都不能通过 from 包名 import * 的方式导入.
  __all__ = ['my_module1']
  ```

* my_module1 和 my_module2模块代码同上.

* 测试代码

  ```python
  """
  包 解释:
      概述:
          包 = 文件夹 = 一堆的.py文件(模块) +  __init__.py 初始化文件.
  
      背景:
          当我们的模块(.py文件)越来越多的时候, 就要分包来管理它们了(模块).
  
      导包方式:
          方式1: import 包名.模块名
              必须通过 包名.模块名.函数名() 的方式, 来调用 函数.
          方式2: from 包名 import 模块名
              必须通过 模块名.包名()的形式, 来调用函数.
  """
  
  # 演示 导入包的 方式1: import 包名.模块名
  # import my_package.my_module1
  #
  # # 包名        模块       函数名
  # my_package.my_module1.fun01()
  # my_package.my_module1.fun02()
  # print('-' * 28)
  
  
  # 演示 导入包的 方式2: from 包名 import 模块名
  # from my_package import my_module1
  #
  # my_module1.fun01()
  # my_module1.fun02()
  
  
  from my_package import *        # 会去 init.py文件 初始化文件中, 只加载 all属性的信息.
  my_module1.fun01()
  
  # my_module2.fun01()      # 报错, 因为 from 包名 import * 的时候, 只会到 __init__.py文件中 all属性的内容.
  
  ```



#### 9.学生管理系统

```python
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


```



#### 