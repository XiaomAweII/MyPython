"""
案例: 回顾open语法, 引出 with open 写法, 即: 上下文管理器.
"""

# 场景1: 原生的open()写法.
# # 1. 创建文件对象.
# file_obj = open('./1.txt', 'w', encoding='utf-8')
# # 2. 往文件中写数据.
# file_obj.write('好好学习, 天天向上!')
# # 3. 释放资源.
# file_obj.close()


# 场景2: 上述代码有可能出问题, 如果模式写错了, 直接就报错了, 可以通过 try.except解决.
# try:
#     # 1. 创建文件对象.
#     file_obj = open('./1.txt', 'r', encoding='utf-8')
#     # 2. 往文件中写数据.
#     file_obj.write('好好学习, 天天向上!')
# except:
#     print('程序出问题了!')
# finally:
#     # 3. 释放资源.
#     file_obj.close()
#     print('资源释放了!')


# 场景3: 上述的代码容易遗忘 close(), 导致 文件对象没有关闭, 从而浪费资源.
# 思考: 如何实现, 我们的代码执行完毕后, 自动帮我们回收资源呢?
# 答案: 上下文管理器的方式实现即可, 即: with open()语法.


# 1. 创建文件对象.
with open('./1.txt', 'r', encoding='utf-8') as file_obj:
    # 2. 往文件中写数据.
    file_obj.write('好好学习, 天天向上!')
    # 3. 释放资源, 不需要写了, 代码执行完毕后, 会自动释放.
    # file_obj.close()


# 至此, with open()语法, 需要大家掌握的就讲完了. 接下来, 我们来写个代码, 研究下 with open的底层原理, 看懂即可.