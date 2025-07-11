"""
pymysql 模块解释:
    概述:
        它属于第三方的模块, 用之前需要先安装一下.  它是Python操作MySQL数据库的规范和规则.
        里边定义了一些API(函数), 可以帮助我们实现通过Python操作MySQL, 进行 增删改查的操作.
    安装方式:
        方式1: DOS命令方式, pip install pymysql [-i 镜像地址]
            例如: 清华大学镜像 https://pypi.tuna.tsinghua.edu.cn/simple

        方式2: 导包的时候 安装.
            写完包名后, 按下 alt + enter, 给出建议, 选择: install 包名.
    pymysql的操作步骤:
        1. 获取连接对象.              Python连接MySQL的对象.
        2. 获取游标对象.              可以执行SQL语句的对象.
        3. 执行SQL语句, 获取结果集.
        4. 操作结果集.
        5. 释放资源.

    大白话解释pymysql的步骤:
        1. 找到前台小姐姐.           你(Python) 和 黑马(MySQL)建立了连接, 通过: 前台小姐姐(连接对象)
        2. 前台小姐姐找到 任课老师.    任课老师: 讲解知识点的.
        3. 任课老师给大家(学生)讲课, 获取资料: 视频, 代码, 笔记, 图片, 作业...
        4. 你(学生)操作结果集, 视频: 看, 代码: 敲, 笔记: 整理...
        5. 跟 任课老师, 前台小姐姐 说再见.

细节:
    需要开启MySQL服务, 例如: 小皮.
"""

import pymysql

# 1. 获取连接对象.  6个参数: MySQL所在的主机ip或者主机名, 端口号, 账号, 密码, 库名, 码表
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123456',
    database='day02',
    charset='utf8'
)

# 2. 获取游标对象.
cus = conn.cursor()

# 3. 执行SQL语句, 获取结果集.
sql = 'select * from hero;'
cus.execute(sql)

# 4. 操作结果集.
# 场景1: 从游标对象中, 获取所有的数据. 格式为: 元组嵌套元组, ((1, '鸠摩智', 9), (3, '乔峰', 1)...)
# data = cus.fetchall()

# 场景2: 从游标对象中, 获取第1条数据.
# data = cus.fetchone()

# 场景3: 从游标对象中, 获取n条数据.
data = cus.fetchmany(3)

print(data)

# 5. 释放资源.
cus.close()
conn.close()