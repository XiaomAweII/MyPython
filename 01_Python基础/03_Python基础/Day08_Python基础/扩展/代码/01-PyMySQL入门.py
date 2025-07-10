"""
PyMysql介绍:
    概述:
        它是用Python写的一个包, 主要是实现 通过Python语言来操作各种数据库的.
    操作步骤:
        1. 导包.
        2. 获取连接对象.
        3. 获取游标对象, 可以执行SQL语句.
        4. 执行SQL语句.
        5. 获取结果集.
        6. 操作结果集.
        7. 释放资源.
    大白话解释:      Python: 你,   MySQL: 咱们公司, PyMysql = 你要来咱们公司学习.
        1. 打听到 黑马北京昌平校区地址.
        2. 来到校区, 找到前台小姐姐.
        3. 前台小姐姐帮你找个技术老师, 可以传授你技术.
        4. (技术老师)具体的传授技术(讲课)的动作.
        5. 获取到老师给的资料(代码, 笔记, 视频, 图片等...).
        6. 你处理资料, 视频: 看, 代码: 敲,练, 图片: 看
        7. 跟技术老师, 前台小姐姐说再见.


    pymysql包需要我们额外安装, 具体安装方式如下.
        方式1: 导包方式安装.
        方式2: 在python编辑器(anaconda)中安装
        方式3: 在windows的dos窗口中安装
"""
# 1. 导包.
import pymysql

# 2. 获取连接对象.
conn = pymysql.connect(
    host='localhost',       # 要连接到的机器的ip地址或者主机名
    port=3306,              # 端口号
    user='root',            # 数据库的账号
    password='123456',      # 数据库的密码
    database='day08',       # 要连接到的具体的数据库名
    charset='utf8'          # 码表
)
# print(conn)     # 不报错, 能看到地址值, 连接成功.

# 3. 获取游标对象, 可以执行SQL语句.
cursor = conn.cursor()

# 4. 执行SQL语句.
sql = "select * from product;"
rows = cursor.execute(sql)
print(f'受到的影响行数是: {rows} 行')

# 5. 获取结果集.
# result = cursor.fetchone()      # 一次读取一行
# result = cursor.fetchall()      # 一次读取所有行
result = cursor.fetchmany(3)      # 一次读取指定行

# 6. 操作结果集.
# print(result)   # 会把每行数据封装成1个元组, 如果是多行, 则再把所有元组放到一个元组中.
for line in result:
    # print(line[1], line[2])
    print(line)

# 7. 释放资源.
cursor.close()
conn.close()