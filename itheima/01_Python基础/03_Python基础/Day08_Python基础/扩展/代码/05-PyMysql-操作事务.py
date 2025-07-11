"""
事务简介:
    概述:
        事务指的是 transaction, 指的是逻辑上的一组操作, 组成该操作的各个逻辑单元, 要么全部执行成功, 要么全部执行失败.
        大白话翻译: 同生共死.
    例如:  金柱 转账给 志伟 1000元.
        逻辑单元1: 金柱账号 - 1000
        逻辑单元2: 志伟账号 + 1000
    特点(ACID):
        1. 原子性.
            指的是: 组成事务的各个逻辑单元已经是最小单位, 不可分割.
        2. 一致性.
            指的是: 事务执行前后, 数据应该保持一致.
        3. 隔离性(Isolation)
            指的是: 一个事务在执行期间, 不应该受到其它事务的干扰, 否则容易出现脏读, 不可重复读, 虚读.
        4. 持久性.
            指的是: 无论执行是否成功, 结果都应该永久的存储到数据库中.
    涉及到的SQL:
        show variables like '%commit%';     # 查看MySQL会发现, 它自动开启了事务的提交功能, 即: 每个SQL语句是都一个单独的事务, 会自动提交.
        commit;     提交事务, 即: 相当于保存结果.
        rollback;   事务回滚, 即: 会把数据还原到事务执行前的状态.
"""
import pymysql
# 案例: 模拟转账.
# 场景1: 非事务版.

# # 1. 获取连接对象.
# conn = pymysql.connect(
#     host='localhost', port=3306, user='root', password='123456', database='day08', charset='utf8'
# )
#
# # 2. 获取游标对象.
# cursor = conn.cursor()
#
# # 3. 执行SQL语句, 获取结果集.
# # 3.1 转账动作1: 金柱 - 1000
# sql1 = "update account set money = money - 1000 where name = '金柱';"
# num1 = cursor.execute(sql1)     # 获取受影响行数
# conn.commit()
#
# # 模拟Bug
# print(1 / 0)
#
# # 3.2 转账动作2: 志伟 + 1000
# sql2 = "update account set money = money + 1000 where name = '志伟';"
# num2 = cursor.execute(sql2)     # 获取受影响行数
# conn.commit()
#
# # 4. 操作结果集.
# print('转账成功' if num1 == 1 and num2 == 1 else '转账失败')
#
# # 5. 释放资源.
# cursor.close()
# conn.close()

# 场景2: 加入事务 以及 异常处理后的代码.
try:
    # 1. 获取连接对象.
    conn = pymysql.connect(
        host='localhost', port=3306, user='root', password='123456', database='day08', charset='utf8'
    )

    # 2. 获取游标对象.
    cursor = conn.cursor()

    # 3. 执行SQL语句, 获取结果集.
    # 3.1 开启事务(不写也行, 标准写法, 要写), 表示直至 提交事务 或者 事务回滚前, 都属于 同一组逻辑, 要么全成功, 要么全失败.
    conn.begin()

    # 3.2 转账动作1: 金柱 - 1000
    sql1 = "update account set money = money - 1000 where name = '金柱';"
    num1 = cursor.execute(sql1)     # 获取受影响行数

    # 模拟Bug
    # print(1 / 0)

    # 3.3 转账动作2: 志伟 + 1000
    sql2 = "update account set money = money + 1000 where name = '志伟';"
    num2 = cursor.execute(sql2)     # 获取受影响行数

except Exception as e:
    # print(f'出问题了, 异常原因是: {e}')
    conn.rollback()     # 事务回滚, 相当于: 把数据还原到 事务执行前的状态.
    print('转账失败')
else:
    # 4. 操作结果集, 提交事务.
    conn.commit()
    print('转账成功' if num1 == 1 and num2 == 1 else '转账失败')
finally:
    # 5. 释放资源.
    cursor.close()
    conn.close()