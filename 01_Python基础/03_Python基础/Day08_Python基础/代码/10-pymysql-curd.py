# 案例: 演示PyMySQL操作 MySQL数据库, 进行 CURD 增删改查操作.
import pymysql

# 1. pymysql模块 操作 mysql数据库, 增
def insert_method():
    # 1. 获取连接对象.
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='123456', database='day02',
                           charset='utf8')
    # 2. 根据连接对象, 获取 游标对象.
    cur = conn.cursor()
    # 3. 执行SQL语句, 获取结果集.
    sql = 'insert into hero values(6, "杨过", 20);'
    n = cur.execute(sql)        # n就是受到影响的行数, 例如: 增了几行, 删了几行, 改了几行.
    # 核心操作: 增, 删, 改属于更新语句, 操作之后必须 commit()提交, 才会保存结果.
    conn.commit()
    # 4. 操作结果集.
    # if n > 0:
    #     print('添加成功!')
    # else:
    #     print('添加失败!')
    print('添加成功' if n > 0 else '添加失败!')
    # 5. 释放资源.
    cur.close()
    conn.close()

# 2. pymysql模块 操作 mysql数据库, 删
def delete_method():
    # 1. 获取连接对象.
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='123456', database='day02',
                           charset='utf8')
    # 2. 根据连接对象, 获取 游标对象.
    cur = conn.cursor()
    # 3. 执行SQL语句, 获取结果集.
    sql = 'delete from hero where hid > 3;'
    n = cur.execute(sql)  # n就是受到影响的行数, 例如: 增了几行, 删了几行, 改了几行.
    # 核心操作: 增, 删, 改属于更新语句, 操作之后必须 commit()提交, 才会保存结果.
    conn.commit()
    # 4. 操作结果集.
    print('删除成功' if n > 0 else '删除失败!')
    # 5. 释放资源.
    cur.close()
    conn.close()

# 3. pymysql模块 操作 mysql数据库, 改
def update_method():
    # 1. 获取连接对象.
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='123456', database='day02',
                           charset='utf8')
    # 2. 根据连接对象, 获取 游标对象.
    cur = conn.cursor()
    # 3. 执行SQL语句, 获取结果集.
    sql = 'update hero set hname="神雕侠", kongfu_id=100 where hid=6;'
    n = cur.execute(sql)  # n就是受到影响的行数, 例如: 增了几行, 删了几行, 改了几行.
    # 核心操作: 增, 删, 改属于更新语句, 操作之后必须 commit()提交, 才会保存结果.
    conn.commit()
    # 4. 操作结果集.
    print('修改成功' if n > 0 else '修改失败!')
    # 5. 释放资源.
    cur.close()
    conn.close()

# 4. pymysql模块 操作 mysql数据库, 查
def query_method():
    # 1. 获取连接对象.
    conn =  pymysql.connect(host='localhost', port=3306, user='root', password='123456', database='day02', charset='utf8')
    # 2. 根据连接对象, 获取 游标对象.
    cur = conn.cursor()
    # 3. 执行SQL语句, 获取结果集.
    sql = 'select * from hero;'
    cur.execute(sql)
    # 4. 操作结果集.
    data = cur.fetchall()
    for line in data:
        print(line)
    # 5. 释放资源.
    cur.close()
    conn.close()


# 5. 在main方法中, 测试调用:
if __name__ == '__main__':
    # 添加 表数据
    # insert_method()

    # 修改 表数据
    # update_method()

    # 删除 表数据
    delete_method()

    # 查看 表数据.
    query_method()