# 导包.
import pymysql

# 1.定义函数, 完成PyMysql的: 增
def add_data():
    # 1. 获取连接对象.
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='123456', database='day08', charset='utf8')
    # print(conn)
    # 2. 根据连接对象, 获取游标对象.
    cursor = conn.cursor()
    # 3. 执行SQL语句, 获取结果集.
    sql = "INSERT INTO product values(null,'传智播客',66666,'c005');"
    lines = cursor.execute(sql)
    # 核心细节: (更新语句, 增, 删, 改)必须提交才有效.
    conn.commit()
    # 5. 操作结果集.
    print('添加成功' if lines > 0 else '添加失败')
    # 6. 释放资源
    cursor.close()
    conn.close()

# 2.定义函数, 完成PyMysql的: 删
def delete_data():
    # 1. 获取连接对象.
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='123456', database='day08',
                           charset='utf8')
    # print(conn)
    # 2. 根据连接对象, 获取游标对象.
    cursor = conn.cursor()
    # 3. 执行SQL语句, 获取结果集.
    sql = "delete from product where pid = 15;"
    lines = cursor.execute(sql)
    # 核心细节: (更新语句, 增, 删, 改)必须提交才有效.
    conn.commit()
    # 5. 操作结果集.
    print('删除成功' if lines > 0 else '删除失败')
    # 6. 释放资源
    cursor.close()
    conn.close()

# 3.定义函数, 完成PyMysql的: 改
def update_data():
    # 1. 获取连接对象.
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='123456', database='day08',
                           charset='utf8')
    # print(conn)
    # 2. 根据连接对象, 获取游标对象.
    cursor = conn.cursor()
    # 3. 执行SQL语句, 获取结果集.
    sql = "update product set pname='黑马', price=60000 where pid = 15;"
    lines = cursor.execute(sql)
    # 核心细节: (更新语句, 增, 删, 改)必须提交才有效.
    conn.commit()
    # 5. 操作结果集.
    print('修改成功' if lines > 0 else '修改失败')
    # 6. 释放资源
    cursor.close()
    conn.close()

# 4.定义函数, 完成PyMysql的: 查
def query_data():
    # 1. 获取连接对象.
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='123456', database='day08',
                           charset='utf8')
    # print(conn)
    # 2. 根据连接对象, 获取游标对象.
    cursor = conn.cursor()
    # 3. 执行SQL语句, 获取结果集.
    sql = "select * from product where pid >= 3"
    cursor.execute(sql)
    result = cursor.fetchall()      # 读取所有数据.
    # 5. 操作结果集.
    for line in result:
        print(line)
    # 6. 释放资源
    cursor.close()
    conn.close()

# 5. 在main函数中测试上述的功能.
if __name__ == '__main__':
    # 增
    # add_data()
    # 删
    # delete_data()
    # 改
    # update_data()
    # 查
    query_data()
