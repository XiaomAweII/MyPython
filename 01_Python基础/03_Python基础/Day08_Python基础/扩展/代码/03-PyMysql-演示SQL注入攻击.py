# 案例: 模拟SQL注入攻击问题.

# 1. 导包.
import pymysql

# 2. 提示用户录入账号 和 密码, 并接收.
uname = input('请录入您的账号: ')
pwd = input('请录入您的密码: ')

# 3. 获取连接对象.
conn = pymysql.connect(
    host='localhost', port=3306, user='root', password='123456', database='day08', charset='utf8'
)

# 4. 获取游标对象.
cursor = conn.cursor()

# 5. 执行SQL语句, 获取结果集.
# SQL语句写法1: 拼接形式.
# sql = f" select * from users where username = '{uname}' and password = '{pwd}';"

# SQL语句写法2: 占位符形式.  %s 字符串, %d 整数, %f 小数
sql = " select * from users where username = '%s' and password = '%s';" % (uname, pwd)
line = cursor.execute(sql)

# 6. 操作结果集, 即: 判断是否登录成功.
print(f'登陆成功, 欢迎您, {uname}' if line >= 1 else '登陆失败, 请检验您的账号和密码!')

# 7. 释放资源.
cursor.close()
conn.close()