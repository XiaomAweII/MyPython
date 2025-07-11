"""
SQL注入攻击问题解决思路:
    采用占位符的思想解决, 即: 预先用占位符来填充SQL语句中变化的地方, 然后对整个SQL语句完成预编译的动作,
    即: 在预编译的时候已经决定了SQL语句的格式是什么, 之后无论传入什么字符, 都只会当做普通的字符来处理.
"""

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
# SQL语句写法: 占位符形式. 注意: 不要给 %s加单引号
sql = " select * from users where username = %s and password = %s;"     # 此时, 已经预先对SQL语句格式进行了编译, 之后无论传入什么, 都只当普通字符来处理.
# 核心细节: 用容器类型(例如: 元组, 列表), 用来记录 要给占位符填充的值.
params = [uname, pwd]
# 执行SQL语句时, 参1: 要被执行的SQL语句,  参2: 要给占位符填充值的 容器对象
line = cursor.execute(sql, params)

# 6. 操作结果集, 即: 判断是否登录成功.
print(f'登陆成功, 欢迎您, {uname}' if line >= 1 else '登陆失败, 请检验您的账号和密码!')

# 7. 释放资源.
cursor.close()
conn.close()