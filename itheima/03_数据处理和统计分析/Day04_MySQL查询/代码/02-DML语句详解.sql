/*
DML语句介绍:
    概述:
        它主要是用来 操作 表数据的, 进行: 增, 删, 改 的操作.
        增删改统称为: 更新语句.
    关键词:
        insert, delete, update
*/

# --------------------- 案例1: DML语句操作表数据-增 ---------------------
/*
添加表数据, 格式如下:
    一次添加1条数据:
        insert into 数据表名(列名1, 列名2...) values(值1, 值2...);
        上述格式的语法糖, 如果是操作全列, 则: 可以简写成如下的写法.
            insert into 数据表名 values(值1, 值2...);
    一次性添加多条数据:
        insert into 数据表名(列名1, 列名2...)
        values
            (值1, 值2...),
            (值1, 值2...),
            ......
            (值1, 值2...);
细节:
    1. 列名 和 值,  个数, 对应类型等均要匹配.
    2. 如果不写列名, 默认是: 全列名
    3. 传值的时候, 数字可以直接写(小数, 整数均可), 其它要用单引号包裹.
*/
# 1. 切库
use day01;
# 2. 查看所有数据表.
show tables;
# 3. 建表, 如果表不存在.
create table if not exists stu(
    sid int not null,       # 学生id, int类型, 非空约束
    name varchar(20),       # 学生姓名, 字符串类型
    gender varchar(10),     # 学生性别.
    age int                 # 学生年龄
);

# 4. 添加表数据
# 添加 单条数据, 1, 张三, 男, 23
insert into stu(sid, name, gender, age) values(1, '张三', '男', 23);
# 添加 单条数据, 2, 李四, 男, 24
insert into stu values(2, '李四', '男', 24);   # 语法糖, 不写列名, 默认是: 全列名.
# 添加 单条数据, 3, 王五, 25
insert into stu(sid, name, age) values(3, '王五', 25);
insert into stu(name, age) values('赵六', 26);        # 报错, sid列有非空约束, 必须传值, 不传就报错.

# 添加多条数据,  (4, '阿朱', '女', 25), (5, '李清露', '女', 23), (6, '王语嫣', '女', 21)
insert into stu values
    (4, '阿朱', '女', 25),
    (5, '李清露', '女', 23),
    (6, '王语嫣', '女', 21);

# 5. 查看表数据.  格式: select * from 数据表名;
select * from stu;



# --------------------- 案例2: DML语句操作表数据-改 ---------------------
# 格式: update 表名 set 列名1=新值, 列名2=新值, 列名3=新值... where 条件;
# 细节: 1. 改值的时候, 数据类型, 个数要匹配.    2. 进行修改,删除操作时, 一定一定一定要加where条件, 不然就是针对于全表数据做操作.
# 1. 查看表数据.
select * from stu;

# 2. 修改表数据 李四, 男, 24 => 杨过, 男, 31
update stu set name='杨过', age=31 where sid=2;

# 3. 一个非常危险的 "坐牢"命令.
update stu set name='杨过', age=31;       # 不加where条件, 则会一次性更新表中所有的数据.


# --------------------- 案例3: DML语句操作表数据-删 ---------------------
/*
格式:
    delete from 表名 where 条件;
细节:
    1. 删除数据时, 一定一定一定要加where条件.
    2. delete from 和 truncate table 都可以一次性清空表数据, 那么它们之间有什么区别?
        delete from:
            属于DML语句, 用于删除表数据的, 不会重置: 主键id.
            可以结合事务一起使用.
        truncate table:
            属于DDL语句, 用于"清空"表数据的, 它相当于把表摧毁了, 然后创建1张和该表一模一样的新表.
            即: 会重置之间id, 一般不结合 事务 一起使用.
*/
# 1. 查看表数据
select * from stu;

# 2. 删除 sid为奇数的 数据.
delete from stu where sid % 2 != 0;

# 3. 一次性删除所有的数据.
delete from stu;
truncate table stu;     # table 还可以省略不写.
truncate stu;           # 效果同上.


# --------------------- 案例4: DML语句操作表数据-删-是否会重置主键id ---------------------
# 1. 建表, 带: 主键约束(特点: 非空, 唯一), 且结合自动增长一起使用.
drop table student;
create table student(
    sid int primary key auto_increment,     # 学生id, 主键约束(非空, 唯一), 自动增长(只针对于数值有效)
    name varchar(20),   # 姓名
    age int             # 年龄
);

# 2. 查看表数据.
desc student;
select * from student;

# 3. 添加表数据.
insert into student values(1, '张三', 23);
insert into student values(1, '李四', 24);        # 报错, 主键约束: 具有唯一性.
insert into student values(null, '李四', 24);     # 报错, 主键约束: 具有非空性. 除非结合 自增 一起使用, 即: 不报错了

# 4. 两种删除的区别.
delete from student where sid >= 8;
delete from student;        # 仅仅是清空表数据, 不会重置主键id
truncate student;           # 会重置主键id.


# --------------------- 案例5: DML语句操作表数据-快速备份表数据 ---------------------
# 1. 查看表数据(即: 源表)
insert into student values(null, '王五', 25), (null, '赵六', 26);
select * from student;

# 2. 查看所有的数据表.
show tables;

# 3. 备份数据表.
# 场景1: 备份表不存在, 例如:  源表 student => 备份表 student_tmp
# 格式: create table 备份表名 select * from 原表名 where 条件;
create table student_tmp select * from student;

# 场景2: 备份表存在, 例如:  源表 student => 备份表 student_tmp
insert into student_tmp select * from student;

# 4. 查看备份表的数据.
truncate student_tmp;
select * from student_tmp;

# 5. 查看表结构.
desc student;       # 源表结构
desc student_tmp;   # 备份表结构


# --------------------- 案例6: 约束详解 ---------------------
/*
约束 介绍:
    概述:
        约束就使用来保证 数据的 完整性, 安全性 和 一致性的.
    分类:
        单表约束:
            主键约束: primary key
                特点: 非空, 唯一, 且一般结合自增(自动增长 auto_increment)一起使用.
            非空约束: not null
                特点: 不能为null, 可以重复.
            唯一约束: unique
                特点: 可以为空, 不能重复.
            默认约束: default
                特点: 如果添加数据时没有给该列指定值, 则用默认值.
        多表约束:
            外键约束: foreign key
                特点: 外表的外键列 不能出现 主表的主键列 没有的数据.
*/
# 1. 建表, 英雄表hero(hid, name, age, phone, address)
create table hero(
    id int primary key auto_increment,  # id, 主键约束(非空, 唯一), 自增
    name varchar(20) not null,          # 名字, 非空约束.
    age int,                            # 年龄
    phone varchar(11) unique ,          # 手机号, 唯一约束.
    address varchar(10) default '北京'  # 住址, 默认约束
);

# 2. 查看表数据.
select * from hero;

# 3. 添加表数据.
insert into hero values(null, '乔峰', 39, '111', '上海');       # 对
insert into hero values(null, '乔峰', 39, '222', '广州');       # 对
insert into hero values(null, '虚竹', 31, null, '广州');        # 对
insert into hero values(null, '段誉', 29, null, '大理');        # 对
insert into hero(id, name) values(null, '小龙女');              # 对

insert into hero values(null, null, 23, '111', '上海');        # 报错, 姓名不能为空
insert into hero values(null, '乔峰', 39, '111', '广州');       # 报错, 手机号具有唯一性.
insert into hero values(null, '杨过', 26, null);               # 报错, 值的个数 和 列的个数不匹配
