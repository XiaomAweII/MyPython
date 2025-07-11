/*
SQL的通用语法:
    1. SQL语句可以写一行, 也可以写多行, 以 分号 结尾.
    2. SQL语句不分区大小写, 建议: 关键字大写,其它小写.
    3. 为了增加阅读性, 可以加入空格, 换行等操作SQL语句.
    4. SQL语句注释写法:
        /星
            多行
            注释
        星/

        -- 单行注释, 注意: --后边 必须加 空格.
        # 单行注释, 注意: #后边 可以不加空格

SQL常用的数据类型:
    整型: int
    浮点型: double, decimal
    字符型: varchar(n), char(n)        前者: 变长,  后者: 定长(不够用空格补齐)
    日期型: date, datetime
 */
 -- 我是单行注释
 # 我是单行注释

# --------------------- 案例1: DDL语句操作数据库-增删改查(CURD) ---------------------
# Create: 增, Update: 改, Read: 查, Delete: 删
# 1. 查看所有的数据库. 格式: show databases;
show databases;                 # 查看所有的数据库
show create database day01;     # 可以查看该(当前数据库)的码表.

# 2. 创建数据库.      格式: create database 数据库名 charset '码表';
create database day01;                  # 采用默认码表创建数据库.
create database day02 charset 'utf8';   # 采用 utf8码表创建数据库
create database day03 charset 'gbk';    # 采用 gbk码表创建数据库
create database if not exists day01;    # 采用默认码表创建数据库, 如果不存在, 就创建数据表. 如果存在, 就啥都不做.

# 3. 修改数据库(的码表)
alter database day03 charset 'utf8';        # 修改day03数据库的码表为: u8

# 4. 删除数据库, 格式: drop database 数据库名;
drop database day02;
drop database day03;

# 5. 切换数据库.
use day01;      # 切库, 表示后续的操作, 都是在当前这个库中完成的.

# 6. 查看当前使用的是 哪个数据库.
select database();      # day01

# 7. 查看数据库版本.
select version();       # 8.0.12


# --------------------- 案例2: DDL语句操作数据表-增删改查(CURD) ---------------------
# 前提: 先切库, 后续操作表, 都是在 数据库中操作的.
# 1. 查看数据表.
show tables;        # 查看(当前数据库中)所有数据表
desc student;       # 查看某张表的字段信息(字段名, 类型, 约束)

# 2. 创建数据表.
/*
格式:
    create table [if not exists] 数据表名(
        字段名 数据类型 [约束],
        字段名 数据类型 [约束],
        ......
        字段名 数据类型 [约束]
    );
细节:
    上述的中括号的内容是 可选项, 写不写都行.
*/
create table if not exists student(
    sid int not null,       # 学生id, int类型, 非空约束
    name varchar(20),       # 学生姓名, 字符串类型
    gender varchar(10),     # 学生性别.
    age int                 # 学生年龄
);

# 3. 修改数据表的 名字.  格式: alter table 旧表名 rename to 新表名;
alter table student rename to stu;

# 4. 删除数据表.
drop table stu;


# --------------------- 案例3: DDL语句操作字段-增删改查(CURD) ---------------------
# 1. 查看表结构.
desc stu;

# 2. 修改表名.
alter table student rename to stu;
rename table student to stu;            # 效果同上.

# 3. 给表 新增1列.  格式: alter table 表名 add 新列名 数据类型 [约束];
alter table stu add address varchar(10) not null;

# 4. 修改字段: 数据类型, 约束.     address varchar(10) => address double
# 格式: alter table 表名 modify 旧列名 数据类型 [约束];
alter table stu modify address double;

# 5. 修改字段: 列名, 数据类型, 约束.    address double => desc char(10) 非空约束
# 格式: alter table 表名 change 旧列名 新列名 数据类型 [约束];
alter table stu change address `desc` char(10) not null;      # 列名如果和关键字重名了, 要用 反引号``包裹.

# 6. 删除某列.  例如: 删除 desc 列
# 格式: alter table 表名 drop 旧列名;
alter table stu drop `desc`;








