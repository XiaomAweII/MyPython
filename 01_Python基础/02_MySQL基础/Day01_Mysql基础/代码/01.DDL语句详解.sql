# -------------------- DDL语句 操作数据库 --------------------
-- 1. 查看所有的数据库
show databases;

-- 格式: create database if not exists 数据库名 charset '码表名';
-- 2. 创建数据库, 采用默认码表.
create database day01;                  -- 默认码表: utf8

-- 3. 创建数据库, 采用: 指定码表.
create database day02 charset 'gbk';    -- 指定码表: gbk


-- 4. 创建数据库, 如果数据库不存在.   (掌握)
create database if not exists day01;    -- day01数据库不存在就创建, 存在就: 啥也不做.

-- 5. 查看(单个)数据库的详细信息(码表)
show create database day01;
show create database day02;

-- 6. 修改数据库的(码表).
alter database day02 charset 'utf8';

-- 7. 删除指定的数据库.
-- 格式: drop database 数据库名;
# drop database day02;
# drop database day03;

-- 8. 切库.    (掌握)
use day01;

-- 9. 查看当前使用的是哪个数据库.
select database();




# -------------------- DDL语句 操作数据表 --------------------
-- 0. 切库.
use day01;

-- 1. 查看(当前数据库中)所有的数据表
show tables;

-- 2. 创建数据表.    (掌握)
/*
格式:
    create table [if not exists] 数据表名(
        列名1 数据类型 [约束],
        列名2 数据类型 [约束],
        列名3 数据类型 [约束],
        ...
        列名n 数据类型 [约束]       # 最后1个列名的结尾, 没有逗号.
    );
 */
-- 需求: 创建学生表student, 字段(学生编号, 姓名 非空约束, 性别, 年龄)
create table if not exists student(
    id int,                      # 学生编号
    name varchar(20) not null,   # 学生姓名, 非空
    gender varchar(10),          # 学生性别
    age int                      # 学生年龄
);

-- 3. 查看(单个)数据表的详细信息(码表)
show create table student;

# 需要你记忆的格式.   (掌握)
desc student;       # 查看表的字段(有哪些字段, 每个字段是什么数据类型, 约束等...),  describe: 描述.
desc stu;

-- 4. 修改数据表的(表名).
-- 格式: rename table 旧表名 to 新表名;
rename table student to stu;

-- 5. 删除指定的数据表.
drop table stu;



# -------------------- DDL语句 操作字段(了解) --------------------
-- 1. 查看表的所有(列)
desc student;

-- 2. 给表 新增列, 这个语句还有可能会用到, 理解即可.
-- 格式: alter table 表名 add 列名 数据类型 [约束];
alter table student add kongfu varchar(20) not null;

-- 3. 修改列, 数据类型 和 约束.    kongfu列, varchar(20) -> int
-- 格式: alter table 表名 modify 列名 数据类型 [约束];
alter table student modify kongfu int ;      -- 不加约束, 则会取消之前的约束.

-- 4. 修改列, 列名, 数据类型 和 约束.   kongfu -> gf varchar(10) 非空
-- 格式: alter table 表名 change 旧列名 新列名 数据类型 [约束];
alter table student change kongfu gf varchar(10) not null;

-- 5. 删除指定的列.
-- 格式: alter table 表名 drop 列名;
alter table student drop gf;






