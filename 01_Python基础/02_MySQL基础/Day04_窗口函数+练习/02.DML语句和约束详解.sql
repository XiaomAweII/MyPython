-- DML语句: 数据操作语言, 主要是对表数据进行更新操作的, 即: 增删改.
-- 关键字: insert, delete, update

-- 切库.
use day01;

# -------------------- 案例1: DML语句 操作数据表 增 --------------------
/*
格式:
    insert into 表名(列名1, 列名2...) values(值1, 值2...);
    insert into 表名 values(值1, 值2...);
    insert into 表名 values(null, 值2...);

    insert into 表名 values
        (值1, 值2...),
        (值1, 值2...),
        ......
        (值1, 值2...);

细节:
    1. 添加表数据的时候, 值的个数和类型, 要和 列名保持一致.
    2. 如果不写列名, 则默认是全列名, 即: 必须给所有的列依次赋值.
    3. 如果是主键列且配合有自增, 则: 传值的时候, 可以直接传入null, 系统会根据最大主键值 +1, 然后存储.
    4. 如果同时添加多行值, 多组值之间用逗号隔开, 最后一组值的后边写分号.
*/
-- 1. 添加单条数据.
insert into student(id, name, gender, age) values(1, '乔峰', '男', 33);
insert into student(name, age) value('虚竹', 30);
insert into student(gender, age) value('男', 21);    -- 报错, 姓名不能为空
insert into student(name,gender, age) value('段誉', '男', 21);    -- 正确

-- 2. 同时添加多条数据.
insert into student values
    (1, '阿朱', '女', 30),
    (2, '李清露', '女', 24),
    (3, '王语嫣', '女', 19);

-- 注意: 在进行修改或者删除操作的时候, 一定一定一定要加where条件, 一个老屌丝的含泪忠告.
# -------------------- 案例2: DML语句 操作数据表 改 --------------------
-- 格式: update 表名 set 字段名1=新值, 字段名2=新值... where 条件;
update student set id=10, name='萧峰' where id = 1;

-- 如果不写where条件, 一下子改变所有.
update student set id=1, name='无崖子';


# -------------------- 案例3: DML语句 操作数据表 删 --------------------
-- 格式: delete from 表名 where 条件;
delete from student where age > 25;

-- 不写where条件, 一次删除所有.
delete from student;

-- 还有一个删除语句叫: truncate table 表名, 明天详解, 要结合 主键约束, 才会更好的演示.

# -------------------- 扩展: DQL语句 操作数据表 查 --------------------
-- 简单查询, 查询表中所有的数据.
-- select * from day01.student;        # 数据库名.数据表名
select * from student;              # 如果直接写数据表用, 默认用的是 当前库中的表.


# -------------------- 约束详解(掌握) 主键约束 --------------------
/*
约束:
    概述:
        就是在数据类型的基础上, 对某列值进一步做限定, 例如: 非空, 唯一等...
    目的:
        保证数据的完整性 和 安全性.
    分类:
        单表约束:
            主键约束: primary key, 一般结合自增 auto_increment一起使用.  特点: 非空, 唯一, 一般是数字列.
            非空约束: not null
            唯一约束: unique
            默认约束: default, 如果我们不给值, 则会用默认值填充.
        多表约束:
            外键约束 foreign key
*/
-- 1. 建库, 切库, 查表.
drop database if exists day02;  -- 如果存在, 就删除day02数据库.
create database day02;
use day02;
show tables;

-- 2. 创建学生表, 字段(id, name, gender, age)
drop table  student;
create table student(
    id int primary key auto_increment,     # 学生id, 主键(非空, 唯一), 自增
    name varchar(20),       # 学生姓名
    gender varchar(10),     # 学生性别
    age int                 # 学生年龄
);

-- 3. 给学生表添加数据.
insert into student values(1, '萧炎', '男', 33);
insert into student values(2, '林动', '男', 33);
insert into student values(10, '牧尘', '男', 31);
insert into student values(2, '萧薰儿', '女', 25);      -- 报错, 主键2已经存在了.  主键: 唯一性.
insert into student values(null, '萧薰儿', '女', 25);   -- 报错, 主键: 不能为空.

-- 4. 查看学生表结构 和 数据.
desc student;           -- 查看表结构.
select * from student;  -- 查看表数据.


# -------------------- 扩展: 面试题 delete from 和 truncate table的区别 --------------------
/*
区别:
    1. delete from只删除表数据, 不会重置主键id.
       而truncate table 相当于把表摧毁了, 然后创建一张和该表一模一样的新表, 即: 会重置主键id
    2. delete from 属于DML语句, 可以结合 事务 一起使用.
       truncate table属于DDL语句.
 */
-- 1. 查看表数据.
select * from student;

-- 2. delete from方式删除表数据.
delete from student;

-- 3. truncate table方式删除表数据.
truncate table student;
truncate student;       # 效果同上, table 关键字可以不写.

-- 4. 插入表数据.
insert into student values(null, '萧薰儿', '女', 25);


# -------------------- 约束详解(掌握) 所有常用单表约束(主键,非空,唯一,默认...) --------------------
-- 0. 建库, 切库, 查表.
drop database if exists day02;  -- 如果存在, 就删除day02数据库.
create database day02;
use day02;
show tables;

-- 1. 建表, teacher(老师表), 字段(id 主键约束, name 非空, phone 唯一约束, address 默认:北京)
create table teacher(
    id int primary key auto_increment,  # 老师id, 主键约束(非空, 唯一)
    name varchar(10) not null,          # 姓名, 非空约束, 必须传值, 不能是 null
    phone varchar(11) unique ,          # 手机号, 唯一约束, 不能重复.
    address varchar(50) default '北京'   # 住址, 默认: 北京
);

-- 2. 添加表数据.
insert into teacher values(null, '夯哥', '13112345678', '新乡');
insert into teacher values(null, null, '222', '新乡');            -- 报错, name列不能为空.
insert into teacher values(null, '梅婷', '13112345678', '濮阳');   -- 报错, phone列具有唯一性
insert into teacher values(null, '梅婷', '1311111');               -- 报错, 不写列名默认是全列名, 值的个数要和列的个数匹配.

insert into teacher(name) values('梅婷');                         -- 正确
insert into teacher(name) values('夯哥');                         -- 正确


-- 3. 查看表结构 和 表数据.
desc teacher;
select * from teacher;


# -------------------- 扩展: 备份表数据 --------------------
-- 0. 查看所有的数据表.
show tables;

-- 1. 查看源表数据.
select * from teacher;

-- 2. 备份表数据, 只会备份: 表数据, 列名, 数据类型, 不会备份约束(主键约束, 唯一约束, 因为它们的底层其实是: 索引).
-- 场景1: 备份表, 不存在.
-- 格式: create table 备份表名 select * from 源表名 where 条件...;
create table teacher_tmp
    select * from teacher;

-- 场景2: 备份表, 存在.
-- 格式: insert into 备份表名 select * from 源表名 where 条件...;
insert into teacher_tmp select * from teacher;

-- 3. 查看备份表的数据.
select * from teacher_tmp;
-- 清空备份表的数据.
truncate table teacher_tmp;

-- 4. 模拟紧急情况下的"数据恢复".
truncate table teacher;
insert into teacher select * from teacher_tmp;

-- 5. 查看数据表的约束.
desc teacher;       -- 源表
desc teacher_tmp;   -- 备份表
