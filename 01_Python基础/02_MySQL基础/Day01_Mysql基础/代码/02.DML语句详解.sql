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
