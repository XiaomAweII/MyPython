/*
回顾: MySQL是关系型数据库, 表与表之间是有关系的, 具体如下:
    一对多:
		建表原则:
			在多的一方新建一列, 充当外键列, 去关联一的一方的主键列.
		例如:
			部门表 和 员工表, 分类和商品...
	多对多:
		建表原则:
			新建中间表, 该表至少有2列, 充当外键列 分别去关联多的两方的主键列.
		例如:
			学生表 和 选修课表
	一对一:
		建表原则:
			思路1: 主键对应
            思路2: 唯一外键关联.
            思路3: 把数据直接放到1张表.
		例如:
			1个人有1个身份证号
			1家公司注册地址只能有1个

外键约束详解:
    概述:
        它属于多表约束, 用于保证 数据的完整性 和 一致性.
    格式:
        添加外键约束:
            alter table 外表名 add constraint 外键约束名 foreign key(外键列名) references 主表名(主键列名);
        删除外键约束:
            alter table 外表名 drop foreign key 外键约束名;
    特点:
        外表的外键列 不能出现 主表的主键列, 没有的数据.
*/
# --------------------- 案例1: 多表约束-外键约束 ---------------------
# 1. 切库, 查表.
use day02;
show tables;
# 2. 创建 部门表(dept), id, name
drop table dept;
create table if not exists dept(
    id int primary key auto_increment,  # 部门id
    name varchar(10)                    # 部门名
);

# 3. 创建 员工表(employee), id, name, gender, salary, dept_id
drop table employee;
create table if not exists employee(
    id int primary key auto_increment,  # 员工id
    name varchar(20),                   # 员工姓名
    gender varchar(2),                  # 员工性别
    salary double,                      # 员工工资
    dept_id int                         # 员工所属的部门id
    # constraint fk01 foreign key (dept_id) references dept(id)       # 添加外键约束, 属于: 建表时添加
);

# 4. 设置 部门表 和 员工表的 外键约束.
alter table employee add constraint fk01
    foreign key (dept_id) references dept(id);

# 5. 往表中添加数据.
# 添加部门表 信息
insert into dept values(null, '人事部'), (null, '研发部'), (null, '财务部'), (null, '行政部');
# 添加员工信息
insert into employee values(null, '乔峰', '男', 5000.5, 1), (null, '虚竹', '男', 33333.3, 2);
insert into employee values(null, '段誉', '男', 50000.6, 10);      # 非法数据, 因为根本就没有编号为10的部门
# 6. 查看表数据.
select * from dept;             # 部门表
select * from employee;        # 员工表
# 7. 删除外键约束(了解)
alter table employee drop foreign key fk01;


# --------------------- 案例2: 多表建表-多对多 ---------------------
# 原则: 新建中间表, 该表至少有2列, 充当外键列, 分别去关联多的两方的主键列.
# 1. 建表
# 学生表
create table student(
    sid int primary key auto_increment,  # 学生id
    name varchar(10)        # 学生姓名
);
# 选修课表
create table course(
    cid int primary key auto_increment,  # 课程id
    name varchar(10)        # 课程名
);
# 学生-选修课关系表, 即: 中间表.
create table stu_cur(
    id int unique not null auto_increment,  # 自身id, 伪主键(非空, 唯一), 自增
    sid int,        # 学生id
    cid int         # 选修课id
);

# 2. 添加外键约束.
# 2.1 学生表 和 中间表
alter table stu_cur add constraint fk_stu_mid foreign key (sid) references student(sid);
# 2.2 课程表 和 中间表,  这个我们不给外键约束起名字了, 让程序自动生成.
alter table stu_cur add  foreign key (cid) references course(cid);

# 3. 设置中间表的 学生id(sid) 和 选修课id(cid) 为: 联合主键.
alter table stu_cur add primary key(sid, cid);
# 4. 添加表数据.
# 4.1 学生表.
insert into student values(null, '乔峰'), (null, '虚竹'), (null, '段誉');
# 4.2 选修课表
insert into course values(null, 'AI人工智能'), (null, 'Py大数据'), (null, '鸿蒙');
# 4.3 中间表.
insert into stu_cur values(null, 1, 1);        # 乔峰, AI
insert into stu_cur values(null, 1, 2);        # 乔峰, 大数据
insert into stu_cur values(null, 2, 1);        # 虚竹, AI
insert into stu_cur values(null, 2, 1);        # 虚竹, AI, 报错.
# 5. 查看表数据.
# 4.1 学生表.
select * from student;
# 4.2 选修课表
select * from course;
# 4.3 中间表.
select * from stu_cur;



/*
交叉查询:
    格式:
        select * from A, B;
    结果:
        两张表的 笛卡尔积.
连接查询:
    内连接:
        格式:
            select * from A inner join B on 关联条件;		# inner可以省略不写.
            select * from A, B where 关联条件;
        结果:
            表的 交集.
    外连接:
        左外连接: select * from A left outer join B on 关联条件;	 # outer可以省略不写
            结果:
                左表的全集 + 交集
        右外连接: select * from A right outer join B on 关联条件;  # outer可以省略不写
            结果:
                右表全集 + 交集
子查询:
    概述:
        1个SQL语句的查询条件, 需要依赖另1个SQL语句的查询结果, 外边的查询称之为: 主查询(父查询)
        里边的查询称之为: 子查询.
    格式:
        #      父查询(主查询)					子查询
        select * from A where 列名 > (select 列名 from 表名 where...);
扩展: 自连接查询(自关联查询)
    就是在连接查询的基础上, 表自己和自己做关联查询, 例如: select * from A left join A on 关联条件;

多表查询的精髓:
    想办法把多张表"拼接成"一张表, 然后对该表 进行"单表查询".
 */
# --------------------- 案例3: 多表查询-交叉查询 ---------------------
# 1. 建表.
# 创建hero表
CREATE TABLE hero (
    hid   INT PRIMARY KEY,
    hname VARCHAR(255),
    kongfu_id INT
);
# 创建kongfu表
CREATE TABLE kongfu(
    kid     INT PRIMARY KEY,
    kname   VARCHAR(255)
);

# 2. 添加表数据.
# 插入hero数据
INSERT INTO hero VALUES(1, '鸠摩智', 9),(3, '乔峰', 1),(4, '虚竹', 4),(5, '段誉', 12);

# 插入kongfu数据
INSERT INTO kongfu VALUES(1, '降龙十八掌'),(2, '乾坤大挪移'),(3, '猴子偷桃'),(4, '天山折梅手');

# 3. 查看表数据.
select * from hero;     # 英雄表.

select * from kongfu;   # 功夫表.

# 4. 演示 交叉查询.   查询结果是: 表的笛卡尔积, 即: 表A的总数 * 表B的总数.
select * from hero, kongfu;     # 16条


# --------------------- 案例4: 多表查询-连接查询 ---------------------
# 场景1: 内连接, 查询结果是: 表的交集.
# 显式内连接, select * from A inner join B on 关联条件.
select * from hero h inner join kongfu kf on h.kongfu_id = kf.kid;
select * from hero h join kongfu kf on h.kongfu_id = kf.kid;  # 语法糖, inner可以省略不写.

# 隐式内连接, select * from A, B where 关联条件.
select * from hero h, kongfu kf where h.kongfu_id = kf.kid;

# 场景2: 外连接.
# 左外连接: 查询结果是 左表全集 + 交集.
# 格式: select * from A left outer join B on 关联条件.
select * from hero h left outer join kongfu kf on h.kongfu_id = kf.kid;
select * from hero h left join kongfu kf on h.kongfu_id = kf.kid;   # 语法糖, outer可以省略不写.

# 右外连接: 查询结果是 右表全集 + 交集.
# 格式: select * from A right outer join B on 关联条件.
select * from hero h right outer join kongfu kf on h.kongfu_id = kf.kid;
select * from hero h right join kongfu kf on h.kongfu_id = kf.kid;   # 语法糖, outer可以省略不写.


# --------------------- 案例5: 多表查询-子查询 ---------------------
# 概述: 一个SQL语句的查询条件, 需要依赖另1个SQL语句的查询结果, 这种写法就称之为: 子查询.
#           主查询(父查询)                      子查询
# 格式: select ... from 表名 where 字段 in (select 列名 from 表名...);
# 1. 建表.
create table category (         # 商品分类表
  cid varchar(32) primary key , # 分类id
  cname varchar(50)             # 分类名
);
create table products(          # 商品表
  pid varchar(32) primary key , # 商品id
  pname varchar(50),            # 商品名
  price int,                    # 商品价格
  flag varchar(2),              # 是否上架标记为：1表示上架、0表示下架
  category_id varchar(32),      # 分类id
  constraint products_fk foreign key (category_id) references category (cid)    # 建表时, 添加: 外键约束
);

# 2. 添加表数据.
#分类
INSERT INTO category(cid,cname) VALUES('c001','家电');
INSERT INTO category(cid,cname) VALUES('c002','服饰');
INSERT INTO category(cid,cname) VALUES('c003','化妆品');
INSERT INTO category(cid,cname) VALUES('c004','奢侈品');
#商品
INSERT INTO products(pid, pname,price,flag,category_id) VALUES('p001','联想',5000,'1','c001');
INSERT INTO products(pid, pname,price,flag,category_id) VALUES('p002','海尔',3000,'1','c001');
INSERT INTO products(pid, pname,price,flag,category_id) VALUES('p003','雷神',5000,'1','c001');
INSERT INTO products (pid, pname,price,flag,category_id) VALUES('p004','JACK JONES',800,'1','c002');
INSERT INTO products (pid, pname,price,flag,category_id) VALUES('p005','真维斯',200,'1','c002');
INSERT INTO products (pid, pname,price,flag,category_id) VALUES('p006','花花公子',440,'1','c002');
INSERT INTO products (pid, pname,price,flag,category_id) VALUES('p007','劲霸',2000,'1','c002');
INSERT INTO products (pid, pname,price,flag,category_id) VALUES('p008','香奈儿',800,'1','c003');
INSERT INTO products (pid, pname,price,flag,category_id) VALUES('p009','相宜本草',200,'1','c003');

# 3. 查看表数据.
select * from category;       # 分类表
select * from products;       # 商品表

# 4. 查询哪些分类的商品已经上架.
select distinct c.cid, c.cname, flag from category c left join products p on c.cid = p.category_id;
select distinct c.cid, c.cname, flag from category c inner join products p on c.cid = p.category_id;

# 5. 查询所有分类商品的个数.
# 根据 分类商品的id, 名字 进行分组.
select cid, cname, count(cid) as total_cnt from category c left join products p on c.cid = p.category_id group by cid, cname;

# 6. 查询"化妆品"分类, 上架商品详情.
# select * from products where category_id = '化妆品 分类的id';
# step1: 获取化妆品的 分类id
select cid from category where cname='化妆品';

# step2: 把上边的查询结果, 作为下边SQL语句的 条件
select * from products where category_id = 'c003';

# 合并版: 子查询.
#             父查询/主查询                                子查询
select * from products where category_id = (select cid from category where cname='化妆品');



# --------------------- 案例6: 自关联查询 ---------------------
/*
自关联查询介绍:
    概述:
        自关联查询也叫自连接查询, 即: 表自己 和 表自己做关联查询, 获取结果.
    典型应用:
        省市区三级联动.
    格式:
        select * from 表A join 表A on 关联条件...;
        select * from 表A left join 表A on 关联条件...;
*/
# 1. 导入数据, area.sql, 记录的市: 行政区域表的信息.
# 2. 查看表数据.
select * from areas limit 10;       # id: 自身id,  title: 自身名称.   pid:父级id

# 3. 初始表数据.
select * from areas where pid is null;      # 省, 直辖市级别
# 河南省所有的市.
select * from areas where pid = '410000';   # 河南省所有的市
# 新乡市所有的 县区
select * from areas where pid = '410700';   # 新乡市所有的 县区

# 4. 查看 所有省, 所有市, 所有县区的信息.
select
    province.id, province.title,    # 省的id, 名字
    city.id, city.title,            # 市的id, 名字
    county.id, county.title         # 县区的id, 名字
from
    areas as province           # 省
left join areas as city on city.pid = province.id   # 市,  市的父级id = 省的id
left join areas as county on county.pid = city.id;   # 县区, 县区的父级id = 市的id


select
    province.id, province.title,    # 省的id, 名字
    city.id, city.title,            # 市的id, 名字
    county.id, county.title         # 县区的id, 名字
from
    areas as province           # 省
left join areas as city on city.pid = province.id   # 市,  市的父级id = 省的id
left join areas as county on county.pid = city.id   # 县区, 县区的父级id = 市的id
where county.id = '350427';


# --------------------- 案例7: 窗口函数 ---------------------
/*
窗口函数介绍:
    概述:
        它是MySQL8.0的特性, 窗口函数有很多, 常用的有:
            row_number():   可以理解为: 行号, 即: 1, 2, 3, 4...
            rank():         (稀疏)排名, 会跳跃.  1, 2, 2, 2, 5...
            dense_rank()    (密集)排名, 不会跳跃. 1, 2, 2, 2, 3...
    精髓:
        窗口函数 = 给表新增1列, 至于新增的是什么, 取决于 用哪个窗口函数.
    格式:
        窗口函数 over(partition by 分组字段 order by 排序字段 asc | desc)
    细节:
        1. 学窗口函数, 首先要掌握两个知识点.
            A. 分组排名.
            B. 分组排名求TopN
        2. where后边写的字段, 必须是表中已有的字段.
*/
# 1. 准备数据, 建表, 添加数据.
create table employee (empid int,ename varchar(20) ,deptid int ,salary decimal(10,2));

insert into employee values(1,'刘备',10,5500.00);
insert into employee values(2,'赵云',10,4500.00);
insert into employee values(2,'张飞',10,3500.00);
insert into employee values(2,'关羽',10,4500.00);

insert into employee values(3,'曹操',20,1900.00);
insert into employee values(4,'许褚',20,4800.00);
insert into employee values(5,'张辽',20,6500.00);
insert into employee values(6,'徐晃',20,14500.00);

insert into employee values(7,'孙权',30,44500.00);
insert into employee values(8,'周瑜',30,6500.00);
insert into employee values(9,'陆逊',30,7500.00);

# 2. 查看数据.
select * from employee;

# 3. 分组排名.   假设数据是: 100, 90, 90, 80, 则: row_number是 1, 2, 3, 4  rank是 1, 2, 2, 4  dense_rank()是 1, 2, 2, 3
select
    *,
    row_number() over (partition by deptid order by salary desc) as rn,     # 根据部门id分组, 组内按照工资降序排名
    rank() over (partition by deptid order by salary desc) as rk,           # 根据部门id分组, 组内按照工资降序排名
    dense_rank() over (partition by deptid order by salary desc) as dr      # 根据部门id分组, 组内按照工资降序排名
from employee;

# 4.分组排名, 求TopN, 例如: 每个部门, 工资最高的2人.
select * from (
    select
        *,
        rank() over (partition by deptid order by salary desc) as rk           # 根据部门id分组, 组内按照工资降序排名
    from employee
) as t1             # 把查询结果, 封装成一张新表, 后续就可以用 rk这个字段了.
where rk <= 2;
