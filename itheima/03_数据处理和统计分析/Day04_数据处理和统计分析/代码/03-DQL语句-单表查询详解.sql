/*
前言:
    我们学习SQL语句, 主要学习的就是DQL(数据查询语言), 又分为: 单表查询, 多表查询, 窗口函数.
一个完整的单表查询的语法格式如下:
    select
        [distinct] 列1 as 别名, 列2...
    from
        数据表名
    where
        组前筛选
    group by
        分组字段1, 分组字段2...
    having
        组后筛选
    order by
        排序的列1, 列2 [asc | desc]...
    limit
        起始索引, 数据条数;
*/

# --------------------- 案例1: 准备动作 ---------------------
# 1. 建库, 切库.
create database day02;
use day02;
show tables;

# 2. 建表, 添加表数据.   ctrl + shift + 字母U   字母大小写切换.
# 创建商品表：
create table product(
    pid         int primary key auto_increment,    # 商品id,
    pname       varchar(20),        # 商品名
    price       double,             # 商品价格
    category_id varchar(32)         # 商品所属的分类id
);
# 插入数据
INSERT INTO product(pid,pname,price,category_id) VALUES(null,'联想',5000,'c001');
INSERT INTO product(pid,pname,price,category_id) VALUES(null,'海尔',3000, null);
INSERT INTO product(pid,pname,price,category_id) VALUES(null,'雷神',5000,'c001');
INSERT INTO product(pid,pname,price,category_id) VALUES(null,'杰克琼斯',800,'c002');
INSERT INTO product(pid,pname,price,category_id) VALUES(null,'真维斯',200,null);
INSERT INTO product(pid,pname,price,category_id) VALUES(null,'花花公子',440,'c002');
INSERT INTO product(pid,pname,price,category_id) VALUES(null,'劲霸',2000,'c002');
INSERT INTO product(pid,pname,price,category_id) VALUES(null,'香奈儿',800,'c003');
INSERT INTO product(pid,pname,price,category_id) VALUES(null,'相宜本草',200,'c003');
INSERT INTO product(pid,pname,price,category_id) VALUES(null,'面霸',5,'c003');
INSERT INTO product(pid,pname,price,category_id) VALUES(null,'好想你枣',56,'c004');
INSERT INTO product(pid,pname,price,category_id) VALUES(null,'香飘飘奶茶',1,'c005');
INSERT INTO product(pid,pname,price,category_id) VALUES(null,'海澜之家',1,'c002');

# 3. 查询表数据.
select * from product;


# --------------------- 案例2: 简单查询 ---------------------
# 格式: select 列名1, 列名2... from 表名;
# 1.查询所有的商品.
select pid, pname, price, category_id from product;
select * from product;      # 上述格式的语法糖, 即: *在这里代表 全列名

# 2.查询商品名和商品价格.
select pname, price from product;

# 3.查询结果是表达式（运算查询）：将所有商品的价格+10元进行显示.
select pname, price + 10 from product;

# 4. 别名查询, 列名, 表名都可以起别名.  格式:  列名/表名 as 别名.
select pname, price + 10 as price from product;
select pname, price + 10 price from product;     # as 可以省略不写
select pname 商品名, price + 10 商品价格 from product;     # 别名还可以是中文

# 5. 去重查询.
# 需求1: 查询所有的商品价格.
select distinct price from product;
# 需求2: 查看商品分类, 及其对应的价格.
# 这样写是把 商品分类id 和 商品价格 当做一个整体来去重的. 即: c001, 5000 和 c001,3000 不是同一条数据.
select distinct category_id, price from product;


# --------------------- 案例3: 条件查询 ---------------------
/*
条件查询, 格式为:
    select ... from 数据表名 where 条件;
条件查询, 几种写法, 即: where后边可以跟什么:
    比较运算符: >, >=, <, <=, =, !=, <>
    范围查询:
        区间校验: between 起始值 and 结束值
        固定值校验: 列名 in (值1, 值2...)
    模糊查询:   列名 like '_%'
    非空查询:   is null 和 is not null
    逻辑运算符: and, or, not
*/
# 1.查询商品名称为“花花公子”的商品所有信息：
select * from product where pname = '花花公子';
# 2.查询价格为800商品
select * from product where price=800;
select * from product where price in (800);
# 3.查询价格不是800的所有商品
select * from product where price != 800;
select * from product where price <> 800;   # 效果同上
select * from product where not price=800;  # 效果同上
select * from product where price not in (800); # 效果同上
# 4.查询商品价格大于60元的所有商品信息
select * from product where price > 60;
# 5.查询商品价格小于等于800元的所有商品信息
select * from product where price <= 800;
# 6.查询商品价格在200到800之间所有商品
select * from product where price between 200 and 800;     # 包左包右
select * from product where price >= 200 and price <= 800;
# 7.查询商品价格是200或800的所有商品
select * from product where price in (200, 800);
select * from product where price=200 or price=800;
# 8.查询以'香'开头的所有商品
select * from product where pname like '香%';      # 模糊查询规则, _表示任意的1个字符, %表示多个任意的字符.
select * from product where pname like '香__';     # 查询以香开头, 一共3个字符的内容.
# 9.查询第二个字为'想'的所有商品
select * from product where pname like '_想%';
# 10.查询没有分类的商品
select * from product where category_id = null;     # 不会报错, 结果不是我们要的.
select * from product where category_id is null;    # 正确写法.
# 11.查询有分类的商品
select * from product where category_id is not null;    # 正确写法.



# --------------------- 案例4: 排序查询 ---------------------
# 格式: select ... from 表名 where ... order by 排序字段1 [asc | desc], 排序字段2 [asc | desc]...
# 1.使用价格排序(降序)
select * from product order by price;           # 默认是: 升序
select * from product order by price asc;       # asc(ascending), 表示: 升序, 可以省略不写.
select * from product order by price desc;      # desc: 降序

# 2.在价格排序(降序)的基础上，以分类排序(降序)
select * from product order by price desc, category_id desc;      # desc: 降序



# --------------------- 案例5: 聚合查询 ---------------------
/*
聚合查询 介绍:
    概述:
        之前我们写的查询都是一行行操作的, 聚合查询是 一列一列来操作的.
    例如:
        计算某列的非空值的数据总条数, 某列最大值/最小值...
    分类:
        count() 一般用于统计行数.
        sum()   求和
        max()   求最大值
        min()   求最小值
        avg()   求平均值
    细节:
        1. 面试题: count(1), count(*), count(列) 区别是什么?
            区别1: 是否统计null值.
                count(1), count(*): 统计
                count(列): 不统计.
            区别2: 效率问题
                count(主键列) > count(1) > count(*) > count(普通列)
*/
# 1、查询商品的总条数
select count(*) as total_cnt from product;       # 统计数据条数, 包括: null值, 13条
select count(1) as total_cnt from product;       # 统计数据条数, 包括: null值, 13条
select count(pid) as total_cnt from product;     # 统计数据条数, 只统计该列的 非空值, 13条
select count(category_id) as total_cnt from product;     # 统计数据条数, 只统计该列的 非空值, 11条
# 2、查询价格大于200商品的总条数
select count(pid) from product where price > 200;
# 3、查询分类为'c001'的所有商品的总和
select sum(price) as total_price from product where category_id='c001';
# 4、查询分类为'c002'所有商品的平均价格
select * from product where category_id='c002';
select avg(price) as avg_price from product where category_id='c002';
select round(avg(price), 1) as avg_price from product where category_id='c002';
# 扩展: 四舍五入, 保留指定位数的小数.
select round(13.12345, 4);      -- 13.1235
select round(13.12345, 2);      -- 13.12
select round(810.25, 1);        -- 810.3
# 5、查询商品的最大价格和最小价格
select max(price) max_price, min(price) min_price from product;


# --------------------- 案例6: 分组查询 ---------------------
/*
分组查询 介绍:
    概述:
        分组查询 = 按照分组字段, 把数据划分成n个组, 然后再各组中做 (聚合)统计, 例如: 每组的总条数, 某列最大值/最小值...
    格式:
        select
            分组字段, 聚合函数...
        from
            数据表名
        where
            组前筛选
        group by
            分组字段
        having
            组后筛选;
    细节:
        1. 分组查询的查询列: 只能出现 分组字段 和 聚合函数.
        2. where 和 having的区别:
            区别1: 作用时机不同.
                where: 组前筛选
                having: 组后筛选.
            区别2: 后边是否可以跟 聚合函数
                where: 不可以
                having: 可以
*/
# 1.统计各个分类商品的个数
select category_id, count(pid) total_cnt from product group by category_id;

# 2.统计各个分类商品的个数, 且只显示个数大于1的信息
select category_id, count(pid) total_cnt from product group by category_id having total_cnt > 1;

# 3. 统计商品表中, 各分类商品的总个数, 只显示个数小于3的信息, 且只统计价格在100以上的商品, 并按照商品个数降序排列, 如果(每个分类)商品个数一致, 按照分类id 升序排列.
select
    category_id, count(pid) total_cnt
from
    product
where
    price > 100         # 组前筛选, 价格在100以上的
group by
    category_id         # 按 商品分类id 分组
having
    total_cnt < 3       # 组后筛选, 只统计 商品数量 小于 3的 分组信息
order by
    total_cnt desc, category_id;    # 按照每个分类商品个数降序排列, 如一致, 按照分类id升序排列.

# 4. 统计商品表中, 各分类商品的总价格, 只统计单价在200以上的商品信息, 且只显示总金额大于 2000 的数据, 按照总金额降序排列.
select category_id, sum(price) total_price from product where price > 200 group by category_id having total_price > 2000 order by total_price desc;



# --------------------- 案例7: 分页查询 ---------------------
/*
分页查询:
    概述:
        就是分批次从数据表中获取数据, 一方面可以降低服务器压力, 另一方面可以降低浏览器端解析数据的压力, 且能提高用户体验.
    格式:
        limit 起始索引, 数据条数;
    细节:
        1. SQL表中, 每条数据都是有索引的, 且索引是从 0 开始的, 即: 第1条数据, 索引为0, 第2条数据, 索引为1...
        2. 和分页相关的四个名词:
            数据总条数:      count()函数
            每页数据条数:    产品经理, 项目经理, 你...
            每页起始索引:    (当前页数 - 1) * 每页的数据条数
            总页数:         (数据总条数 + 每页的数据条数 - 1) / 每页的数据条数      注意: 这里是整除, 即: 只保留整数部分.
                例如: (13 + 3 - 1) / 3 = 15 / 3 = 5
                例如: (13 + 5 - 1) / 3 = 17 / 5 = 3
*/
# 1. 查看原表数据.
select * from product;

# 2. 只获取前5条数据.
select * from product limit 0, 5;       # 从索引0开始, 获取 5条数据
select * from product limit 5;          # 语法糖, 索引如果从0开始, 则可以省略不写.

# 3. 每页5条, 获取第2页, 第3页数据.
select * from product limit 5, 5;       # 从索引5(第6条)开始, 获取 5条数据
select * from product limit 10, 5;      # 从索引10(第11条)开始, 获取 5条数据

# 4. 每页7条, 如何获取?
select * from product limit 0, 7;       # 第1页
select * from product limit 7, 7;       # 第2页


# 5. 每页3条, 如何获取?
select * from product limit 0, 3;       # 第1页
select * from product limit 3, 3;       # 第2页
select * from product limit 6, 3;       # 第3页
select * from product limit 9, 3;       # 第4页
select * from product limit 12, 3;       # 第5页
