# -------------- 案例1: 准备动作 --------------
-- 1. 建库, 切库, 查表.
create database if not exists day02;
use day02;
show tables;

-- 2. 创建商品表.   快捷键: ctrl + shift + u  大小写切换
create table product (
    pid         int primary key auto_increment,     # 商品id
    pname       varchar(20),        # 商品名
    price       double,             # 商品单价
    category_id varchar(32)         # 商品所属的 分类id
);

-- 3. 添加表数据.
INSERT INTO product(pid,pname,price,category_id) VALUES(1,'联想',5000,'c001');
INSERT INTO product(pid,pname,price,category_id) VALUES(2,'海尔',3000,'c001');
INSERT INTO product(pid,pname,price,category_id) VALUES(3,'雷神',5000, null);
INSERT INTO product(pid,pname,price,category_id) VALUES(4,'杰克琼斯',800,'c002');
INSERT INTO product(pid,pname,price,category_id) VALUES(5,'真维斯',200,'c002');
INSERT INTO product(pid,pname,price,category_id) VALUES(6,'花花公子',440,null);
INSERT INTO product(pid,pname,price,category_id) VALUES(7,'劲霸',2000,'c002');
INSERT INTO product(pid,pname,price,category_id) VALUES(8,'香奈儿',800,'c003');
INSERT INTO product(pid,pname,price,category_id) VALUES(9,'相宜本草',200,'c003');
INSERT INTO product(pid,pname,price,category_id) VALUES(10,'面霸',5,'c003');
INSERT INTO product(pid,pname,price,category_id) VALUES(11,'好想你枣',56,'c004');
INSERT INTO product(pid,pname,price,category_id) VALUES(12,'香飘飘奶茶',1,'c005');
INSERT INTO product(pid,pname,price,category_id) VALUES(13,'海澜之家',1,'c002');

-- 4. 查看表数据.
select * from product;


/*
 记忆: 一个单表查询的完整语法如下
    select
        distinct 列1, 列2...
    from
        数据表名
    where
        组前筛选
    group by
        分组字段
    having
        组后筛选
    order by
        排序字段 [asc | desc]
    limit
        起始索引, 数据条数;
 */
-- ------------------------------- 案例2: 简单查询 -------------------------------
# 1.查询所有的商品.
select pid, pname, price, category_id from product;
select * from product;      -- * 在这里代表 所有的列.

# 2.查询商品名和商品价格.
select pname, price from product;

# 3.查询结果是表达式（运算查询）：将所有商品的价格+10元进行显示.
select pname, price + 10 from product;

# 4. 起别名, as 别名即可, 其中 as 可以省略不写.
select pname, price + 10 as 价格 from product;
select pname, price + 10 价格 from product;    -- as 可以省略


-- ------------------------------- 案例3: 条件查询 -------------------------------
/*
格式:
    select * from 表名 where 条件;
条件可以是:
    比较运算符:
        >, >=, <, <=, =, !=, <>
    范围判断:
        between 起始值 and 结束值     包左包右.
        in (值1, 值2, 值3);          满足任意1个条件即可.
    模糊查询:
        like '张%'       %代表任意个占位符, _代表1个占位符.
    逻辑运算符:
        and     并且的意思, 叫: 逻辑与, 要求所有的条件都要满足.
        or      或者的意思, 叫: 逻辑或, 要求满足任意1个条件即可.
        not     取反的意思, 叫: 逻辑非, 取相反的条件即可.
    非空查询:
        is null
        is not null
*/
# 1. 查询商品名称为“花花公子”的商品所有信息：
select * from product where pname = '花花公子';
select * from product where pname in ('花花公子');

# 2. 查询价格为800商品
select * from product where price=800;
select * from product where price in (800);

# 3. 查询价格不是800的所有商品
select * from product where price != 800;
select * from product where price <> 800;
select * from product where price not in (800);

# 4. 查询商品价格大于60元的所有商品信息
select * from product where price > 60;
select * from product where not price <= 60;

# 5. 查询商品价格小于等于800元的所有商品信息
select * from product where price <= 800;

# 6. 查询商品价格在200到800之间所有商品
select * from product where price between 200 and 800;      -- 包左包右
select * from product where price >= 200 and price <= 800;

# 7. 查询商品价格是200或800的所有商品
select * from product where price=200 or price=800;
select * from product where price in (200, 800);

# 8. 查询价格不是800的所有商品
select * from product where price != 800;
select * from product where price <> 800;
select * from product where price not in (800);

# 9. 查询以'香'开头的所有商品
select * from product where pname like '香%';
select * from product where pname like '香_';    # 两个字, 第1个字是香, 第2个字无所谓.
select * from product where pname like '香__';   # 三个字, 第1个字是香, 剩下2个字无所谓.

# 10. 查询第二个字为'想'的所有商品
select * from product where pname like '_想%';

# 11. 查询没有分类的商品
select * from product where category_id=null;      -- 无结果, null表示空, 即: 啥都没有. 所以不能用 比较运算符直接判断.
select * from product where category_id is null;   -- 正确写法.

# 12. 查询有分类的商品
select * from product where category_id is not null;



-- ------------------------------- 案例4: 排序查询 -------------------------------
# 格式: select * from 数据表名 order by 排序字段1 [asc | desc], 排序字段2 [asc | desc]...;
# 单词: ascending

# 1.使用价格排序(降序)
select * from product order by price;       # 如果不写, 默认是升序, 即: asc
select * from product order by price asc;   # asc: 升序
select * from product order by price desc;  # desc: 降序

# 2.在价格排序(降序)的基础上，以分类排序(降序)
select * from product order by price desc, category_id desc;

-- ------------------------------- 案例5: 聚合查询 -------------------------------
/*
概述/作用:
    聚合函数是用来操作 某列数据 的.
分类:
    count() 功能是: 统计表的总行数(总条数)
    max()   功能是: 最大值, 只针对于 数字列 有效.
    min()   功能是: 最小值, 只针对于 数字列 有效.
    sum()   功能是: 求总和, 只针对于 数字列 有效.
    avg()   功能是: 平均值, 只针对于 数字列 有效.

面试题: count(*), count(列), count(1)的区别是什么?
答案:
    1. 是否统计null值.
        count(列): 只统计该列的非null值.
        count(*), count(1): 都会统计null值.
    2. 效率问题.
        从高到低, 分别是: count(主键列) > count(1) > count(*) > count(普通列)
*/

-- 1. 求 product 表的总数据条数.
select count(*) as total_cnt from product;       -- 表数据总条数: 13条

-- 2. 求商品价格的 最大值.
select max(price) as max_price from product;

-- 3. 求商品价格的 最小值.
select min(price) as min_price from product;

-- 4. 求商品价格的 求总和.
select sum(price) as total_price from product;

-- 5. 求商品价格的 平均值.
select 10 / 3;      -- 3.3333
select avg(price) avg_price from product;

-- 扩展: 因为SQL中的 整数相除, 结果可能是小数, 所以我们可以保留指定的小数位, 让结果更好看.
select round(avg(price), 2) avg_price from product;

-- 扩展: 四舍五入, 保留两位小数.
select round(3.12545, 2);   -- 3.13

-- 面试题: count(*), count(1), count(列)的区别.
-- 1. 是否统计null值.  count(1), count(*)统计,  count(列名) 不统计.
-- 2. 效率问题.  count(主键列) => count(1) > count(*) > count(普通列)
select count(*) from product;       -- 13条
select count(1) from product;       -- 13条
select count(price) from product;   -- 13条

select count(category_id) from product; -- 11条, count(列)的时候, 只统计该列的非空值.

-- ------------------------------- 案例6: 分组查询 -------------------------------
/*
解释:
    相当于把表 按照 分组字段 分成 n个组(n份), 然后就可以对每组的数据做筛选统计了.
    逻辑分组, 数据(物理存储上)还在一起.
大白话解释:
    咱们有40个人, 分成了4组(逻辑分组), 但是上课大家还都坐到1个教室(物理上)
格式:
    select
        分组字段, 聚合函数...
    from
        表名
    where
        组前筛选
    group by
        分组字段
    having
        组后筛选;

细节:
    1. 分组查询的 查询列, 只能出现: 分组字段 或者 聚合函数.
    2. where是组前筛选, having是组后筛选.
    3. 分组查询一般要结合聚合函数一起使用, 否则没有意义.

面试题: where 和 having的区别是什么?
答案:
    where:  组前筛选, 后边不能跟聚合函数.
    having: 组后筛选, 后边可以跟聚合函数.
*/
# 1.统计各个分类商品的个数
select category_id, count(*) total_cnt from product group by category_id;

# 2.统计各个分类商品的个数, 且只显示个数大于1的信息
select
    category_id,            # 分类id
    count(*) total_cnt      # 该分类的总商品个数.
from
    product
group by
    category_id
having total_cnt > 1;

# 3.统计各个分类商品的个数, 且只显示个数大于1的信息, 按照 商品总数, 降序排列.
select
    category_id,            # 分类id
    count(*) total_cnt      # 该分类的总商品个数.
from
    product
group by
    category_id
having
    total_cnt > 1
order by
    total_cnt desc;

# 4. 综合版, 统计每类商品的总价格, 只统计单价在500以上的商品信息, 且只显示总价在 2000 以上的分组信息, 然后按照总价升序排列, 求出价格最低的那个分类信息.
select
    category_id,                # 分类id
    sum(price) total_price      # 该分类的商品 总价格.
from
    product
where
    price > 500             # 组前筛选
group by
    category_id             # 分组字段
having
    total_price > 2000     # 组后筛选
order by                   # 排序字段
    total_price
limit                      # 分页查询
    0, 1;                  # 数据表中, 每条数据都有自己的编号, 编号是从0开始的. 这里的0, 1意思是: 从编号为0的数据开始拿, 拿1条数据.



-- ------------------------------- 案例7: 分页查询 -------------------------------
/*
概述:
    相当于一次性从表中获取n条数据, 例如: 总条数为100条, 每页10条, 则一共有10页.
好处:
    1. 一方面可以降低服务器, 数据库的压力.
    2. 另一方面, 可以提高用户体验, 阅读性更强.
语法格式:
    limit 起始索引, 数据条数;
格式解释:
    起始索引: 表示 从索引为几的数据行, 开始获取数据. 数据表中每条数据都有自己的索引, 索引是从0开始的.
    数据条数: 表示 获取几条数据.

扩展提高: 和分页相关的4个名词如下.
    数据总条数:      select count(1) from 表名;        假设: 23条
    每页的数据总数:   产品经理, 项目经理.                 假设: 5条
    每页的起始索引:   (当前页数 - 1) * 每页的数据条数      假设: 第3页, 则: (3 - 1) * 5 = 10, 起始索引为10, 即: 从第11条数据开始获取.
    总页数:         (总条数 + 每页的数据条数 - 1) / 每页的数据条数.  注意: 这里是整除, 只要整数部分, 等价于: 求地板数.
                   例如: 总23条, 每页5条, 则:  (23 + 5 - 1) / 5 = 27 / 5 =  5
                   例如: 总25条, 每页5条, 则:  (25 + 5 - 1) / 5 = 29 / 5 =  5
*/
-- 从商品表中, 按照 5条/页 的方式, 获取数据.
select * from product limit 0, 5;           -- 第1页
select * from product limit 5, 5;           -- 第2页
select * from product limit 10, 5;           -- 第3页

-- 从商品表中, 按照 3条/页 的方式, 获取 第3页数据.
select * from product limit 6, 3;           -- 第3页

# 语法糖, 如果起始索引为0, 则可以省略不写.
select * from product limit 5;           -- 第1页

# floor(), 求地板数, 即: 比这个数字小的所有数字中, 最大的那个整数.
select floor(4.1);      -- 4
select floor(4.9);      -- 4

# ceil(), 求天花板数, 即: 比这个数字大的所有数字中, 最小的那个整数.
select ceil(5.3);       -- 6
select ceil(5.0);       -- 5

# 扩展: distinct, 去重查询, 即: 去除重复的数据.
select distinct category_id, price from product;    -- 是把 category_id, price当做整体来去重的.

# 思考, 如何去重呢?
# 方式1: distinct
select distinct category_id from product;

# 方式2: 分组方式.
select category_id from product group by category_id;
