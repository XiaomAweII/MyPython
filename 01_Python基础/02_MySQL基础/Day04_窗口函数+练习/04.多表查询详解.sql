-- 多表查询的精髓是: 根据关联条件把多张表变成一张表, 然后进行"单表"查询.

-- 1. 切库.
use day02;

-- 2. 建表.
# 创建hero表
CREATE TABLE hero(
    hid   INT PRIMARY KEY,  # 英雄id
    hname VARCHAR(255),     # 英雄名
    kongfu_id INT           # 功夫id
);

# 创建kongfu表
CREATE TABLE kongfu (
    kid     INT PRIMARY KEY,    # 功夫id
    kname   VARCHAR(255)        # 功夫名
);


-- 3. 添加表数据.
# 插入hero数据
INSERT INTO hero VALUES(1, '鸠摩智', 9),(3, '乔峰', 1),(4, '虚竹', 4),(5, '段誉', 12);
# 插入kongfu数据
INSERT INTO kongfu VALUES(1, '降龙十八掌'),(2, '乾坤大挪移'),(3, '猴子偷桃'),(4, '天山折梅手');

-- 4. 查看表数据.
select * from hero;

select * from kongfu;


#  ------------- 多表查询: 交叉查询 -------------
# 格式: select * from 表A, 表B;
# 结果: 两张表的笛卡尔积, 即: 表A的总条数 * 表B的总条数
select * from hero, kongfu;     -- 会产生大量的 脏数据, 实际开发不用.


#  ------------- 多表查询: 连接查询 -------------
# 场景1: 内连接, 查询结果: 表的交集.
# 格式1: 显式内连接.  select * from A inner join B on 关联条件 where ...;
select * from hero h inner join kongfu kf on h.kongfu_id = kf.kid;
select * from hero h join kongfu kf on h.kongfu_id = kf.kid;        # inner 可以省略不写

# 格式2: 隐式内连接.  select * from A, B where 关联条件;
select * from hero h, kongfu kf where h.kongfu_id = kf.kid;

# 场景2: 外连接.
# 格式1: 左外连接.  select * from A left outer join B on 关联条件 where ...;      结果: 左表全集 + 交集
select * from hero h left outer join kongfu kf on h.kongfu_id = kf.kid;
select * from hero h left join kongfu kf on h.kongfu_id = kf.kid;    # outer可以省略

# 格式2: 右外连接.  结果: 右表全集 + 交集
select * from hero h right join kongfu kf on h.kongfu_id = kf.kid;    # outer可以省略

