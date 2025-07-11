/*
窗口函数介绍:
    概述:
        目前我们讲的窗口函数主要是: row_number(), rank(), dense_rank(), 主要是用来做: 排名的.
        当然, 窗口函数还有其他的很多, 后续我们会深入讲解.
    大白话:
        窗口函数 = 给表 新增1列, 至于新增的一列是什么内容, 取决于你的窗口函数怎么写.
    格式:
        开窗函数 over(partition by 分组字段 order by 排序字段 asc | desc)
    格式解释:
        开窗口函数:  这里指的是 row_number(), rank(), dense_rank(), 当前还有其他的, 我们后续讲解.
        over():    固定格式, 里边写的是: 分组, 排序的相关操作.
        partition by:   类似于group by, 做分组的.
        order by:       (局部)排序的, 即: 组内排序.
    细节:
        1. row_number(), rank(), dense_rank()主要区别就是 遇到相同数据了, 如何处理, 具体如下:
            例如: 数据为, 100, 90, 90, 80
            则:
                row_number():   1, 2, 3, 4
                rank():         1, 2, 2, 4
                dense_rank():   1, 2, 2, 3
        2. 目前对于窗口函数, 大家只要掌握: 分组排名, 求TopN即可.
        3. 窗口函数是MySQL8.X的新特性, 所以如果你是MySQL5.X版本, 用不了这个功能.
 */

-- 1. 建库, 切库, 查表.
drop database if exists day04;
create database day04;
use day04;
show tables;

-- 2. 建表.
create table employee (
    empid int,              # 员工id
    ename varchar(20) ,     # 员工名
    deptid int ,            # 员工的部门id
    salary decimal(10,2)    # 员工工资, 10表示期望等到数字总长度(供参考, 没实际意义),  2表示保留2位小数.
);

-- 3. 添加表数据.
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

-- 4. 查看表数据.
select * from employee;

-- 5. 窗口函数案例.
-- 需求1: 对employee表中按照deptid进行分组，并对每一组的员工按照薪资进行排名：       分组排名.
select * from employee order by salary desc;    -- 全局排序.

-- 分组排名写法如下
select
    *,
    row_number() over(partition by deptid order by salary desc) as rn,
    rank() over(partition by deptid order by salary desc) as rk,
    dense_rank() over(partition by deptid order by salary desc) as dr
from employee;

-- 需求2: 对employee表中按照deptid进行分组，并对每一组的员工按照薪资进行排名，取各组中的前两名员工信息   分组排名, 求TopN.
-- 按照我们的设想, 代码写法如下, 结果报错了, 因为: where后边的字段必须是 表中已有的字段, 而rk是我们通过窗口函数新增的字段, 不属于原表本身字段.
select
    *,
    rank() over(partition by deptid order by salary desc) as rk
from employee
where rk <= 2;

-- 正确如下, 把窗口结果封装成一张新表, 然后做查询即可.
select * from (
      select
        *,
        rank() over(partition by deptid order by salary desc) as rk
      from employee
) t1 where rk <= 2;



-- 扩展语句: case when语句
/*
格式:
    case
        when 条件1 then 结果1
        when 条件2 then 结果2
        ...
        else 结果n
    end [as 别名]

细节: 上述格式的变形版, 如果 when后边的条件, 全都是 等于的操作, 则可以优化写法如下.   (语法糖)
    case 字段名
        when 值1 then 结果1
        when 值2 then 结果2
        ...
        else 结果n
    end [as 别名]
*/

-- 1. 查看源表.
select * from employee;

-- 2. 用 case when优化上述结果, 部门id: 10 -> 蜀国,  20 -> 魏国,  30 -> 吴国
select
    *,
    case
        when deptid = 10 then '蜀国'
        when deptid = 20 then '魏国'
        when deptid = 30 then '吴国'
        else '灭国'
    end as c1
from employee;

-- 我们发现, 上述的 case when语句, 条件都是 等于 的判断, 所以可以优化上述写法, 如下:
select
    *,
    case deptid
        when 10 then '蜀国'
        when 20 then '魏国'
        when 30 then '吴国'
        else '灭国'
    end as c1
from employee;