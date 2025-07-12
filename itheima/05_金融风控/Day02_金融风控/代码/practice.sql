-- 查看数据库
show databases ;
-- 切换数据库
use financial;
-- 查看表
show tables;

-- 各个阶段转化率表
-- 原则：一旦group by之后，select后面的字段：（1）分组的字段，（2）聚合的字段
select
    t1.id as user_id, -- 用户id
    t3.id as list_id, -- 申请id
    t4.id as order_id, -- 放款id
    date(t1.inserttime) as regist_time,
    max(case when t2.user_id is not null then 1 else 0 end) as if_fillin, -- 注册
    max(case when t3.borrower_id is not null then 1 else 0 end) as if_apply, -- 申请
    max(case when t3.status > 70 then 1 else 0 end) as if_pass, -- 通过,标的状态分超过70算通过，具体要产品来定义
    max(case when t4.borrower_id is not null then 1 else 0 end) as if_loan, -- 借款
    max(case when t4.payment_amount > 0 then 1 else 0 end) as if_pay, -- 还款,已还金额 > 0，说明有还过款
    max(case when t4.owing_principal = 0 then 1 else 0 end) as if_loan_1done -- 还清一笔
from u_user t1 left join u_personal_info t2 on t1.id = t2.user_id
left join loan_list t3 on t1.id = t3.borrower_id
left join loan_debt t4 on t3.id = t4.list_id
group by t1.id,
         t3.id,
         t4.id,
         date(t1.inserttime);

-- 问题：上面已经是一个SQL了，需要再上面SQL基础之上，继续完善SQL，怎么办？

-- 解决方案：（1）子查询；（2）CTE语句（with as语句）
-- CTE语句的语法：
# with 临时表名称1 as(
#     SQL语句1
# ), 临时表名称2 as(
#     SQL语句2
# ), 临时表名称3 as(
#     SQL语句3
# )....
# select colA,colB... from 临时表名称3;

with tmp1 as(
    select
        t1.id as user_id, -- 用户id
        t3.id as list_id, -- 申请id
        t4.id as order_id, -- 放款id
        date(t1.inserttime) as regist_time,
        max(case when t2.user_id is not null then 1 else 0 end) as if_fillin, -- 注册
        max(case when t3.borrower_id is not null then 1 else 0 end) as if_apply, -- 申请
        max(case when t3.status > 70 then 1 else 0 end) as if_pass, -- 通过,标的状态分超过70算通过，具体要产品来定义
        max(case when t4.borrower_id is not null then 1 else 0 end) as if_loan, -- 借款
        max(case when t4.payment_amount > 0 then 1 else 0 end) as if_pay, -- 还款,已还金额 > 0，说明有还过款
        max(case when t4.owing_principal = 0 then 1 else 0 end) as if_loan_1done -- 还清一笔
    from u_user t1 left join u_personal_info t2 on t1.id = t2.user_id
    left join loan_list t3 on t1.id = t3.borrower_id
    left join loan_debt t4 on t3.id = t4.list_id
    group by t1.id,
             t3.id,
             t4.id,
             date(t1.inserttime)
), tmp2 as (
    select
        regist_time,
        count(user_id) as regist_num,-- 注册用户总数
        sum(if_fillin) as fillin_num, -- 填表的用户
        sum(if_apply) as apply_num, -- 申请的用户
        sum(if_pass) as pass_num, -- 申请通过的用户
        sum(if_loan) as loan_num, -- 借款的用户
        sum(if_pay) as pay_num, -- 还款的用户
        sum(if_apply) / count(user_id) as '注册->申请',
        sum(if_pass) / sum(if_apply) as '申请->通过',
        sum(if_loan) / sum(if_pass) as '通过->借款',
        round(sum(if_pay) / sum(if_loan),2) as '借款->还过款',
        sum(if_loan_1done) / sum(if_pay) as '还过款->还清1笔'
    from tmp1
    group by regist_time
)
select * from tmp2;


-- 通过率表
with t1 as (
    select borrower_id,
       min(effective_time) min_effective_time
    from loan_list
    where stage in (80,100)
    group by borrower_id
),t2 as (
    select temp.*,
           case when temp.inserttime > t1.min_effective_time then '老客' else '新客' end as user_type
    from loan_list temp
    left join t1 on t1.borrower_id = temp.borrower_id
),t3 as (
    select date(t4.inserttime) as apply_time,
           t2.user_type,
           t4.period_no,
           t4.term_quantity,
           count(t4.id) as apply_num, -- 总人数
           sum(case when t4.status > 70 then 1 else 0 end) as if_pass_num, -- 通过的人数
           avg(t4.apply_amount) as mean_apply_amount, -- 平均贷款
           concat(round(sum(case when t4.status > 70 then 1 else 0 end)/count(t4.id) * 100,2),'%') as pass_rate -- 通过率
    from loan_list t4
    left join t2 on t4.id = t2.id
    group by apply_time,
             t2.user_type,
             t4.period_no,
             t4.term_quantity
)
select * from t3;

