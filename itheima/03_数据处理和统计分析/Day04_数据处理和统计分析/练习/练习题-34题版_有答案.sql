-- ******************** 准备动作 ********************
-- 1. 创建数据库.
create database north_wind;
-- 我们一会儿要做的34个题用的数据源是从Git上下载的, 微软的北风项目的源数据.

-- 2. 切换数据库.
use north_wind;

-- 3. 查询所有表.
show tables;

-- 4. 导入北风项目的数据源.


-- ******************** 以下是 34个练习题 ********************
-- 需求1: 选中employees 表的所有数据
select *
from employees;

-- 需求2: 查询每个客户的 ID, company name, contact name, contact title, city, 和 country.并按照国家名字排序
select customer_id, company_name, contact_name, contact_title, city, country
from customers
order by country;

-- 替换快捷键: ctrl + 字母R
-- 需求3: 查询每一个商品的product_name, category_name, quantity_per_unit, unit_price, units_in_stock 并且通过 unit_price 字段排序
-- 方式1: 显示内连接
select product_name,
       category_name,
       quantity_per_unit,
       unit_price,
       units_in_stock
from categories c
         join products p
              on c.category_id = p.category_id
order by unit_price desc;

-- 方式2: 隐式内连接.
select product_name,
       category_name,
       quantity_per_unit,
       unit_price,
       units_in_stock
from categories c,
     products p
where c.category_id = p.category_id
order by unit_price desc;

# 这个题其实还可以使用左外链接实现, 因为: 商品表的分类id字段(category_id) 和 商品分类表的id(category_id)是一一对应的.
select product_name,
       category_name,
       quantity_per_unit,
       unit_price,
       units_in_stock
from products p
         left join categories c on p.category_id = c.category_id
order by unit_price desc;

-- 需求4: 列出所有提供了4种以上不同商品的供应商列表所需字段：supplier_id, company_name, and products_count (提供的商品种类数量).
# 步骤1: 计算每个供应商, 供应的商品的总数.
select s.supplier_id,
       s.company_name,
       count(product_id) products_count
from products p,
     suppliers s
where p.supplier_id = s.supplier_id
group by s.supplier_id, s.company_name;

# 步骤2: 基于上述的步骤, 筛选出 供应了4种以上不同商品的供应商信息.
select s.supplier_id,
       s.company_name,
       count(product_id) products_count
from products p,
     suppliers s
where p.supplier_id = s.supplier_id
group by s.supplier_id, s.company_name
having products_count > 4;

# step3: 验证数据.  例如: 供应商id 12, 公司名  Plutzer Lebensmittelgroßmärkte AG,  5类商品.
select *
from products
where supplier_id = 12;

-- 需求5: 提取订单编号为10250的订单详情, 显示如下信息：
-- product_name, quantity, unit_price （ order_items 表), discount , order_date 按商品名字排序
select o.order_id,
       product_name,
       quantity,
       oi.unit_price,
       discount,
       order_date
from products p,
     order_items oi,
     orders o
where p.product_id = oi.product_id
  and oi.order_id = o.order_id
  and o.order_id = 10250
order by product_name;

-- 需求6: 收集运输到法国的订单的相关信息，包括订单涉及的顾客和员工信息，下单和发货日期等.
select e.employee_id,
       e.last_name,
       e.first_name,   # 员工信息.
       c.customer_id,
       c.company_name, # 客户信息
       o.order_date,
       o.shipped_date  # 下单和发货日期
from employees e
         join orders o on e.employee_id = o.employee_id
         join customers c on o.customer_id = c.customer_id
where ship_country = 'France';

-- 需求7: 提供订单编号为10248的相关信息，包括product name, unit price (在 order_items 表中), quantity（数量）,company_name（供应商公司名字 ，起别名 supplier_name).
select product_name,
       oi.unit_price,
       quantity,
       company_name as supplier_name
from products p,
     order_items oi,
     suppliers s
where p.product_id = oi.product_id
  and p.supplier_id = s.supplier_id
  and order_id = 10248;

-- 需求8:  提取每件商品的详细信息，包括 商品名称（product_name）, 供应商的公司名称 (company_name，在 suppliers 表中),
-- 类别名称 category_name, 商品单价unit_price, 和每单位商品数量quantity per unit
select product_name,
       company_name,
       category_name,
       unit_price,
       quantity_per_unit
from products p,
     categories c,
     suppliers s
where c.category_id = p.category_id
  and p.supplier_id = s.supplier_id;

-- 需求9: 另一种常见的报表需求是查询某段时间内的业务指标, 我们统计2016年7月的订单数量，
# 方式1: between..and..
select count(1) order_cnt
from orders
where order_date between '2016-07-01' and '2016-07-31';
# 方式2: 比较运算符
select count(*) order_cnt
from orders
where order_date >= '2016-07-01'
  and order_date <= '2016-07-31';
# 方式3: 模糊查询
select count(*) order_cnt
from orders
where order_date like '2016-07%';
# 方式4: 函数实现.
select count(*) order_cnt
from orders
where year(order_date) = 2016
  and month(order_date) = 7;


-- 需求11: 统计每个供应商供应的商品种类数量, 结果返回供应商IDsupplier_id
-- ，公司名字company_name ，商品种类数量（起别名products_count )使用 products 和 suppliers 表.
select p.supplier_id,
       company_name,
       count(1) products_count
from products p,
     suppliers s
where p.supplier_id = s.supplier_id
group by
    # p.supplier_id, company_name;      # 标准写法, 根据谁分组, 聚合查询的时候, 就根据谁查询.
    p.supplier_id;
# 上边的 company_name(公司名) 可以省略不写, 因为它(company_name) 和 supplier_id(供应商id)是 一对一的关系.
# 语法糖, 其实就是 简化写法的统称, 什么语言都有.

-- 需求12: 我们要查找ID为10250的订单的总价（折扣前），SUM(unit_price * quantity)
select sum(unit_price * quantity) total_price
from order_items
where order_id = 10250;

-- 需求13:  统计每个员工处理的订单总数, 结果包含员工IDemployee_id，姓名first_name 和 last_name，处理的订单总数(别名 orders_count)
select e.employee_id,
       first_name,
       last_name,
       count(order_id) orders_cnt
from employees e,
     orders o
where e.employee_id = o.employee_id
group by e.employee_id, first_name, last_name;

-- 需求14: 统计每个类别中的库存产品值多少钱？显示三列：category_id, category_name, 和 category_total_value, 如何计算库存商品总价：SUM(unit_price * units_in_stock)。
select c.category_id,
       category_name,
       sum(unit_price * units_in_stock) category_total_value
from categories c,
     products p
where c.category_id = p.category_id
group by c.category_id, category_name;

-- 需求15: 计算每个员工的订单数量.
select e.employee_id, last_name, first_name, count(order_id) order_cnt
from employees e,
     orders o
where e.employee_id = o.employee_id
group by e.employee_id, last_name, first_name;


-- 需求16: 计算每个客户的下订单数 结果包含：用户id、用户公司名称、订单数量（customer_id, company_name, orders_count ）
select c.customer_id, company_name, count(order_id) orders_count
from orders o,
     customers c
where o.customer_id = c.customer_id
group by c.customer_id, company_name;

-- 需求17: 统计2016年6月到2016年7月用户的总下单金额并按金额从高到低排序
-- 结果包含：顾客公司名称company_name 和总下单金额（折后实付金额）total_paid
-- 提示：
-- 计算实际总付款金额： SUM(unit_price quantity (1 - discount))
-- 日期过滤 WHERE order_date >= '2016-06-01' AND order_date < '2016-08-01'
select company_name,
       sum(unit_price * quantity * (1 - discount)) total_paid
from order_items oi,
     orders o,
     customers c
where oi.order_id = o.order_id
  and o.customer_id = c.customer_id
  and order_date between '2016-06-01' and '2016-07-31'
group by company_name
order by total_paid desc;


-- 需求18: 统计客户总数和带有传真号码的客户数量
-- 需要字段：all_customers_count 和 customers_with_fax_count
select count(1) all_customers_count, count(fax) customers_with_fax_count
from customers;

/*
case when语句介绍:
    概述/作用:
        它适用于多种情况的判断和校验, 当满足一种情况之后, 会执行其对应的then语句, 然后整个case.when就结束了.
    格式:
        写法1:
            case
                when 条件1 then 结果1
                when 条件2 then 结果2
                ...
            else 结果n end
        写法2:
            case 列名
                when 值1 then 结果1
                when 值2 then 结果2
                ...
                else 结果n
            end          这个是语法糖.
    执行流程:
        1. 先执行条件1, 看其结果是True(成立) 还是 False(不成立).
        2. 如果是True, 则执行结果1, 然后整个 case.when语句就结束了.
        3. 如果是False, 则执行条件2, 看其结果是True还是False.
        4. 如果是True则执行结果2, 然后整个 case.when语句就结束了, 如果是False, 则继续执行条件3, 以此类推.
        5. 如果所有的条件都不满足, 则执行else, 最后遇到end, case when语句结束.
*/
-- 需求19: 我们要在报表中显示每种产品的库存量，但我们不想简单地将“ units_in_stock”列放在报表中。报表中只需要一个总体级别，例如低，高：
-- 库存大于100 的可用性为高(high)
-- 50到100的可用性为中等(moderate)
-- 小于50的为低(low)
-- 零库存 为 (none)
select product_id,
       product_name,
       units_in_stock,
       case
           when units_in_stock > 100 then 'high'
           when units_in_stock between 50 and 100 then 'moderate'
           when units_in_stock between 1 and 49 then 'low'
           else 'none'
           end as stock_level # 库存级别
from products;

-- 需求20: 创建一个报表，统计员工的经验水平
-- 显示字段：first_name, last_name, hire_date, 和 experience
-- 经验字段（experience ）：
-- 'junior' 2014年1月1日以后雇用的员工
-- 'middle' 在2013年1月1日之后至2014年1月1日之前雇用的员工
-- 'senior' 2013年1月1日或之前雇用的员工
select first_name,
       last_name,
       hire_date, # 姓名, 入职时间
       case
           when hire_date > '2014-01-01' then 'junior'
           when hire_date > '2013-01-01' and hire_date <= '2014-01-01' then 'middle'
           when hire_date <= '2013-01-01' then 'senior'
           else 'rookie'
           end experience
from employees;


-- 需求21: 我们的商店要针对北美地区的用户做促销活动：任何运送到北美地区（美国，加拿大) 的包裹免运费。 创建报表，查询订单编号为10720~10730 活动后的运费价格
select order_id,
       ship_country,
       freight '优惠前运费',
       case
           when ship_country in ('USA', 'Canada') then 0
           else freight
           end '优惠后的运费'
from orders
where order_id between 10720 and 10730;

-- 需求22: 需求：创建客户基本信息报表, 包含字段：客户id customer_id, 公司名字 company_name
-- 所在国家 country, 使用语言language, 使用语言language 的取值按如下规则
-- Germany, Switzerland, and Austria 语言为德语 'German', 	UK, Canada, the USA, and Ireland -- 语言为英语 'English', 其他所有国家 'Other'
select customer_id,
       company_name,
       country,
       case
           when country in ('Germany', 'Switzerland', 'Austria') then 'German'
           when country in ('UK', 'Canada', 'USA', 'Ireland') then 'English'
           else 'other'
           end language
from customers;

-- 需求23: 需求：创建报表将所有产品划分为素食和非素食两类
-- 报表中包含如下字段：产品名字 product_name, 类别名称 category_name
-- 膳食类型 diet_type:
-- 	非素食 'Non-vegetarian' 商品类别字段的值为 'Meat/Poultry' 和 'Seafood'.
-- 	素食
select product_name,
       category_name,
       case
           when category_name in ('Meat/Poultry', 'Seafood') then 'Non-vegetarian'
           else 'vegetarian'
           end diet_type
from products p,
     categories c
where p.category_id = c.category_id;

-- 需求24: 在引入北美地区免运费的促销策略时，我们也想知道运送到北美地区和其它国家地区的订单数量
-- 促销策略, 参见需求21的代码. 即:  北美指的是 'USA', 'Canada'
# 方式1: 不起别名.
select case
           when ship_country in ('USA', 'Canada') then '北美地区'
           else '其它国家地区'
           end,
       count(order_id) order_cnt
from orders
group by case
             when ship_country in ('USA', 'Canada') then '北美地区'
             else '其它国家地区'
             end;
# 方式2: 起别名.
select case
           when ship_country in ('USA', 'Canada') then '北美地区'
           else '其它国家地区'
           end         new_country,
       count(order_id) order_cnt
from orders
group by new_country;

# 数据验真
select *
from orders
where ship_country not in ('USA', 'Canada');

-- 需求25: 创建报表统计供应商来自那个大洲, 报表中包含两个字段：供应商来自哪个大洲（supplier_continent ）和 供应产品种类数量（product_count）
-- 供应商来自哪个大洲（supplier_continent ）包含如下取值：
-- 'North America' （供应商来自 'USA' 和 'Canada'.）
-- 'Asia' （供应商来自 'Japan' 和 'Singapore')
-- 'Other' (其它国家)
select case
           when country in ('USA', 'Canada') then 'North America'
           when country in ('Japan', 'Singapore') then 'Asia'
           else 'other'
           end           supplier_continent,
       count(product_id) product_count
from products p,
     suppliers s
where p.supplier_id = s.supplier_id
group by supplier_continent;


-- 需求26: 需求：创建一个简单的报表来统计员工的年龄情况
-- 报表中包含如下字段
-- 年龄（ age ）：生日大于1980年1月1日 'young' ，其余'old'
--  员工数量 （ employee_count）
select case
           when birth_date > '1980-01-01' then 'young'
           else 'old'
           end  age,
       count(1) employee_count
from employees
group by age;

# 数据验真
select *
from employees
where birth_date > '1980-01-01';

-- 需求27: 统计客户的contact_title 字段值为 ’Owner' 的客户数量
-- 查询结果有两个字段：represented_by_owner 和 not_represented_by_owner
# 方式1: case when + 分组
select case
           when contact_title != 'Owner' then 'Not_Owner'
           else 'Owner'
           end            title,
       count(customer_id) customer_cnt
from customers
group by title;
# Owner: 17,  Not_Owner: 74

# 验真
select *
from customers
where contact_title != 'Owner';

# 方式2: case when
select count(case when contact_title = 'Owner' then 1 else null end)  represented_by_owner,    # Owner: 17人
       count(case when contact_title != 'Owner' then 1 else null end) not_represented_by_owner # not_Owner: 74人
from customers;

# 方式3: case when的语法糖.
select count(case contact_title when 'Owner' then 1 else null end) represented_by_owner,    # Owner: 17人
       count(case contact_title when 'Owner' then null else 1 end) not_represented_by_owner # not_Owner: 74人
from customers;

# 方式4: if()函数, 格式: if(条件, 值1, 值2),   执行流程: 先判断条件, 如果是True返回值1, 如果是False返回值2
select count(if(contact_title = 'Owner', 1, null))  represented_by_owner,    # Owner: 17人
       count(if(contact_title != 'Owner', 1, null)) not_represented_by_owner # not_Owner: 74人
from customers;

-- 需求28: Washington (WA) 是 Northwind的主要运营地区，统计有多少订单是由华盛顿地区的员工处理的，
-- 多少订单是有其它地区的员工处理的
-- 结果字段： orders_wa_employees 和 orders_not_wa_employees
select count(if(region = 'WA', 1, null))  '华盛顿员工处理',
       count(if(region != 'WA', 1, null)) '非华盛顿员工处理'
from employees e,
     orders o
where e.employee_id = o.employee_id;

-- 需求29: 创建报表，统计不同类别产品的库存量，将库存量分成两类 >30 和 <=30 两档分别统计数量
-- 报表包含三个字段, 类别名称 category_name, 库存充足 high_availability, 库存紧张 low_availability
# Step1: 简化需求, 统计每类商品的总库存.
select category_name,
       sum(units_in_stock) total_stock
from categories c,
     products p
where c.category_id = p.category_id
group by category_name;

# Step2: 基于上一步的操作, 统计每类商品 库存充足 和 库存紧张的分别有多少.
select category_name,
       sum(if(units_in_stock > 30, units_in_stock, 0))  high_availability,
       sum(if(units_in_stock <= 30, units_in_stock, 0)) low_availability
from categories c,
     products p
where c.category_id = p.category_id
group by category_name;


-- 需求30: 创建报表统计运输到法国的的订单中，打折和未打折订单的总数量
-- 结果包含两个字段：full_price （原价）和 discounted_price（打折）
select count(if(discount = 0, 1, null))  full_price,      -- 1317
       count(if(discount != 0, 1, null)) discounted_price -- 838
from orders o,
     order_items oi
where oi.order_id = o.order_id
  and ship_country = 'France';

# 验真
select count(1)
from order_items oi,
     orders o
where oi.order_id = o.order_id
  and discount = 0;


-- 需求31: 输出报表，统计不同供应商供应商品的总库存量，以及高价值商品的库存量（单价超过40定义为高价值）
-- 结果显示四列：
-- 供应商ID supplier_id
-- 供应商公司名 company_name
-- 由该供应商提供的总库存 all_units
-- 由该供应商提供的高价值商品库存 expensive_units
select p.supplier_id,
       company_name,
       sum(units_in_stock)                            all_units,
       sum(if(unit_price > 40, units_in_stock, null)) expensive_units
from products p,
     suppliers s
where p.supplier_id = s.supplier_id
group by p.supplier_id, company_name;


-- 需求32: 创建报表来为每种商品添加价格标签，贵、中等、便宜
-- 结果包含如下字段：product_id, product_name, unit_price, 和 price_level
-- 价格等级price_level的取值说明：
-- 'expensive' 单价高于100的产品
-- 'average' 单价高于40但不超过100的产品
-- 'cheap' 其他产品

select product_id,
       product_name,
       unit_price,
       case
           when unit_price > 100 then 'expensive'
           when unit_price > 40 and unit_price <= 100 then 'average'
           else 'cheap'
           end price_level
from products;

-- 需求33: 制作报表统计所有订单的总价（不计任何折扣）对它们进行分类。
-- 包含以下字段：
-- 	order_id
-- 	total_price（折扣前）
-- 	price_group
-- 字段 price_group 取值说明：
-- 	'high' 总价超过2000美元
-- 	'average'，总价在$ 600到$ 2,000之间，包括两端
-- 	'low' 总价低于$ 600
# 方式1: 多组case when实现.
select order_id,
       sum(unit_price * quantity) total_price, # 订单编号, 该订单的总价
       case
           when sum(unit_price * quantity) > 2000 then 'high'
           when sum(unit_price * quantity) between 600 and 2000 then 'average'
           when sum(unit_price * quantity) < 600 then 'low'
           end                    price_group  # 根据订单总价, 设置 价格分组.
from order_items
group by order_id;

# 方式2: 套表.
select *,
       case
           when total_price > 2000 then 'high'
           when total_price between 600 and 2000 then 'average'
           when total_price < 600 then 'low'
           end price_group
from (
         select order_id,
                sum(unit_price * quantity) total_price
         from order_items
         group by order_id
     ) t1;

-- 需求34: 统计所有订单的运费，将运费高低分为三档
-- 报表中包含三个字段
-- low_freight freight值小于“ 40.0”的订单数
-- avg_freight freight值大于或等于“ 40.0”但小于“ 80.0”的订单数
-- high_freight freight值大于或等于“ 80.0”的订单数
select
       order_id, freight,
       case
           when freight < 40.0 then 'low_freight'
           when freight >=40 and freight < 80.0 then 'avg_freight'
           when freight >= 80.0 then 'high_freight'
       end freight_level
from orders
group by order_id;