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
from employees
limit 100;

-- 需求2: 查询每个客户的 ID, company name, contact name, contact title, city, 和 country.并按照国家名字排序
select orders.employee_id ID,
       customers.company_name,
       customers.contact_name,
       contact_title,
       customers.city,
       customers.country
from employees,
     orders,
     customers
where employees.employee_id = orders.employee_id
  and orders.customer_id = customers.customer_id
limit 100;

-- 替换快捷键: ctrl + 字母R
-- 需求3: 查询每一个商品的product_name, category_name, quantity_per_unit, unit_price, units_in_stock 并且通过 unit_price 字段排序
-- 方式1: 显示内连接
select products.product_name, category_name, quantity_per_unit, unit_price, units_in_stock
from products
         inner join
     categories
     on products.category_id = categories.category_id
limit 10;
-- 需求3: 查询每一个商品的product_name, category_name, quantity_per_unit, unit_price, units_in_stock 并且通过 unit_price 字段排序
-- 方式2: 隐式内连接.
select products.product_name, category_name, quantity_per_unit, unit_price, units_in_stock
from products,
     categories
where products.category_id = categories.category_id
limit 10;


-- 需求4: 列出所有提供了4种以上不同商品的供应商列表所需字段：supplier_id, company_name, and products_count (提供的商品种类数量).
select suppliers.supplier_id, company_name, count(product_id) products_count
from suppliers,
     products
where products.supplier_id = suppliers.supplier_id
group by products.supplier_id
having products_count > 4
limit 10;

-- 需求5: 提取订单编号为10250的订单详情, 显示如下信息：
-- product_name, quantity, unit_price （ order_items 表), discount , order_date 按商品名字排序
select product_name, quantity, order_items.unit_price, discount, order_date
from orders,
     products,
     order_items
where orders.order_id = order_items.order_id
  and order_items.product_id = products.product_id
  and orders.order_id = 10250
order by product_name;

-- 需求6: 收集运输到法国的订单的相关信息，包括订单涉及的顾客和员工信息，下单和发货日期等.
select orders.customer_id, orders.employee_id, orders.order_date, orders.shipped_date
from orders
where ship_country = "France";

-- 需求7: 提供订单编号为10248的相关信息，包括product name, unit price (在 order_items 表中), quantity（数量）,company_name（供应商公司名字 ，起别名 supplier_name).
select products.product_name, order_items.unit_price, quantity, company_name supplier_name
from products,
     order_items,
     suppliers
where order_items.product_id = products.product_id
  and products.supplier_id = suppliers.supplier_id
  and order_id = 10248

-- 需求8:  提取每件商品的详细信息，包括 商品名称（product_name）, 供应商的公司名称 (company_name，在 suppliers 表中),
-- 类别名称 category_name, 商品单价unit_price, 和每单位商品数量quantity per unit
select products.product_name, company_name, category_name, unit_price, quantity_per_unit
from products,
     suppliers,
     categories
where products.supplier_id = suppliers.supplier_id
  and categories.category_id = products.category_id


-- 需求9: 另一种常见的报表需求是查询某段时间内的业务指标, 我们统计2016年7月的订单数量，
select count(order_id) cnt
from orders
where order_date between "2016-07-01 00:00:00" and "2016-07-31 00:00:00"

# desc orders

-- 需求11: 统计每个供应商供应的商品种类数量, 结果返回供应商ID supplier_id
-- ，公司名字company_name ，商品种类数量（起别名products_count )使用 products 和 suppliers 表.
select suppliers.supplier_id, company_name, count(products.category_id) products_count
from suppliers,
     products
group by suppliers.supplier_id
;

-- 需求12: 我们要查找ID为10250的订单的总价（折扣前），SUM(unit_price * quantity)
select sum(unit_price * quantity) `SUM`
from order_items
where order_id = 10250;

-- 需求13:  统计每个员工处理的订单总数, 结果包含员工IDemployee_id，姓名first_name 和 last_name，处理的订单总数(别名 orders_count)
select employees.employee_id, first_name, last_name, count(order_id) orders_count
from employees,
     orders
where employees.employee_id = orders.employee_id
group by employee_id
;

-- 需求14: 统计每个类别中的库存产品值多少钱？显示三列：category_id, category_name, 和 category_total_value, 如何计算库存商品总价：SUM(unit_price * units_in_stock)。
select categories.category_id, categories.category_name, sum(unit_price * units_in_stock) category_total_value
from categories,
     products
where categories.category_id = products.category_id
group by categories.category_id
;

select c.category_id, category_name, sum(unit_price * units_in_stock) category_total_value
from categories c, products p
where c.category_id = p.category_id
group by  c.category_id
;

-- 需求15: 计算每个员工的订单数量
select employees.employee_id, count(orders.order_id) total_cnt
from orders,
     employees
where employees.employee_id = orders.employee_id
group by employees.employee_id


-- 需求16: 计算每个客户的下订单数 结果包含：用户id、用户公司名称、订单数量（customer_id, company_name, orders_count ）
select customers.customer_id, company_name, count(order_id) orders_count
from customers,
     orders
where customers.customer_id = orders.customer_id
group by customers.customer_id
;
-- 需求17: 统计2016年6月到2016年7月用户的总下单金额并按金额从高到低排序
-- 结果包含：顾客公司名称company_name 和总下单金额（折后实付金额）total_paid
-- 提示：
-- 计算实际总付款金额： SUM(unit_price quantity (1 - discount))
-- 日期过滤 WHERE order_date >= '2016-06-01' AND order_date < '2016-08-01'
select customers.company_name, sum(unit_price * quantity * (1-discount)) total_paid
from customers,
     orders,
     order_items
where customers.customer_id = orders.customer_id
  and orders.order_id = order_items.order_id
  and order_date >= '2016-06-01'
  AND order_date < '2016-08-01'
group by customers.company_name
order by total_paid desc;


-- 需求18: 统计客户总数和带有传真号码的客户数量
-- 需要字段：all_customers_count 和 customers_with_fax_count
select count(1) all_customers_count
from customers;
select count(1) customers_with_fax_count
from customers
where fax is not null;

select
    count(fax)   customers_with_fax_count
from customers
-- 需求19: 我们要在报表中显示每种产品的库存量，但我们不想简单地将“ units_in_stock”列放在报表中。报表中只需要一个总体级别，例如低，高：
-- 库存大于100 的可用性为高(high)
-- 50到100的可用性为中等(moderate)
-- 小于50的为低(low)
-- 零库存 为 (none)
select products.product_id,
       products.product_name,
       case
           when units_in_stock > 100 then "high"
           when units_in_stock between 50 and 100 then "moderate"
           when units_in_stock < 50 and units_in_stock > 0 then "low"
           else "none"
           end as level
from products
;
-- 需求20: 创建一个报表，统计员工的经验水平
-- 显示字段：first_name, last_name, hire_date, 和 experience
-- 经验字段（experience ）：
-- 'junior' 2014年1月1日以后雇用的员工
-- 'middle' 在2013年1月1日之后至2014年1月1日之前雇用的员工
-- 'senior' 2013年1月1日或之前雇用的员工
select first_name,
       last_name,
       hire_date,
       case
           when hire_date > "2014-01-01" then "junior"
           when hire_date > "2013-01-01" and hire_date <= "2014-01-01" then "moddle"
           else "senior"
           end as experience
from employees;

-- 需求21: 我们的商店要针对北美地区的用户做促销活动：任何运送到北美地区（美国，加拿大) 的包裹免运费。 创建报表，查询订单编号为10720~10730 活动后的运费价格
select order_id,
       ship_country,
       if(ship_country in ("USA", "Canada"), 0, freight) as new_freight
from orders
where order_id between 10720 and 10730;



-- 需求22: 需求：创建客户基本信息报表, 包含字段：客户id customer_id, 公司名字 company_name
-- 所在国家 country, 使用语言language, 使用语言language 的取值按如下规则
-- Germany, Switzerland, and Austria 语言为德语 'German', 	UK, Canada, the USA, and Ireland -- 语言为英语 'English', 其他所有国家 'Other'
select customer_id,
       company_name,
       case
           when country in ("Germany", "Switzerland", "Austria") then 'German'
           when country in ("UK", "Canada", "USA", "Ireland") then 'English'
           else "Other"
           end as language
from customers;

-- 需求23: 需求：创建报表将所有产品划分为素食和非素食两类
-- 报表中包含如下字段：产品名字 product_name, 类别名称 category_name
-- 膳食类型 diet_type:
-- 	非素食 'Non-vegetarian' 商品类别字段的值为 'Meat/Poultry' 和 'Seafood'.
-- 	素食
select products.product_name,
       category_name,
       if(category_name in ('Meat/Poultry', 'Seafood'), 'Non-vegetarian', "Vegetarian") as diet_type
from products,
     categories
where products.category_id = categories.category_id;

-- 需求24: 在引入北美地区免运费的促销策略时，我们也想知道运送到北美地区和其它国家地区的订单数量
-- 促销策略, 参见需求21的代码.
/*
-- 需求21: 我们的商店要针对北美地区的用户做促销活动：任何运送到北美地区（美国，加拿大) 的包裹免运费。 创建报表，查询订单编号为10720~10730 活动后的运费价格
select orders.order_id,
       orders.ship_country,
       if(ship_country in ("USA", "Canada"), 0, freight) as new_freight
from orders
where order_id between 10720 and 10730;
*/
select
    count(case when orders.ship_country in ("USA", "Canada") then 0 end) North_America,
    count(case when orders.ship_country not in ("USA", "Canada") then 1 end) Other
from orders;


-- 需求25: 创建报表统计供应商来自那个大洲, 报表中包含两个字段：供应商来自哪个大洲（supplier_continent ）和 供应产品种类数量（product_count）
-- 供应商来自哪个大洲（supplier_continent ）包含如下取值：
-- 'North America' （供应商来自 'USA' 和 'Canada'.）
-- 'Asia' （供应商来自 'Japan' 和 'Singapore')
-- 'Other' (其它国家)
select supplier_continent, count(company_name) product_count
from (select suppliers.company_name,
             case
                 when country in ('USA', 'Canada') then 'North America'
                 when country in ('Japan', 'Singapore') then 'Asia'
                 else 'Other'
                 end supplier_continent
      from suppliers,
           products
      where suppliers.supplier_id = products.supplier_id) t1
group by t1.supplier_continent;

# group by suppliers.company_name


-- 需求26: 需求：创建一个简单的报表来统计员工的年龄情况
-- 报表中包含如下字段
-- 年龄（ age ）：生日大于1980年1月1日 'young' ，其余'old'
--  员工数量 （ employee_count）
select count(case when birth_date > "1980-01-01" then 1 end)  'young',
       count(case when birth_date <= "1980-01-01" then 1 end) 'old'
from employees



-- 需求27: 统计客户的contact_title 字段值为 ’Owner' 的客户数量
-- 查询结果有两个字段：represented_by_owner 和 not_represented_by_owner
select count(case when contact_title = "Owner" then 1 end)  represented_by_owner,
       count(case when contact_title != "Owner" then 1 end) not_represented_by_owner
from customers;

-- 需求28: Washington (WA) 是 Northwind的主要运营地区，统计有多少订单是由华盛顿地区的员工处理的，
-- 多少订单是有其它地区的员工处理的
-- 结果字段： orders_wa_employees 和 orders_not_wa_employees
select count(case when region = "WA" then 1 end)            orders_wa_employees,
       count(case when employees.region != "WA" then 1 end) orders_not_wa_employees
from employees,
     orders
where employees.employee_id = orders.employee_id;

-- 需求29: 创建报表，统计不同类别产品的库存量，将库存量分成两类 >30 和 <=30 两档分别统计数量
-- 报表包含三个字段, 类别名称 category_name, 库存充足 high_availability, 库存紧张 low_availability
-- 简化需求: 统计不同类别产品的库存量
select category_name,
       sum(case when units_in_stock > 30 then units_in_stock end)  as high_availability,
       sum(case when units_in_stock <= 30 then units_in_stock end) as low_availability
from products p
         join categories c on p.category_id = c.category_id
group by category_name


-- 需求30: 创建报表统计运输到法国的的订单中，打折和未打折订单的总数量
-- 结果包含两个字段：full_price （原价）和 discounted_price（打折）
-- select ship_country, discount from orders o, order_items oi where ship_country='France' and o.order_id = oi.order_id;  -- 184
select count(case when discount > 0 then 1 end) discounted_price,
       count(case when discount = 0 then 1 end) full_price
from (select discount
      from orders,
           order_items
      where orders.order_id = order_items.order_id
        and ship_country = "France") t1;
-- 需求31: 输出报表，统计不同供应商供应商品的总库存量，以及高价值商品的库存量（单价超过40定义为高价值）
-- 结果显示四列：
-- 供应商ID supplier_id
-- 供应商公司名 company_name
-- 由该供应商提供的总库存 all_units
-- 由该供应商提供的高价值商品库存 expensive_units
select products.supplier_id,
       suppliers.company_name,
       sum(units_in_stock)                                    all_units,
       sum(case when unit_price > 40 then units_in_stock end) expensive_units
from products,
     suppliers
where products.supplier_id = suppliers.supplier_id
group by products.supplier_id;

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
           when unit_price >= 40 and unit_price <= 100 then 'average'
           else 'cheap'
           end as price_level
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
select order_id,
       sum(unit_price * quantity) total_price,
       case
           when sum(unit_price * quantity) > 2000 then "high"
           when sum(unit_price * quantity) >= 600 and sum(unit_price * quantity) <= 2000 then 'average'
           else 'low'
           end as                 price_group
from order_items
group by order_id;


-- 需求34: 统计所有订单的运费，将运费高低分为三档
-- 报表中包含三个字段
-- low_freight freight值小于“ 40.0”的订单数
-- avg_freight freight值大于或等于“ 40.0”但小于“ 80.0”的订单数
-- high_freight freight值大于或等于“ 80.0”的订单数
select count(case when orders.freight < 40.0 then 0 end)                            as low_freight,
       count(case when orders.freight >= 40.0 and orders.freight < 80.0 then 1 end) as avg_freight,
       count(case when orders.freight >= 80.0 then 2 end)                           as high_freight
from orders;


-- 需求35: 统计所有客户消费总金额top10

-- 需求36: 统计订单数前top10员工