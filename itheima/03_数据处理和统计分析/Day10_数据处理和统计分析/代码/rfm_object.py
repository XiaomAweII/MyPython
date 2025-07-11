# 需求: 通过 面向对象 思维, 实现RFM案例, 即: 定义1个类, 把各个步骤封装成函数, 然后测试调用即可.
# 细节: 以后我们写Numpy, Pandas代码, 不一定非得都要写到 jupyter notebook中, 在普通的python文件中也是可以直接写.

import time  # 时间库
import numpy as np  # numpy库
import pandas as pd  # pandas库
from pyecharts.charts import Bar3D  # 3D柱形图
import os
# 显示图形
from pyecharts.commons.utils import JsCode
import pyecharts.options as opts
from sqlalchemy import create_engine


# 定义 RFM_Object类, 表示: RFM案例整个完整的业务对象.
class RFM_Object(object):       # 所有的类都直接 或者间接 继承自 Object类.
    # 1. 定义函数, get_data_merge(), 从源文件加载数据, 处理数据, 并合并数据.  load_data(), check_data(), merge_data()
    def get_data_merge(self):
        # 1.1 定义列表, 记录: excel表名
        sheet_names = ['2015', '2016', '2017', '2018', '会员等级']
        # 1.2 具体的从数据源文件中读取数据的操作.
        sheet_datas = pd.read_excel('data/sales.xlsx', sheet_name=sheet_names)
        # 1.3 筛选出我们要处理的sheet表名
        for i in sheet_names[:-1]:
            # 1.3.2 从上述的四张表中, 删除空值.
            sheet_datas[i] = sheet_datas[i].dropna()
            # 1.3.3 从上述的四张表中, 筛出订单金额 > 1的值.
            sheet_datas[i] = sheet_datas[i][sheet_datas[i]['订单金额'] > 1]
            # 1.3.4 新增 max_year_date列, 表示: 统计每年数据的基本时间: 年中最后一天.
            sheet_datas[i]['max_year_date'] = sheet_datas[i]['提交日期'].max()
        # 1.4 汇总数据, 把钱四张表结果做汇总
        data_merge = pd.concat(list(sheet_datas.values())[:-1])
        # 1.5 给表新增两列数据, date_interval: 购买间隔时间, year: 订单所属的年费
        data_merge['date_interval'] = data_merge['max_year_date'] - data_merge['提交日期']
        data_merge['date_interval'] = data_merge['date_interval'].dt.days
        # 1.6 给表新增year列, 表示该条数据所属的年份
        data_merge['year'] = data_merge['max_year_date'].dt.year
        return data_merge


    # 2. 定义函数 get_rfm_gb(), 表示具体的 rfm计算动作, 包括: 分组统计, 划分区间, 生成 r,f,m 三个维度的值.
    def get_rfm_gb(self, data_merge):
        # 2.1 按照 年, 会员ID分组, 统计 每年每个会员的: 最小购买间隔时间, 购买总次数, 支付的总金额.
        rfm_gb = data_merge.groupby(['year', '会员ID'], as_index=False).agg({
            'date_interval': 'min',  # 最小购买间隔时间
            '订单号': 'count',  # 总购买次数
            '订单金额': 'sum'  # 总支付金额
        })
        # 2.2 修改上述的列名.
        rfm_gb.columns = ['year', '会员ID', 'r', 'f', 'm']
        # 2.3 划分区间, 分别指定 r(Recency: 最小购买的间隔时间), f(Frequency: 购买频率), m(money: 购买总金额)的区间.
        # 因为我们用的是 三分法, 即: 3个区间, 所以我们要指定 4个值.
        r_bins = [-1, 79, 255, 365]
        f_bins = [0, 2, 5, 130]  # 和业务人员沟通 + 你自己的开发经验
        m_bins = [1, 69, 1199, 206252]
        # 2.4 具体的生成 r,f,m 纬度值的动作
        rfm_gb['r_score'] = pd.cut(rfm_gb['r'], bins=r_bins, labels=[i for i in range(len(r_bins) - 1, 0, -1)])
        rfm_gb['f_score'] = pd.cut(rfm_gb['f'], bins=f_bins, labels=[i for i in range(1, len(f_bins))])
        rfm_gb['m_score'] = pd.cut(rfm_gb['m'], bins=m_bins, labels=[i + 1 for i in range(len(m_bins) - 1)])
        # 2.5 把上述的结果, r_score, f_score, m_score列的类型转成 字符串类型.
        rfm_gb['r_score'] = rfm_gb['r_score'].astype(np.str)
        rfm_gb['f_score'] = rfm_gb['f_score'].astype(np.str)
        rfm_gb['m_score'] = rfm_gb['m_score'].astype(np.str)

        # 2.6 添加 rfm_group列, 表示最终的: 会员等级模型.
        rfm_gb['rfm_group'] = rfm_gb['r_score'] + rfm_gb['f_score'] + rfm_gb['m_score']
        # 2.7 把 rfm_group列的类型转成 int类型.
        rfm_gb['rfm_group'] = rfm_gb['rfm_group'].astype(np.int32)
        # 2.8 获取最终处理后的结果, 即: 一会儿要写到本地的文件 或者 MySQL数据库中.
        return rfm_gb  # 这个就是我们的"最终"分析结果


    # 3. 具体的绘图动作, 绘制 3D柱状图.
    def rfm_bi(self, rfm_gb):
        # 3.1 绘制3D柱状图的时候, 只要3个值, 分别是: 年份, rfm分组, 用户数量
        display_data = rfm_gb.groupby(['rfm_group', 'year'], as_index=False)['会员ID'].count()
        # 4.2 修改类名
        display_data.columns = ['rfm_group', 'year', 'number']
        # 4.3 display_data就是我们最终要展示的数据, 调用 PyEchart框架绘图即可. 直接复制, 不需要写.
        # 如下是具体的绘制动作.
        # 颜色池
        range_color = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf',
                       '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
        range_max = int(display_data['number'].max())
        c = (
            Bar3D()  # 设置了一个3D柱形图对象
            .add(
                "",  # 图例
                [d.tolist() for d in display_data.values],  # 数据
                xaxis3d_opts=opts.Axis3DOpts(type_="category", name='rfm_group'),  # x轴数据类型，名称，rfm_group
                yaxis3d_opts=opts.Axis3DOpts(type_="category", name='year'),  # y轴数据类型，名称，year
                zaxis3d_opts=opts.Axis3DOpts(type_="value", name='number'),  # z轴数据类型，名称，number
            )
            .set_global_opts(  # 全局设置
                visualmap_opts=opts.VisualMapOpts(max_=range_max, range_color=range_color),  # 设置颜色，及不同取值对应的颜色
                title_opts=opts.TitleOpts(title="RFM分组结果"),  # 设置标题
            )
        )
        c.render()  # 数据保存到本地的网页中.
        # c.render_notebook() #在notebook中显示

    # 4. 导出结果数据到, 本地的 excel文件中.
    def write_to_localFile(self, rfm_gb):
        rfm_gb.to_excel('sale_rfm_score.xlsx', index=False)  # index=False, 导出结果时, 不要索引列

    # 5.导出结果到MySQL数据库中.
    def write_to_mysql(self, rfm_gb):
        # 1. 导包
        # from sqlalchemy import create_engine  写到第1个代码块了.

        # 2. 创建引擎对象.
        # 格式: 数据库名 + 协议名://账号:密码@ip地址或者主机名:端口号/要导出数据到的数据库?charset=码表名
        # 前提: 数据库必须存在.
        engine = create_engine("mysql+pymysql://root:123456@localhost:3306/rfm_db?charset=utf8")

        # 3. 具体的导出动作.
        rfm_gb.to_sql('rfm_table', engine, index=False, if_exists='append')