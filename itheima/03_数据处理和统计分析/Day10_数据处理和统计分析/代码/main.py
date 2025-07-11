# 导包
from rfm_object import RFM_Object


# 这个是程序的主入口, 所有的代码都是从这里开始执行的.
if __name__ == '__main__':
    # 1. 创建 RFM_Object对象, 表示: RFM案例的 业务对象.
    rfm = RFM_Object()

    # 2. 读取数据源文件, 获取结果(加载, 预处理, 合并).
    data_merge = rfm.get_data_merge()
    # print(data_merge)

    # 3. 具体的计算rfm维度值的过程, 获取结果.
    rfm_gb = rfm.get_rfm_gb(data_merge)
    print(rfm_gb)

    # 4. 绘制3D柱状图
    rfm.rfm_bi(rfm_gb)

    # 5. 导出结果到 本地文件.
    rfm.write_to_localFile(rfm_gb)

    # 6. 导出结果到 MySQL数据库.
    rfm.write_to_mysql(rfm_gb)