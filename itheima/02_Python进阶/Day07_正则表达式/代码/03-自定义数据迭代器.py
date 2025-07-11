"""
案例: 自定义 数据迭代器, 即: 自定义生成器, 从原始文件中读取所有的数据, 然后按照指定条数, 生成每批次的数据.

目的: 为后续的AI模型训练课程做铺垫, 后续训练模型的时候, 不是一次性喂给大批量的数据, 而是分批次来训练的.
"""
import math

# 需求: 读取 jaychou_lyrics.txt 文件的数据, 按照 n条/批次, 生成生成器, 并测试.
# 1. 定义 dataset_loader(), 接受: 每批次的数据条数, 获取 生成器.
def dataset_loader(batch_size):             # 假设: 8条 / 批次
    """
    自定义的数据迭代器(生成器), 按照 n条/批次, 获取生成器.
    :param batch_size: 每批次的数据条数
    :return: 生成器对象.
    """
    # 1.1 读取文件, 获取到所有的数据, 即: readlines()一次性读取所有行, 放到列表中.
    with open('./data/jaychou_lyrics.txt', 'r', encoding='utf-8') as src_f:
        data_lines = src_f.readlines()      # 一次性读取所有行, 放到列表中.

    # 1.2 计算: 数据的总条数.
    line_count = len(data_lines)        # 假设: 共 100 条数据

    # 1.3 计算: 数据的总批次数, 即:  总批次 = 数据总条数 / 每批次的数据条数       细节: 记得向上取整, 例如: 12.3 => 13
    batch_count = math.ceil(line_count / batch_size)        # math.ceil(100 / 8) = 12.5 => 13

    # 1.4 遍历批次总数, 获取到具体的每个批次编号. 例如: 一共5批, 那就是: 0, 1, 2, 3, 4  分别代表这5批.
    for batch_id in range(batch_count):    # batch_id的值: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
        """
            batch_id 就代表着 批次id, 例如: 0代表第1批, 1代表第2批.  假设每批次 8 条数据.
            batch_id = 0, 代表第 1 批,  8条/批次, 则第1批的数据为:  data_lines[0:8],  即: 获取索引为 0 ~ 8的数据, 包左不包右
            batch_id = 1, 代表第 2 批,  8条/批次, 则第2批的数据为:  data_lines[8:16]
            batch_id = 2, 代表第 3 批,  8条/批次, 则第3批的数据为:  data_lines[16:24]
        """
        # 1.5 具体的生成每批次的数据, 然后通过 yield 放到生成器中(并返回生成器)
        yield data_lines[batch_id * batch_size : batch_id * batch_size + batch_size]


# 在main方法中测试
if __name__ == '__main__':
    # 2. 测试上述的函数, 获取 指定条数的 批次数据.
    # my_generator = dataset_loader(5)

    # 3. 获取第1批的数据
    # print(next(my_generator))   # ['想要有直升机\n', '想要和你飞到宇宙去\n', '想要和你融化在一起\n', '融化在宇宙里\n', '我每天每天每天在想想想想著你\n']

    # 4. 获取第2批的数据
    # print(next(my_generator))   # ['这样的甜蜜\n', '让我开始相信命运\n', '感谢地心引力\n', '让我碰到你\n', '漂亮的让我面红的可爱女人\n']

    # 5. 遍历, 获取到每批的数据.
    my_generator = dataset_loader(8)
    for batch_value in my_generator:
        print(batch_value)


    # 比这个数字大的所有整数中, 最小的那个整数.
    # print(math.ceil(10.0))      # ceil(): 向上取整, 天花板数.   10
    # print(math.ceil(10.1))      # ceil(): 向上取整, 天花板数.   11
    # print(math.ceil(10.5))      # ceil(): 向上取整, 天花板数.   11