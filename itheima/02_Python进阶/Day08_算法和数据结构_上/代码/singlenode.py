# 1. 创建SingleNode类, 充当节点类, 有元素域 和 链接域.
class SingleNode(object):
    # 初始化属性
    def __init__(self, item):
        self.item = item     # 代表: 数值域(元素域)
        self.next = None     # 代表: 地址域.
