# 测试类

# 需求: 自定义代码, 模拟: 单向链表.
# 分析结果, 需要两个类: SingleNode(充当节点类, 有元素域 和 链接域).    SingleLinkedList(链表类, 由多个节点连接到一起组成)

# 导包
from singlenode import SingleNode
from singleLinkedList import SingleLinkedList

# 3. 在main方法中测试.
if __name__ == '__main__':
    # 3.1 创建 节点类的对象.
    sn1 = SingleNode('乔峰')  # 节点1

    # 3.2 打印节点对象的 属性值.
    print(sn1.item)  # 打印节点1的 数值域: 乔峰
    print(sn1.next)  # 打印节点1的 地址域: None
    print('-' * 20)

    # 3.2 创建链表类 对象.
    # ll = SingleLinkedList()     # 空链表, 1个节点都没有.
    ll = SingleLinkedList(sn1)    # 用sn1充当 头结点
    print(f'链表对象: {ll}')
    print(f'头结点: {ll.head}')
    print('-' * 20)

    # 3.3 判断链表是否为空.
    print(f'列表是否为空: {ll.is_empty()}')

    # 3.4 打印列表长度.
    print(f'列表长度: {ll.length()}')
    print('-' * 20)

    # 3.5 测试: 往列表的 头部 添加元素.
    # ll.add('虚竹')
    # ll.add('段誉')

    # 3.6 测试: 往列表的 尾部 添加元素.
    ll.append('虚竹')
    ll.append('段誉')

    # 3.7 测试: insert(), 中间插入.
    ll.insert(-10, '张三')        # 等价于 往头部添加
    ll.insert(20, '李四')    # 等价于 往尾部添加
    ll.insert(2, '王五')     # 等价于 往中间插入

    # 3.8 演示删除元素.
    ll.remove('王五')
    ll.remove('段誉')

    # 3.9 遍历链表.
    ll.travel()
    print("-" * 20)

    # 3.10 测试查找元素.
    print(ll.search('阿朱'))
    print(ll.search('乔峰'))

