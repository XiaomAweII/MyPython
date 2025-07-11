# 导包
from singlenode import SingleNode

# 2. 创建 SingleLinkedList(链表类, 由多个节点连接到一起组成)
class SingleLinkedList(object):
    # 初始化属性, 用 head属性, 指向: 链表的头结点
    def __init__(self, node=None):
        self.head = node

    # is_empty(self) 链表是否为空
    def is_empty(self):
        """
        判断链表是否为空.
        :return: True => 为空, False => 不为空
        """
        # 判断 self.head 属性是否为None, 是: 空.  不是: 不为空.
        # 思路1: if else 写法.
        # if self.head == None:
        #     return True
        # else:
        #     return False

        # 思路2: 三元运算符.
        # return True if self.head == None else False

        # 思路3: 直接判断.
        return self.head == None

    # length(self) 链表长度
    def length(self):
        # 定义变量 count = 0 表示链表的长度.
        count = 0
        # 定义变量 cur, 表示: 当前节点, 它是从 头结点开始 往下逐个获取的.
        cur = self.head
        # 循环往后获取每个节点即可, 只要 cur不为None, 就说明还有节点.
        while cur is not None:
            # 走这里, 说明有节点, 计数器加1, 然后设置 cur为 当前节点的下个节点即可.
            count += 1
            cur = cur.next
        return count

    # travel(self. ) 遍历整个链表, 打印每个节点的 数值域(的值)
    def travel(self):
        # 定义变量 cur, 表示: 当前节点, 它是从 头结点开始 往下逐个获取的.
        cur = self.head
        # 循环往后获取每个节点即可, 只要 cur不为None, 就说明还有节点.
        while cur is not None:
            # 走这里, 说明有节点, 打印当前节点的 元素域(数值域) item属性.
            print(cur.item)
            # 重新设置 cur 为: 当前节点的下个节点.
            cur = cur.next

    # add(self, item) 链表头部添加元素
    def add(self, item):
        # 把 item 封装成 节点
        new_node = SingleNode(item)
        # 设置 新节点的地址域 指向 之前旧的头结点.
        new_node.next = self.head
        # 设置新节点为: 新的头结点.
        self.head = new_node

    # append(self, item) 链表尾部添加元素
    def append(self, item):
        # 把 item 封装成 节点
        new_node = SingleNode(item)
        # 判断链表是否为空, 如果为空, 则: 新节点充当头结点即可.
        if self.is_empty():
            # 为空, 新节点充当头结点即可
            self.head = new_node
        else:
            # 不为空, 找到 最后1个节点, 设置它的 地址域为 新节点即可.
            cur = self.head     # cur代表当前的节点, 从头结点往后 逐个获取.
            while cur.next is not None:
                cur = cur.next      # 获取当前节点的 下个节点
            # 走这里, cur就是最后1个节点, 设置它的地址域为: 新节点即可.
            cur.next = new_node


    # insert(self, pos, item) 指定位置添加元素
    def insert(self, pos, item):
        """
        往指定的位置, 添加元素.
        :param pos: 要添加元素到的 位置(索引).
        :param item: 具体的要添加的元素.
        :return: 无.
        """
        # 场景1: 如果pos的值 <= 0, 就往: 头部添加.
        if pos <= 0:
            self.add(item)
        # 场景2: 如果 pos的值 >= 链表的长度, 就往: 尾部添加.
        elif pos >= self.length():
            self.append(item)
        # 场景3: 走到这里, 说明, 插入位置, 是在链表中间的. 需要找到 插入位置前的哪个节点.
        else:
            # 定义变量 count 用于表示: 插入位置前的那个元素的 索引.
            count= 0
            # 定义变量 cur, 表示: 当前元素(节点), 找到: 插入位置前的哪个节点.
            cur = self.head
            # 遍历, 判断: 只要 count的值 < pos - 1要小, 就一致循环.  循环结束后: count = pos - 1
            while count < pos - 1:
                count += 1
                cur = cur.next
            # 设置 新节点的地址域为: cur节点的地址域.
            new_node = SingleNode(item)
            new_node.next = cur.next
            # 设置 cur节点的地址域为: 新节点(的地址)
            cur.next = new_node

    # remove(self, item) 删除节点
    def remove(self, item):
        # 定义变量 cur, 记录: 当前的节点, 默认: 从头结点开始.
        cur = self.head
        # 定义变量 pre, 记录: 当前节点的 前1个节点(前驱节点)
        pre = None
        # 遍历, 获取到每个节点.
        while cur is not None:
            # 判断当前节点是否是要被删除的节点, 即: 当前节点的 数值域 是否等于 item参数
            if cur.item == item:
                # 走这里, 说明 cur就是要被删除的节点.
                # 场景1: 有可能要删除的节点是: 头结点.
                if cur == self.head:
                    self.head = cur.next   # cur的下个节点作为新的 头结点即可.
                    cur = None             # 断开旧的头结点 和 后续节点的链接.
                else:
                    # 场景2: 要删除的节点不是头结点.
                    pre.next = cur.next
                    cur = None      # 断开当前节点(要被删除的节点) 和 后续节点的链接.
                # 核心细节: 删除之后, 程序结束.
                break
            else:
                # 走这里, 说明 cur不是要被删除的节点, 继续往后拿.
                pre = cur       # pre表示刚才判断过的节点
                cur = cur.next  # 更新cur为它的下个节点.


    # search(self, item) 查找节点是否存在
    def search(self, item):
        # 定义cur变量(cursor: 游标, current: 当前), 从头结点开始往后找.
        cur = self.head
        # 循环获取到每个节点.
        while cur is not None:
            # 判断当前节点的 元素域(数值域) 是否 和 要查找的值 一致.
            if cur.item == item:
                return True     # 找到了
            cur = cur.next  # 不匹配, 就开始校验下个节点.
        # 走到这里, 整个while都结束了, 即: 没有找到.
        return False


