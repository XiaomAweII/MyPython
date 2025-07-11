# 自定义代码, 模拟树形结构: 二叉树.

# 1. 定义类Node, 表示: 节点.
class Node:
    # 初始化属性
    def __init__(self, item):
        self.item = item  # 节点的内容
        self.lchild = None  # 左子树
        self.rchild = None  # 右子树


# 2. 定义BinaryTree类, 表示: 二叉树.
class BinaryTree:
    # 初始化属性
    def __init__(self, root=None):
        self.root = root  # root表示: 根节点.

    # 自定义函数add, 表示往: 二叉树中添加元素.
    def add(self, item):
        # 1. 判断 根节点是否为空, 如果为空, 则设置要添加的内容为: 根节点即可.
        if self.root is None:
            self.root = Node(item)
            return  # 细节.

        # 2. 创建队列, 用于记录: 二叉树中的每个元素.
        queue = []
        # 3. 添加 根节点到 队列中, 我们从这里开始查找.
        queue.append(self.root)
        # 4. 循环查找, 要添加的元素, 在二叉树中的 具体添加位置.
        while True:
            # 5. 找到根节点.
            root_node = queue.pop(0)  # 根据索引删除元素, 并返回该元素.
            # 6. 判断当前节点的左子树是否为空, 为空, 就把新节点添加到这里, 不为空, 就将该左子树 加入到 队列中.
            if root_node.lchild is None:
                root_node.lchild = Node(item)
                break
            else:
                # 走到这里, 说明: 左子树不为空, 就将其添加到 队列中
                queue.append(root_node.lchild)

            # 7. 判断当前节点的右子树是否为空, 为空, 就把新节点添加到这里, 不为空, 就将该右子树 加入到 队列中.
            if root_node.rchild is None:
                root_node.rchild = Node(item)
                break
            else:
                # 走到这里, 说明: 右子树不为空, 就将其添加到 队列中
                queue.append(root_node.rchild)

    # 自定义函数 breadth_travel(self), 表示: 遍历二叉树, 获取每个元素, 广度优先.
    def breadth_travel(self):
        # 1. 判断根节点是否为空, 如果为空, 直接返回即可.
        if self.root == None:
            return
        # 2. 走到这里, 说明根节点不为空. 创建队列, 把根节点加到队列中.
        queue = []
        queue.append(self.root)
        # 3. 具体的获取元素的动作, 只要队列中有元素, 我们就一直获取.
        while len(queue) > 0:
            # 4. 打印当前节点的信息.
            node = queue.pop(0)
            print(node.item, end=' ')
            # 5. 判断当前节点左子树, 如果不为空, 就添加到队列中.
            if node.lchild is not None:
                queue.append(node.lchild)
            # 6. 判断当前节点右子树, 如果不为空, 就添加到队列中.
            if node.rchild is not None:
                queue.append(node.rchild)

    # 自定义函数 preorder_travel(self), 表示: 遍历二叉树, 获取每个元素, 深度优先-前序
    def preorder_travel(self, root):  # 根左右
        """
        深度优先, 前序(根左右)
        :param root: 传入的节点.
        :return:
        """
        if root is not None:
            print(root.item, end=' ')          # 根
            self.preorder_travel(root.lchild)  # 递归获取, 左子树
            self.preorder_travel(root.rchild)  # 递归获取, 右子树

    # 自定义函数 inorder_travel(self), 表示: 遍历二叉树, 获取每个元素, 深度优先-中序
    def inorder_travel(self, root):  # 左根右
        """
        深度优先, 中序(左根右)
        :param root: 传入的节点.
        :return:
        """
        if root is not None:
            self.inorder_travel(root.lchild)  # 递归获取, 左子树
            print(root.item, end=' ')  # 根
            self.inorder_travel(root.rchild)  # 递归获取, 右子树

    # 自定义函数 postorder_travel(self), 表示: 遍历二叉树, 获取每个元素, 深度优先-后序
    def postorder_travel(self, root):  # 左右根
        """
        深度优先, 中序(左根右)
        :param root: 传入的节点.
        :return:
        """
        if root is not None:
            self.postorder_travel(root.lchild)  # 递归获取, 左子树
            self.postorder_travel(root.rchild)  # 递归获取, 右子树
            print(root.item, end=' ')  # 根

# 3. 测试上述的功能.
# 3.1 定义函数, 测试: 创建节点 和 二叉树.
def demo01_测试节点和二叉树创建():
    # 创建节点对象.
    node = Node('乔峰')
    print(f'节点的内容: {node.item}')
    print(f'节点的左子树: {node.lchild}')
    print(f'节点的右子树: {node.rchild}')

    # 测试二叉树对象.
    bt = BinaryTree(node)
    print(f'二叉树对象: {bt}')
    print(f'二叉树的根节点的内容: {bt.root.item}')


# 3.2 定义函数, 测试: queue队列的 pop函数()
def demo02_测试队列的pop函数():
    # 创建队列
    queue = []
    # 往队列中添加元素.
    queue.append('A')
    queue.append('B')
    queue.append('C')
    # 打印队列的内容
    print(queue)  # ['A', 'B', 'C']
    # pop() 根据索引删除元素, 并返回该元素.
    print(queue.pop(0))  # ['A', 'B', 'C'] => A
    print(queue.pop(0))  # ['B', 'C'] => B
    print(queue.pop(0))  # ['C'] => C


# 3.3 定义函数, 测试: 广度优先.
def demo03_广度优先():
    # 创建二叉树.
    bt = BinaryTree()
    # 添加元素
    bt.add('A')
    bt.add('B')
    bt.add('C')
    bt.add('D')
    bt.add('E')
    bt.add('F')
    bt.add('G')
    bt.add('H')
    bt.add('I')
    bt.add('J')
    # 广度优先, 遍历.
    bt.breadth_travel()

# 3.4 定义函数, 测试: 深度优先.
def demo04_深度优先():
    # 创建二叉树.
    bt = BinaryTree()
    # 添加元素
    bt.add(0)
    bt.add(1)
    bt.add(2)
    bt.add(3)
    bt.add(4)
    bt.add(5)
    bt.add(6)
    bt.add(7)
    bt.add(8)
    bt.add(9)
    # 深度优先, 遍历.
    # bt.preorder_travel(bt.root)       # 传入根节点,  前序(先序: 根左右)
    # bt.inorder_travel(bt.root)        # 传入根节点,  中序(先序: 左根右)
    bt.postorder_travel(bt.root)        # 传入根节点,  后序(先序: 左右根)

# 4. 在main方法中运行测试代码.
if __name__ == '__main__':
    # demo01_测试节点和二叉树创建()
    # demo02_测试队列的pop函数()
    # demo03_广度优先()
    demo04_深度优先()
