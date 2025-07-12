
class MaxHeap:
    # __init__ 方法初始化一个空堆。
    def __init__(self):
        self.heap = []

    # parent, left_child, right_child 方法分别用来找到给定节点的父节点和左右子节点的索引。
    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    # has_left_child 和 has_right_child 方法检查给定节点是否有左子节点或右子节点。
    def has_left_child(self, i):
        return self.left_child(i) < len(self.heap)

    def has_right_child(self, i):
        return self.right_child(i) < len(self.heap)

    # swap 方法交换堆中的两个元素。
    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    # insert 方法向堆中插入一个新的元素，然后使用 heapify_up 方法来维持最大堆的性质。
    def insert(self, key):
        self.heap.append(key)
        self.heapify_up(len(self.heap) - 1)
        if key == self.heap[0]:
            print(f"Inserted value {key} is equal to the max value in the heap.")

    # heapify_up 方法将新插入的元素与其父节点比较，如果新元素更大，则交换它们，直到恢复最大堆的性质。
    def heapify_up(self, i):
        while i != 0 and self.heap[self.parent(i)] < self.heap[i]:
            self.swap(i, self.parent(i))
            i = self.parent(i)

    # extract_max 方法移除并返回堆中的最大元素（堆顶元素），然后用 heapify_down 方法来维持最大堆的性质。
    def extract_max(self):
        if len(self.heap) == 0:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()  # 删除之后返回
        self.heapify_down(0)
        return root

    # heapify_down 方法将堆顶元素与其子节点比较，如果子节点中的最大值大于堆顶元素，则交换它们，直到恢复最大堆的性质。
    def heapify_down(self, i):
        while self.has_left_child(i):
            max_child_index = self.left_child(i)
            if self.has_right_child(i) and self.heap[self.right_child(i)] > self.heap[max_child_index]:
                max_child_index = self.right_child(i)

            if self.heap[i] < self.heap[max_child_index]:
                self.swap(i, max_child_index)
                i = max_child_index
            else:
                break

    # 打印自身索引和值以及父级索引以及值
    def print_heap_with_parent_indices(self):
        # 遍历堆并打印每个元素及其父级索引
        for i in range(len(self.heap)):
            parent_index = self.parent(i)
            if parent_index >= 0:
                print(f"Index: {i},Element: {self.heap[i]}, Parent Index: {parent_index}, Parent Value: {self.heap[parent_index]}")
            else:
                print(f"Index: {i},Element: {self.heap[i]}, Parent Index: None (Root element)")
# 使用示例
max_heap = MaxHeap()
max_heap.insert(10)
max_heap.insert(20)
max_heap.insert(15)     # [10,20,15]
max_heap.insert(30)
max_heap.insert(60)
max_heap.insert(60)
max_heap.print_heap_with_parent_indices()
# print(len(max_heap.heap))
# for i in range(len(max_heap.heap)):
#     print(max_heap.extract_max())  # 输出 20

