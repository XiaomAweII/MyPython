# 结论: 一个类只要实现了 __iter__(), __next__()这两个函数, 它就是迭代器对象.
# 作用: 迭代器类似于我们之前学的生成器, 都可以 惰性加载, 即: 用的时候再(生成)(遍历->迭代), 这样可以节省内存空间.
# 应用场景: 如果某个变量(或者某个类的对象), 要经常被遍历, 我们可以设置这个类为: 迭代器类, 这样遍历其对象是会非常节省内存资源.

import sys      # system: 系统包.

# 自定义1个迭代器.
class MyIterator:
    # 1. 初始化一个值, 表示该迭代器能迭代到的最大值(最大次数), 这里是为了演示的, 实际开发写不写看需求.
    def __init__(self, max_value):
        self.max_value = max_value  # 表示迭代器能遍历到的最大值.
        self.current_value = 0      # 表示迭代器从哪个值开始遍历.

    # 2. 因为 MyIterator是自定义的迭代器类, 所以它的对象也是: 迭代器对象, 返回即可.
    def __iter__(self):
        return self

    # 3. 从迭代器中, 获取元素. 惰性加载的, 用的时候才会创建-迭代, 节省内存资源.
    def __next__(self):
        # 3.1 限定迭代器迭代的范围.
        if self.current_value >= self.max_value:
            raise StopIteration     # 抛出异常(终止程序)
        # 3.2 记录当前元素值.
        value = self.current_value
        # 3.3 迭代完后, 修改下变量值.
        self.current_value += 1
        # 3.4 返回结果
        return value

if __name__ == '__main__':
    # 1. 迭代 = 逐个获取容器类型(或者迭代器类型)中每个元素的过程.
    list1 = ['A', 'B', 'C']
    for value in list1:
        print(value)    # 底层是调用 next() 逐个获取元素的.

    print('-' * 20)

    # 2. 写一个代码, 你观察下是否使用 惰性加载, 内存占用对比.
    # 列表推导式
    list2 = [i for i in range(1000000)]
    print(type(list2))          # <class 'list'>

    # 生成器
    my_generator = (i for i in range(1000000))
    print(type(my_generator))   # <class 'generator'>, 生成器

    # 3. 查看下(内存)资源占用情况.
    print(sys.getsizeof(list2))         # 8697456
    print(sys.getsizeof(my_generator))  # 112
    print('-' * 20)

    # 4. 创建自定义的迭代器对象
    my_itr = MyIterator(5)
    # 5. 从迭代器中获取元素.
    # print(next(my_itr)) # 0, 底层调用的: my_itr对象的 __next__()函数, 惰性加载, 节省内存.
    # print(next(my_itr)) # 1
    # print(next(my_itr)) # 2
    # print(next(my_itr)) # 3
    # print(next(my_itr)) # 4
    # print(next(my_itr)) # 5

    for i in my_itr:    # for循环的底层, 就要用到: 迭代器.
        print(i)

