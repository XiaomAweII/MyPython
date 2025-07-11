"""
需求:
    定义一个 地瓜类, 属性为: 被烤的时间(cook_time), 地瓜的生熟状态(cook_state), 添加的调料(condiments).
    行为有: cook() 表示烘烤,  add_condiment() 表示 添加调料.
    请用所学, 用面向对象的思维完成这个事情.

烘烤规则(时间及其对应的状态):
    0 ~ 3分钟     生的
    3 ~ 7分钟     半生不熟
    7 ~ 12分钟    熟了
    超过12分钟     已烤焦, 糊了
"""

# 1. 定义地瓜类.
class SweetPotato():
    # 2. 定义 init()魔法方法, 用来对: 属性完成初始化.
    def __init__(self):
        """
        init()魔法方法, 用来对: 属性完成初始化.
        """
        self.cook_time = 0          # 烘烤时间
        self.cook_state = "生的"     # 地瓜状态
        self.condiments = []        # 添加的调料

    # 3. 具体的烘烤动作, 接收: 烘烤时间, 根据时间, 调整地瓜的: 烘烤状态.
    def cook(self, time):
        # 3.1 非法值校验.
        if time < 0:
            print('传入的烘烤时间非法, 请校验后重新传入')
        else:
            # 3.2 走这里, 说明time是 >= 0的, 就修改 烘烤时间.
            self.cook_time = self.cook_time + time

            # 3.3 根据烘烤时间, 判断烘烤状态.
            if 0 <= self.cook_time < 3:
                self.cook_state = "生的"
            elif 3 <= self.cook_time < 7:
                self.cook_state = "半生不熟"
            elif 7 <= self.cook_time <= 12:
                self.cook_state = "熟了"
            else:
                self.cook_state = '已烤焦, 糊了'

    # 4. 具体的添加调料的动作.
    def add_condiment(self, condiment):
        self.condiments.append(condiment)   # 往列表中添加 调料.

    # 5. 打印对象的各个属性值, 即: 地瓜的烘烤时间, 状态, 添加的调料.
    def __str__(self):
        return f'烘烤时间: {self.cook_time}, 地瓜状态: {self.cook_state}, 添加的调料: {self.condiments}'

# 在main函数中完成测试.
if __name__ == '__main__':
    # 6. 创建地瓜对象, 指定: 属性值.
    digua = SweetPotato()

    # 7. 具体的烘烤动作.
    # digua.cook(-10)     # 非法值.
    digua.cook(2)
    digua.cook(5)
    digua.cook(3)
    digua.cook(3)

    # 8. 具体的添加调料的动作.
    digua.add_condiment("辣椒面")
    digua.add_condiment("孜然粉")
    digua.add_condiment("灵魂之汁-浇给...")

    # 9. 打印地瓜的状态.
    print(digua)