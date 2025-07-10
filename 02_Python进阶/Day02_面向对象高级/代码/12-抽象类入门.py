"""
抽象类介绍:
    概述:
        在Python中, 抽象类也叫接口, 指的是: 有抽象方法的类, 就叫抽象类.
    抽象方法:
        没有方法体的方法, 即: 空实现的方法, 用pass修饰, 就叫: 抽象方法.
        例如:
            def get_sum():
                pass
    用法:
        抽象类一般充当 父类, 即: 制定整个继承体系的 标准(规范)
        具体的体现, 实现交由 子类来完成.
"""
# 需求: 定义空调类(AC), 制定标准: 制冷, 制热, 左右摆风.   两个厂商(美的, 格力)根据标准, 制作空调.

# 1. 定义空调类, 制定标准: 制冷, 制热, 左右摆风
class AC(object):
    # 抽象方法.
    # 制冷
    def cool_wind(self):
        pass

    # 制热
    def hot_wind(self):
        pass

    # 左右摆风
    def swing_l_r(self):
        pass

# 2. 美的空调.
class Media(AC):
    # 制冷
    def cool_wind(self):
        print('美的空调 核心制冷技术 制作冷风')

    # 制热
    def hot_wind(self):
        print('美的空调 核心制热技术 制作热风')

    # 左右摆风
    def swing_l_r(self):
        print('美的空调 遥控器设置 左右摆风')

# 3. 格力空调.
class Gree(AC):
    # 制冷
    def cool_wind(self):
        print('格力空调 核心制冷技术 制作冷风')

    # 制热
    def hot_wind(self):
        print('格力空调 太阳能技术 制作热风')

    # 左右摆风
    def swing_l_r(self):
        print('格力空调 AI控制 左右摆风')


# 测试代码
if __name__ == '__main__':
    # 4. 测试 美的空调.
    m = Media()
    m.cool_wind()
    m.hot_wind()
    m.swing_l_r()
    print('-' * 20)

    # 5. 测试 格力空调.
    g = Gree()
    g.cool_wind()
    g.hot_wind()
    g.swing_l_r()