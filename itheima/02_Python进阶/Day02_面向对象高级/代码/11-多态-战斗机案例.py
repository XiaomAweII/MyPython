"""
需求:
    构建对象对战平台object_play
        1 英雄一代战机（战斗力60）与敌军战机（战斗力70）对抗。英雄1代战机失败！
        2 卧薪尝胆，英雄二代战机（战斗力80）出场！，战胜敌军战机！
        3 对象对战平台object_play, 代码不发生变化的情况下, 完成多次战斗.

分析:
    抽象战机类 HeroFighter  AdvHeroFighter；敌机EnemyFighter;
    构建对象战斗平台, 使用多态实现

"""


# 1. 定义英雄1代机 战斗力为: 60
class HeroFighter(object):
    def power(self):
        return 60  # 攻击力: 60


# 2. 定义英雄2代机 战斗力为: 80,  继承自: 英雄1代机.
class AdvHeroFighter(HeroFighter):  # advanced: 进阶...
    def power(self):
        return 80  # 攻击力: 80


# 3. 定义敌机, 战斗力为: 70
class EnemyFighter(object):
    def power(self):
        return 70  # 攻击力: 70


# 4. 构建对象对战平台 object_play(英雄机, 敌机)
def object_play(hf: HeroFighter, ef: EnemyFighter):
    """
    构建对象对战平台, 模拟: 英雄机 和 敌机对战.
    :param hf: 英雄机
    :param ef: 敌机
    :return: 无
    """
    if hf.power() >= ef.power():
        print('英雄机 获胜!')
    else:
        print('敌机 获胜!')

# 在main方法中测试.
if __name__ == '__main__':
    # 5. 分别创建英雄1代机, 2代机, 敌机对象.
    hf = HeroFighter()
    hf2 = AdvHeroFighter()
    ef = EnemyFighter()

    # 6. 具体的对战过程.
    # 英雄1代机 和 敌机
    object_play(hf, ef)         # 敌机 获胜!
    # 英雄2代机 和 敌机
    object_play(hf2, ef)        # 英雄机 获胜!

