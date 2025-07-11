"""
if多分支介绍:
    概述:
        它是分支结构if语句的一种写法, 适用于 多情况 的判断.
    格式:
        if 条件1:
            语句体1
        elif 条件2:
            语句体2
        elif 条件3:
            语句体3
        ......
        else:
            语句体n
    执行流程:
        1. 先执行条件1, 看其是否成立, 成立则执行语句体1, 然后整个if语句结束.
        2. 如果条件1不成立, 则执行条件2, 看其是否成立, 成立则执行语句体2, 然后整个if语句结束.
        3. 如果条件2不成立, 则执行条件3, 看其是否成立, 成立则执行语句体3, 然后整个if语句结束.
        4. 重复上述的步骤, 如果所有的判断条件都不成立, 则执行 最后1个else的 语句体n, 至此, if语句结束.

    细节:
        1. 根据需求 elif 可以写多个.
        2. 最后的else语句可以省略不写, 但是它代表着所有条件都不成立的情况, 所以建议加上.
        3. 以后测试数据的时候, 建议测试三种值:
            和法值.
            边界值.
            非法值.
"""

# 需求: 根据考试成立, 发放奖励.
"""
       成绩               奖励
    90 ~ 100        奖励: 游乐场套票一张
    80 ~ 89         奖励: 游乐场半天游
    70 ~ 79         奖励: 博物馆全天游
    60 ~ 69         奖励: 全套练习题
    0 ~ 59          奖励: 口头激励
"""

# 1. 提示用户录入他/她的考试成绩, 并接收, 记得转成: 整数.
score = eval(input('请录入您的考试成绩: '))

# 2. 根据考试成立发放奖励.
# 写法1: 普通版.
if score >= 90 and score <= 100:
    print('奖励: 游乐场套票一张')
elif score >= 80 and score < 90:
    print('奖励: 游乐场半天游')
elif score >= 70 and score < 80:
    print('奖励: 博物馆全天游')
elif score >= 60 and score < 70:
    print('奖励: 全套练习题')
else:
    print('奖励: 口头激励')

print('-' * 28)

# 写法2: Python独有版.
if 90 <= score <= 100:
    print('奖励: 游乐场套票一张')
elif 80 <= score < 90:
    print('奖励: 游乐场半天游')
elif 70 <= score < 80:
    print('奖励: 博物馆全天游')
elif 60 <= score < 70:
    print('奖励: 全套练习题')
elif 00 <= score < 60:
    print('奖励: 口头激励')
else:
    print('成绩无效, 男女混打 + 扫帚棒法!')

print('-' * 28)


# 写法3: 实际开发版, 能看懂即可.
if score < 0 or score > 100:                # 非法值校验
    print('成绩无效, 男女混打 + 扫帚棒法!')
elif score >= 90:                       # else的默认条件: score >= 0 and score <= 100
    print('奖励: 游乐场套票一张')
elif score >= 80:                       # else的默认条件: score >= 0 and score < 90
    print('奖励: 游乐场半天游')
elif score >= 70:                       # else的默认条件: score >= 0 and score < 80
    print('奖励: 博物馆全天游')
elif score >= 60:                       # else的默认条件: score >= 0 and score < 70
    print('奖励: 全套练习题')
else:                                   # else的默认条件: score >= 0 and score < 60
    print('奖励: 口头激励')

