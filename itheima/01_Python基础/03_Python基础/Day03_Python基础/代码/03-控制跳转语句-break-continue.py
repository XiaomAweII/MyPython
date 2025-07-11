"""
控制跳转语句详解:
    概述:
        控制跳转语句指的是 break 和 continue这两个关键字, 他们可以控制循环的执行.
    作用:
        continue: 结束本次循环, 进行下次循环的.
        break: 终止循环, 即: 循环不再继续执行了.
"""


# 需求: 观察如下的代码, 再其中填充指定内容, 使其能够完成 打印2次, 7次, 13次 'Hello World'

for i in range(1, 11):      # i的值: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    if i % 3 == 0:
        # 这里加内容
        # break         # 看到break, 循环结束, 即: 打印的有 1, 2
        # continue      # 看到continue, 就结束本次循环, 立即开启下次循环, 即: 不打印的值有: 3, 6, 9
        print(f'hello world! {i}')

    print(f'hello world! {i}')

print('至此, for循环就执行结束了!')