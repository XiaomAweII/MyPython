every_tall = []                        # 创建列表记录历次落体运动的移动长度
height = []                            # 创建列表记录历次反弹的高度

start_height = 100.0  # 初始高度
num_jump = 10         # 弹跳次数

for i in range(1, num_jump + 1):
    if i == 1:
        # 第一次仅下落
        every_tall.append(start_height)
    else:
        # 从第二次开始，从反弹到落地的距离应该是反弹高度的两倍
        every_tall.append(2 * start_height)

    start_height /= 2
    height.append(start_height)

total_distance = sum(every_tall)
final_bounce_height = height[-1]

print('球移动的所有长度%f'%total_distance)
print('最终反弹高度%f'%final_bounce_height)