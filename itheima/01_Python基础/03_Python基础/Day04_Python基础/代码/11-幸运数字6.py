"""
需求:
    输入任意的数字, 然后生成 1 ~ 该数字之间额列表, 从中选取幸运数字(能被6整除的)移动到新列表 lucky, 并打印两个列表.
"""

# 1. 定义列表 nums, 用于记录: 生成的数据.
nums = []
# 2. 定义列表 lucky, 用于记录: 幸运数字.
lucky = []

# 3. 提示用户键盘录入1个值, 并接收. 细节: 转成int类型.
input_num = int(input('请录入1个大于0的整数: '))     # 例如: 9

# 4. 生成 1 ~ 用户录入的数字 区间内的所有整数, 然后添加到 nums 列表中.
for i in range(1, input_num + 1):
    nums.append(i)

# 5. 遍历nums列表, 获取到每个值.
for num in nums:
    # 6. 判断当前的值是否是 6的倍数, 如果是, 就将其添加到 lucky列表中.
    if num % 6 == 0:
        lucky.append(num)

# 7. 打印 nums 和 lucky两个列表的信息.
print(f'nums: {nums}')
print(f'lucky: {lucky}')

print('-' * 28)


# 合并版, 列表推导式.
# 1. 提示用户键盘录入1个值, 并接收. 细节: 转成int类型.
input_num = int(input('请录入1个大于0的整数: '))     # 例如: 9

# 2. 生成 1 ~ 用户录入的数字之间的 数字列表.
nums = [i for i in range(1, input_num + 1)]

# 3. 从上述的 nums列表中, 找到 幸运数字.
lucky = [i for i in nums if i % 6 == 0]


# 7. 打印 nums 和 lucky两个列表的信息.
print(f'nums: {nums}')
print(f'lucky: {lucky}')