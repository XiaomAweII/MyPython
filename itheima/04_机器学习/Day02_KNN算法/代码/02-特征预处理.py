"""
## 特征预处理
# 1.导入工具包
# 2.数据
# 3.实例化(归一化,标准化)
# 3-1 归一化
# 3-2 标准化
# 4.特征处理，fit_transform


x = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]
"""

# 1.导入工具包
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 2.数据(只有特征)
x = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 3.实例化(归一化,标准化)
# process =MinMaxScaler()
process = StandardScaler()

# 4.特征处理，fit_transform
data = process.fit_transform(x)
# print(data)

print(process.mean_)
print(process.var_)
