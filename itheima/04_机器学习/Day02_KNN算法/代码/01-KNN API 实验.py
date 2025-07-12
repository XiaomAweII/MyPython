"""
## KNN API实验
# 1.工具包
# 2.数据(特征工程)
# 2-1分类
# 2-2回归
# 3.实例化模型
# 4.模型训练
# 5.模型预测

x = [[0, 2, 3], [1, 3, 4], [3, 5, 6], [4, 7, 8], [2, 3, 4]]
y = [0, 0, 1, 1, 0]
"""

# 1.工具包
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# 2.数据(特征工程)
# 分类
x = [[0, 2, 3], [1, 3, 4], [3, 5, 6], [4, 7, 8], [2, 3, 4]]
y = [0, 0, 1, 1, 0]
# 回归
# x = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
# y = [0.1, 0.2, 0.3, 0.4]

# 3.实例化模型
model = KNeighborsClassifier(n_neighbors=3)
# model = KNeighborsRegressor(n_neighbors=3)

# 4.模型训练
model.fit(x, y)

# 5.模型预测
print(model.predict([[4, 4, 5]]))
