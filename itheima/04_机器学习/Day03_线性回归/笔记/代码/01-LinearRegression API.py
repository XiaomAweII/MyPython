"""
## 线性回归案例1
# 1.导入依赖包
# 2.获取数据
# 3.模型训练
# 3.1 实例化模型
# 3.2 模型训练
# 4.模型预测
"""

# 1.导入依赖包
from sklearn.linear_model import LinearRegression


# 2.获取数据
x = [[160],[166],[172],[174],[180]]
y = [56.3,60.6,65.1,68.5,75]

# 3.模型训练
# 3.1 实例化模型
model =LinearRegression()
# 3.2 模型训练
model.fit(x,y)

# 权重(weight)/偏置(bias)
print(model.coef_)
print(model.intercept_)

# 4.模型预测
print(model.predict([[176]]))