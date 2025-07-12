"""
## Boston房价预测
# 1.导入依赖包
# 2.数据预处理
# 2.1 获取数据
# 2.2 数据集划分
# 2.3 标准化
# 3.模型训练
# 3.1 实例化模型(正规方程)
# 3.2 模型训练 fit
# 4.模型预测
# 5.模型评估
"""

# 1.导入依赖包
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


# 2.数据预处理
# 2.1 获取数据
# data = load_boston()
# print(data)
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 2.2 数据集划分
# x_train,x_test,y_train,y_test =train_test_split(boston.data,boston.target,test_size=0.2,random_state=22)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=22)

# 2.3 标准化
process = StandardScaler()
x_train = process.fit_transform(x_train)
x_test = process.transform(x_test)

# 3.模型训练
# 3.1 实例化模型(正规方程)
# model =LinearRegression(fit_intercept=True)
model = SGDRegressor(learning_rate='constant', eta0=0.01)
# 3.2 模型训练 fit
model.fit(x_train, y_train)

# print(model.coef_)
# print(model.intercept_)

# 4.模型预测
y_predict = model.predict(x_test)
print(y_predict)

# 5.模型评估
print(mean_squared_error(y_test, y_predict))
