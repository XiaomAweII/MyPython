"""
# 1.导入工具包
# 2.准备数据
# 3.模型训练
# 4.模型预测
# 4-1 模型预测
# 4-2 展示效果
"""

# '''
# 欠拟合
# '''
# 1.导入工具包
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 2.准备数据
np.random.seed(22)
x = np.random.uniform(-3, 3, size=100)
print(x)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
print(y)

# 3.模型训练
model = LinearRegression()
X = x.reshape(-1, 1)
model.fit(X, y)

# 4.模型预测
# 4-1 模型预测
y_predict = model.predict(X)
print(mean_squared_error(y, y_predict))

# 4-2 展示效果
plt.scatter(x, y)
plt.plot(x, y_predict)
plt.show()

# ------------------------------------------------------------------------------ #

# '''
# 正好拟合
# '''
# 1.导入工具包
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 2.准备数据
np.random.seed(22)
x = np.random.uniform(-3, 3, size=100)
# print(x)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
# print(y)

# 3.模型训练
model = LinearRegression()
X = x.reshape(-1, 1)
X2 = np.hstack([X, X ** 2])
model.fit(X2, y)

# 4.模型预测
# 4-1 模型预测
y_predict = model.predict(X2)
print(mean_squared_error(y_true=y, y_pred=y_predict))
# 4-2 展示效果
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)])
plt.show()

# ------------------------------------------------------------------------------ #

'''
过拟合
'''
# 1.导入工具包
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 2.准备数据
np.random.seed(22)
x = np.random.uniform(-3, 3, size=100)
# print(x)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
# print(y)

# 3.模型训练
model = LinearRegression()
X = x.reshape(-1, 1)
X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 9, X ** 10])
model.fit(X3, y)

# 4.模型预测
# 4-1 模型预测
y_predict = model.predict(X3)
print(mean_squared_error(y_true=y, y_pred=y_predict))
# 4-2 展示效果
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)])
plt.show()

# ------------------------------------------------------------------------------ #

'''
L1正则化
'''
# 1.导入工具包
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 2.准备数据
np.random.seed(22)
x = np.random.uniform(-3, 3, size=100)
# print(x)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
# print(y)

# 3.模型训练
model = Lasso(alpha=0.1)
X = x.reshape(-1, 1)
X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 9, X ** 10])
model.fit(X3, y)
print(model.coef_)

# 4.模型预测
# 4-1 模型预测
y_predict = model.predict(X3)
print(mean_squared_error(y_true=y, y_pred=y_predict))
# 4-2 展示效果
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)])
plt.show()

# ------------------------------------------------------------------------------ #

'''
L2正则化
'''
# 1.导入工具包
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 2.准备数据
np.random.seed(22)
x = np.random.uniform(-3, 3, size=100)
# print(x)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
# print(y)

# 3.模型训练
model = Ridge(alpha=0.1)
X = x.reshape(-1, 1)
X3 = np.hstack([X, X ** 2, X ** 3, X ** 4, X ** 5, X ** 6, X ** 7, X ** 9, X ** 10])
model.fit(X3, y)
print(model.coef_)

# 4.模型预测
# 4-1 模型预测
y_predict = model.predict(X3)
print(mean_squared_error(y_true=y, y_pred=y_predict))
# 4-2 展示效果
plt.scatter(x, y)
plt.plot(np.sort(x), y_predict[np.argsort(x)])
plt.show()
