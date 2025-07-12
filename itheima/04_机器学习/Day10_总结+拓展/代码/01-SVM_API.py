"""
## SVM API
# 1.导入依赖包
# 2.加载数据及数据预处理
# 2.1 加载数据
# 2.2 数据降维
# 2.3 数据预处理，标准化
# 3.模型训练
# 4.模型预测
# 5.模型评估
# 5.1 准确率
# 5.2 可视化
"""

# 1.导入依赖包
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from plot_util import plot_decision_boundary

# 2.加载数据及数据预处理
# 2.1 加载数据
X, y = load_iris(return_X_y=True)

print(y.shape)
print(X.shape)
# 2.2 数据降维
x = X[y < 2, :2]
y = y[y < 2]
print(y.shape)

plt.scatter(x[y == 0, 0], x[y == 0, 1], c='red')
plt.scatter(x[y == 1, 0], x[y == 1, 1], c='blue')
plt.show()

# 2.3 数据预处理，标准化
transform = StandardScaler()
x_tran = transform.fit_transform(x)

# 3.模型训练
model = LinearSVC(C=10, dual='auto')
model.fit(x_tran, y)

# 4.模型预测
y_pred = model.predict(x_tran)

# 5.模型评估
# 5.1 准确率
print(accuracy_score(y_pred, y))
# 5.2 可视化
plot_decision_boundary(model, axis=[-3, 3, -3, 3])
plt.scatter(x_tran[y == 0, 0], x_tran[y == 0, 1], c='red')
plt.scatter(x_tran[y == 1, 0], x_tran[y == 1, 1], c='blue')
plt.show()
