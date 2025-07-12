"""
## KNN 手写数字识别
# 1.导入依赖包
# 2.读取数据并展示数据
# 2.1 读取数据
# 2.2 展示数据
# 3.数据预处理
# 3.1 归一化
# 3.2 数据集划分
# 4.模型训练
# 4.1 实例化模型
# 4.2 模型训练
# 5.模型预测
# 6.模型评估
# 7.模型保存
# 8.模型加载
"""

# 1.导入依赖包
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# 2.读取数据并展示数据
# 2.1 读取数据
data = pd.read_csv('手写数字识别.csv')
x = data.iloc[:, 1:]
y = data.iloc[:, 0]
# print(Counter(y))

# 2.2 展示数据
# digit = x.iloc[1000].values
# img = digit.reshape(28,28)
# plt.imshow(img,cmap='gray')
# plt.imsave('demo.png',img)
# plt.show()

# 3.数据预处理
# 3.1 归一化
x = x / 255.
# 3.2 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=22)

# 4.模型训练
# 4.1 实例化模型
model = KNeighborsClassifier(n_neighbors=11)
# 4.2 模型训练
model.fit(x_train, y_train)

# 5.模型预测
img = plt.imread('demo.png')
img = img.reshape(1, -1)
y_predict = model.predict(x_test)
print(y_predict)

# 6.模型评估
print(model.score(x_test, y_test))
print(accuracy_score(y_predict, y_test))

# # 7.模型保存
# joblib.dump(model, 'knn.pth')

# 8.模型加载
knn = joblib.load('knn.pth')
# print(knn.score(x_test,y_test))

img = plt.imread('demo.png')
plt.imshow(img, cmap='gray')
plt.show()
img = img.reshape(1, -1)
print(knn.predict(img))

