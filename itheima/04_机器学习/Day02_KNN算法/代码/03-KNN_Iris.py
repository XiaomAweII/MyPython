"""
## KNN_鸢尾花
# 1.导入工具包
# 2.加载数据集
# 2.1 加载数据集
# 2.2 展示数据集
# 3.特征工程(预处理-标准化)
# 3.1 数据集划分
# 3.2 标准化
# 4.模型训练
# 4.1 实例化
# 4.2 调用fit法
# 5.模型预测
# 6.模型评估(准确率)
# 6.1 直接计算准确率
# 6.2 使用预测结果

"""

# 1.导入工具包
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 2.加载数据集并展示数据集
# 2.1 加载数据集
iris_data = load_iris()
# print(iris_data)
# print(iris_data.target)

# 2.2 展示数据集
iris_df = pd.DataFrame(iris_data['data'], columns=iris_data.feature_names)
iris_df['label'] = iris_data.target
# print(iris_df)
# print(iris_data.feature_names)
# sns.lmplot(x='sepal length (cm)',y='sepal width (cm)',data = iris_df,hue='label')
# plt.show()


# 3.特征工程(预处理-标准化)
# 3.1 数据集划分
x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.3, random_state=22)
print(len(iris_data.data))
print(len(x_train))
# 3.2 标准化
process = StandardScaler()
x_train = process.fit_transform(x_train)
x_test = process.transform(x_test)

# 4.模型训练
# 4.1 实例化
model = KNeighborsClassifier(n_neighbors=3)
# 4.2 调用fit法
model.fit(x_train, y_train)

# 5.模型预测
x = [[5.1, 3.5, 1.4, 0.2]]
x = process.transform(x)
print(model.predict_proba(x))

# 6.模型评估(准确率)
# 6.1 直接计算准确率
acc = model.score(x_test, y_test)
print(acc)

# 6.2 使用预测结果
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print(acc)
