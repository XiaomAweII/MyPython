"""
## 聚类参数自定义测试
# 1.导入依赖包
# 2.数据读取及预处理
# 2.1 数据读取
# 2.2 特征选择
# 3.模型训练
# 3.1 K值的选择【2，20】
# 3.2 实例化模型,K=5
# 4.模型预测
# 4.1 获取预测结果
# 4.2 可视化
"""

# 1.导入依赖包
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# 2.数据读取及预处理
# 2.1 数据读取
data = pd.read_csv('./data/customers.csv')
print(data.head())
# 2.2 特征选择
x = data.iloc[:, [3, 4]]
print(x)

# # 3.模型训练
# # 3.1 K值的选择【2，20】
# sse_list =[]
# sc_list = []
# for i in range(2,20):
#     model =KMeans(n_clusters=i, n_init='auto')
#     model.fit(x)
#     sse = model.inertia_
#     sse_list.append(sse)
#     y_pred=model.predict(x)
#     sc_list.append(silhouette_score(x,y_pred))

# plt.figure()
# plt.grid()
# plt.plot(range(2,20),sse_list,'or-')
# plt.show()

# plt.figure()
# plt.grid()
# plt.plot(range(2,20),sc_list,'ob-')
# plt.show()

# 3.2 实例化模型,K=5
model = KMeans(n_clusters=5, n_init='auto')
model.fit(x)

# 4.模型预测
# 4.1 获取预测结果
y_pred = model.predict(x)
# print(y_pred)
# print(model.cluster_centers_)
# 4.2 可视化
plt.figure()
plt.scatter(x.values[y_pred == 0, 0], x.values[y_pred == 0, 1], c='r', label='1')
plt.scatter(x.values[y_pred == 1, 0], x.values[y_pred == 1, 1], c='b', label='2')
plt.scatter(x.values[y_pred == 2, 0], x.values[y_pred == 2, 1], c='y', label='3')
plt.scatter(x.values[y_pred == 3, 0], x.values[y_pred == 3, 1], c='g', label='4')
plt.scatter(x.values[y_pred == 4, 0], x.values[y_pred == 4, 1], c='gray', label='5')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='black', label='center')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
