"""
## Kmeans算法
# 1.导入依赖包
# 2.构建数据集
# 3.模型训练预测
# 4.模型评估
"""

# 1.导入依赖包
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

# 2.构建数据集
x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [4, 4], [8, 8], [2, 2, ]],
                  cluster_std=[0.4, 0.2, 0.3, 0.2])
# plt.figure()
# plt.scatter(x[:,0],x[:,1])
# plt.show()


# 3.模型训练预测
model = KMeans(n_clusters=4, n_init=10)
y_prd = model.fit_predict(x)

plt.figure()
plt.scatter(x[:, 0], x[:, 1], c=y_prd)
plt.show()

# 4.模型评估
print(silhouette_score(x, y_prd))
