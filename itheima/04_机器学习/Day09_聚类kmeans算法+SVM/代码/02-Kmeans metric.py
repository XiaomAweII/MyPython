"""
## Kmeans评估
# 1.导入依赖包
# 2.构建数据集
# 3.迭代不同的k值,获取sse
# 4.绘制图像
"""

# 1.导入依赖包
import os
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

# 2.构建数据集
x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [4, 4], [8, 8], [2, 2, ]],
                  cluster_std=[0.4, 0.2, 0.3, 0.2])

# 3.迭代不同的k值,获取sse
temp_list = []
# for k in range(1,100):
#     model =KMeans(n_clusters=k,n_init='auto')
#     model.fit(x)
#     temp_list.append(model.inertia_)

# for k in range(2,100):
#     model =KMeans(n_clusters=k, n_init='auto')
#     model.fit(x)
#     y_pred=model.predict(x)
#     temp_list.append(silhouette_score(x,y_pred))

for k in range(2, 100):
    model = KMeans(n_clusters=k, n_init='auto')
    model.fit(x)
    y_pred = model.predict(x)
    temp_list.append(calinski_harabasz_score(x, y_pred))

# 4.绘制图像
plt.figure()
plt.grid()
plt.plot(range(2, 100), temp_list, 'or-')
plt.show()
