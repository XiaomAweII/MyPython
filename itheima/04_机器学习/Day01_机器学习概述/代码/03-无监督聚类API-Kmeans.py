"""
## 无监督聚类API-Kmeans
# 1 导包 sklearn.cluster.KMeans sklearn.datasets.make_blobs#
# 2 创建数据集
# 2-1 展示数据效果
# 3 实例化Kmeans模型并预测
# 3-1 展示聚类效果
# 4 评估3种聚类效果好坏
"""

# 1 导入依赖包
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
# make系列-自己构造数据集 fetch系列大数据-从网络加载 load系列小数据集-从本地数据集
from sklearn.metrics import calinski_harabasz_score  # calinski_harabaz_score 废弃


def dm04_kmeans():
    # 2 创建数据集 1000个样本,每个样本2个特征 4个质心蔟数据标准差[0.4, 0.2, 0.2, 0.2]
    x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.2], random_state=22)
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], marker='o')
    plt.show()

    # 3 使用k-means进行聚类, 并使用CH方法评估
    # 3-1 n_clusters=2
    y_pred = KMeans(n_clusters=2, random_state=22, init='k-means++', n_init='auto').fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.show()
    print('1-->', calinski_harabasz_score(x, y_pred))

    # 3-2 n_clusters=3
    y_pred = KMeans(n_clusters=3, random_state=22, n_init='auto').fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.show()
    print('2-->', calinski_harabasz_score(x, y_pred))

    # 3-3 n_clusters=4
    y_pred = KMeans(n_clusters=4, random_state=22, n_init='auto').fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.show()

    # 4 模型评估
    print('3-->', calinski_harabasz_score(x, y_pred))


if __name__ == '__main__':
    dm04_kmeans()
