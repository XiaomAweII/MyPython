import jieba.posseg as pseg
import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain
from wordcloud import WordCloud
# 获取形容词列表
def get_a_list(text):
    r = [] # 存储形容词
    for g in pseg.lcut(text):
        if g.flag == "a":
            r.append(g.word)
    return r

# 产生词云
def get_word_cloud(keywords_list):
    word_cloud = WordCloud(font_path='./cn_data/SimHei.ttf', max_words=100, background_color='white')
    # 将列表转换成字符串
    keywords_str = ' '.join(keywords_list)
    # 产生词云
    word_cloud.generate(keywords_str)
    # 词云展示
    plt.figure()
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()

# 定义主逻辑函数
def dm_wordcloud():
    # 读取数据
    train_data = pd.read_csv('./cn_data/train.tsv', sep='\t')
    print(f'train_data-->{train_data.head()}')
    # 获取正样本数据
    p_train_data = train_data[train_data["label"] == 1]["sentence"]
    print(f'p_train_data--》{p_train_data}')
    # 获取正样本中形容词
    p_train_a = list(chain(*map(lambda x: get_a_list(x), p_train_data)))
    print(f'p_train_a--》{p_train_a}')
    # 获取正样本形容词词云展示
    get_word_cloud(p_train_a)

    # 获取负样本数据
    n_train_data = train_data[train_data["label"] == 0]["sentence"]
    # 获取负样本中形容词
    n_train_a = list(chain(*map(lambda x: get_a_list(x), n_train_data)))
    # 获取负样本形容词词云展示
    get_word_cloud(n_train_a)
if __name__ == '__main__':
    dm_wordcloud()