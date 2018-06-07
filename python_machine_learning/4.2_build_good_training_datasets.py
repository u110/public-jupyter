
# coding: utf-8

# # 4章 データ前処理 -よりよいデータセットの構築-
# 
# 2018/05/18
# 
# - データセットにおける欠測値の削除と補完
# - 機械学習アルゴリズムに合わせたカテゴリデータの整形
# - モデルの構築に適した特徴量の選択
# 
# ## 4.1 欠測データへの対処
# 
# 
# ### 4.1.1 欠測値を含む要素を取り除く
# 
# - 一般的な計算ツールは欠測値へのできない
# - 無視すると予期せぬ結果を生み出す
# - 分析者が適切に対応することが重要
# 
# - pandas.DataFrame.dropna が便利。
#     - デメリット: 削除しすぎると解析の信頼性が失われる可能性もある。
#         - --> 補完（次項）の方法
#         
# ### 4.1.2 補完
# 
# - 平均補完
#     - サンプルの平均値で補完
#     - sklearnのImputerにはstrategyにmedian, やmost_frequentなどもある
# 
# 
# ## 4.2 カテゴリデータの対処
# 
# - 順序尺度のデータを自動的に整数値に置き換えることは困難
#     - 自前でマッピングを定義して生成する必要がある。
# 
# - 順序の性質を持たないクラスラベルには0から値を設定するなど
# - ↑と同様もしくは、sklean LabelEncoder
# 
# - one-hot エンコーディング
#     - 整数値にしてしまうと、順序の意味を持ってしまう。
#         - 学習アルゴリズムによっては順序の意味を解釈してしまうかもしれない。
#         - 問題を回避する方法としてone-hotエンコーディングという手法がある
# 
# ## 4.3 データセットをトレーニングデータとテストデータと分割する
# 
# - テストデータの割合が多い --> アルゴリズムへの価値を間引いている
# - テストデータの割合が少ない --> 汎化誤差の推定の精度が失われる
# 
# - 1) トレーニングデータのみでハイパーパラメタのチューニング
# - 2) テストデータで予測精度を検証
# - 3) 予測精度に満足がいけたらなら、テストとトレーニングをあわせた全体のデータでモデルを推定する。
# 
# ## 4.4 特徴量の尺度を揃える
# 
# - 正規化、標準化は分野によっては意味があいまいに用いられる場合があるので、推測する必要がある。
# 
# ここでは
# 
# - 正規化(normalization)
#     - [0, 1]の範囲内でスケーリングしなおすこと
# 
# $$
# x^{(i)}_{norm} = \frac{x^{(i)} - x_{min}}{x_{max} - x_{min}}
# $$
# 
# - 標準化(standardization)
#     - 平均0、標準偏差1となるように変換する
#     - 一般的な線形モデルの重みを0、もしくは0に近い乱数で初期化するため、正規化よりも標準化のほうが機械学習では実用的らしい。
#     - $\mu_x$ は特徴量列の平均値、 $ \sigma_x $ は対応する標準偏差
#     
# $$
# x^{(i)}_{std} = \frac{x^{(i)} - \mu_x}{\sigma_x}
# $$
# 

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')


# In[ ]:

import pandas as pd

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'], 
    ['red', 'L', 13.5, 'class2'],
    ['blue', 'XL', 15.3, 'class1']
])

# 列名を設定
df.columns = ['color', 'size', 'price' , 'classlabel']
df


# In[ ]:

# 順序尺度のデータを整数値に自動的に置き換えることはできないので、自分で定義する必要がある

# Tシャツのサイズと整数値をマッピング

size_mapping = {
    'XL': 3,
    'L': 2,
    'M': 1
}

_df = df.copy()
_df['size'] = _df['size'].map(size_mapping)
_df


# In[ ]:

# マッピングを元に

inv_size_mapping = { v:k for k,v in size_mapping.items()}
__df = _df.copy()
__df.size = __df['size'].map(inv_size_mapping)
__df


# In[ ]:

import numpy as np


# クラスラベルを整数値に対応させるディクショナリを生成

class_mapping = { val:idx for idx, val in enumerate(np.unique(df.classlabel))}
class_mapping

# マッピング

_df = df.copy()
_df["classlabel"] = _df["classlabel"].map(class_mapping)
_df


# In[ ]:

#　マッピングを元に戻すには 反転した ディクショナリを使えば良い

inv_class_mapping = {v:k for k,v in class_mapping.items()}
inv_class_mapping

__df = _df.copy()
__df["classlabel"]  = __df["classlabel"].map(inv_class_mapping)
__df


# In[ ]:

# sklean LabelEncoder


# In[ ]:

from sklearn.preprocessing import LabelEncoder


# ラベルエンコーダーのインスタンス生成
class_le = LabelEncoder()

# 整数に変換
y = class_le.fit_transform(df.classlabel)
y


# In[ ]:

# 文字列に戻す

class_le.inverse_transform(y)


# In[ ]:

_df = df.copy()
_df["size"] = _df["size"].map(size_mapping)
_df


# In[ ]:

X = _df[["color", "size", "price"]].values
print(X)

print("====")
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])

print(X)


# In[ ]:

# one-hot エンコーディング
from sklearn.preprocessing import OneHotEncoder

# インスタンス生成
ohe = OneHotEncoder(categorical_features=[0])

ohe.fit_transform(X).toarray()


# In[ ]:

# インスタンス生成
ohe = OneHotEncoder(categorical_features=[0], sparse=False)  # toarrayを省略
ohe.fit_transform(X)


# In[ ]:

#pandas の場合はget_dummiesで

pd.get_dummies(
    df[["color", "size", "price"]]
)


# In[ ]:

"""
===============================================
4．3　データセットをトレーニングデータセットとテストデータセットに分割する
===============================================
""";


# In[ ]:

# wineデータセットを読み込む
df_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)


# In[ ]:

df_wine.columns = [
    "Class label",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline"
]
print("Class labels", np.unique(df_wine["Class label"]))


# In[ ]:

# wineデータセットの先頭5行を表示
df_wine.head()


# In[ ]:

from sklearn.cross_validation import train_test_split


# 特徴量とクラスラベルを別々に抽出
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[ ]:

X.shape


# In[ ]:

X_train.shape


# In[ ]:

X_test.shape


# In[ ]:

"""
=================
4.4 特徴量の尺度を揃える
=================
""";


# In[ ]:

from sklearn.preprocessing import MinMaxScaler


# min-maxスケーリングのインスタンスを生成
mms = MinMaxScaler()

# スケーリング
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)


# In[ ]:

from sklearn.preprocessing import StandardScaler


# 標準化のインスタンスを生成
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


# In[ ]:

sns.pairplot(pd.DataFrame(X_train).iloc[:, :5])


# In[ ]:

sns.pairplot(pd.DataFrame(X_train_std).iloc[:, :5])


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



