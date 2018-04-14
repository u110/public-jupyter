import numpy as np
from numpy.random import seed


class AdalineSGD(object):
    """ ADAptive LIner NEurdon分類器

    parameters
    ----------

    :param eta: 学習率 0.0 - 1.0

    :typeof eata: float

    :param n_iter: トレーニングデータのトレーニング回数

    :typeof n_iter: int

    属性
    ----

    :w_: 1次元配列 適合後の重み

    :errors_: リスト 各エポックでの誤分類数

    :shuffle: bool デフォルト: True

    :randome_state: int デフォルト: None
    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle

        # randome_stateが指定された場合は乱数種を指定する
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        """ トレーニングデータに適合させる

        parameters
        ----------

        :param X: shape = [n_samples, n_features] トレーニングデータ

        :param y: shape = [n_samples] 目的変数

        :return: self: object
        """

        self._initialize_weights(X.shape[1])
        self.cont_ = []

        for i in range(self.n_iter):

            if self.shuffle:
                X, y = self._shuffle(X, y)

            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))

            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        """ 重みを再初期化することなくトレーニングデータに適合させる
        """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])

        # 目的変数yの要素が2以上の場合は
        # 各サンプルの特徴量xiと目的変数targetで重みを更新
        if y.raval().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)

        return self

    def _shuffle(self, X, y):
        """ トレーニングデータをシャッフル
        """

        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """ 重みを0に初期化
        """
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """ ADALINEの学習規則を用いて重みを更新 """

        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        """ 総入力を計算
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """ 線形活性化関数の出力を計算 """
        return self.net_intput(X)

    def predict(self, X):
        """ 1ステップ後のクラスラベルを返す
        """
        return np.where(self.activation(X) >= 0.0, 1, -1)
