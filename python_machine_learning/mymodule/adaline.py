# utf-8
import numpy as np


class AdalineGD(object):
    """ADAptive LInear NEuron 分類器

    params:
    -------
    eta: float
        学習率 0.0 - 1.0
    n_iter: int
        トレーニングデータのトレーニング回数

    properties:
    -------------

    w_: 1次元配列
        適合後の重み
    errors_: list
        各エポックでの誤分類数
    """

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ トレーニングデータに適合させる

        parameters:
        -----------
        X: shape = [n_samples, n_features]

        returns:
        --------
        self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            # トレーニング回数分データを反復
            output = self.net_input(X)
            errors = (y - output)

            # update w_1, w_2, ...
            self.w_[1:] += self.eta * X.T.dot(errors)
            # update w_0
            self.w_[0] += self.eta * errors.sum()
            # calculate cost function
            cost = (errors**2).sum() / 2.0
            # save cost
            self.cost_.append(cost)

        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """線形活性化関数の出力を計算"""
        return self.net_input(X)

    def predict(self, X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
