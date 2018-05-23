import time
from contextlib import contextmanager

from logzero import logger

# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def load_data():
    data = pd.read_csv("../input/column_2C_weka.csv")
    data_2c_weka = data.drop("class", axis=1)
    return data_2c_weka


def main():
    with timer("load data"):
        data_2c_weka = load_data()

    with timer("data_2c_weka EDA"):
        print(data_2c_weka.head())

    with timer("PCA"):
        model = PCA()
        model.fit_transform(data_2c_weka)
        print("Principle components: ", model.components_)

    with timer("PCA variance"):
        scaler = StandardScaler()
        pca = PCA()
        pipline = make_pipeline(scaler, pca)
        pipline.fit(data_2c_weka)

        # plt.bar(range(pca.n_components_), pca.explained_variance_)
        # plt.xlabel('PCA feature')
        # plt.ylabel('variance')
        # plt.show()

        print("PCA n_components: ", pca.n_components_)
        print("PCA explained_variance: ", pca.explained_variance_)

    with timer("apply PCA"):
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(data_2c_weka)
        x = transformed[:, 0]
        y = transformed[:, 1]
        # color_list = ['red' if i=='Abnormal' else 'green' for i in data_2c_weka.loc[:,'class']]
        # plt.scatter(x, y, c=color_list)
        # plt.show()
        print("PCA x: ", x[:5])
        print("PCA y: ", y[:5])


if __name__ == "__main__":
    logger.info("start")
    main()
    logger.info("end")
