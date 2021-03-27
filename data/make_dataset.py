import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split



def make_classification_datasets(test_size=0.2, n_features=10, n_informative=5):
    X, Y = make_classification(n_samples=300,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_classes=4,
                               n_clusters_per_class=1,
                               random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    train = pd.concat([
        pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(10)]),
        pd.DataFrame(y_train, columns=['target'])
                     ], axis=1)
    test = pd.concat([
        pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(10)]),
        pd.DataFrame(y_test, columns=['target'])
                     ], axis=1)
    train.to_csv('data/train.csv')
    test.to_csv('data/test.csv')

    return train, test


if __name__ == "__main__":
    make_classification()
