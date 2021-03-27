import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

from data.make_dataset import make_classification_datasets

N_JOBS = 5
N_NEIGHBORS = 3

class Trainer:

    def __init__(self):
        self.n_jobs = N_JOBS

    def load_data(self):
        train, test = make_classification_datasets()
        X_train, y_train = train[[col for col in train.columns if 'feature' in col]], train['target']
        X_test, y_test = test[[col for col in test.columns if 'feature' in col]], test['target']
        return X_train, y_train, X_test, y_test

    def train_model(self):
        self.model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
        X_train, y_train, X_test, y_test = self.load_data()
        self.model.fit(X_train, y_train)
        accuracy = self.model.score(X_test, y_test)
        f1 = f1_score(y_test, self.model.predict(X_test), average='macro')
        return f1
