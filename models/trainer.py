import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

from data.make_dataset import make_classification_datasets

import mlflow
import mlflow.sklearn

N_JOBS = 5
N_NEIGHBORS = 3
TEST_SIZE = 0.2
N_FEATURES = 10
N_INFORMATIVE = 5

class Trainer:

    def __init__(self, n_neighbors=N_NEIGHBORS):
        self.n_neighbors = n_neighbors
        self.n_jobs = N_JOBS

    def load_data(self):
        train, test = make_classification_datasets(
            test_size=TEST_SIZE,
            n_features=N_FEATURES,
            n_informative=N_INFORMATIVE
        )
        X_train, y_train = train[[col for col in train.columns if 'feature' in col]], train['target']
        X_test, y_test = test[[col for col in test.columns if 'feature' in col]], test['target']
        return X_train, y_train, X_test, y_test

    def train_model(self):
        with mlflow.start_run() as run:
            self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
            X_train, y_train, X_test, y_test = self.load_data()
            self.model.fit(X_train, y_train)
            accuracy = self.model.score(X_test, y_test)
            f1 = f1_score(y_test, self.model.predict(X_test), average='macro')

            # MLFLOW LOGS
            mlflow.log_param('n_neighbors', self.n_neighbors)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("n_features", N_FEATURES)
            mlflow.log_param('n_informative', N_INFORMATIVE)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1", f1)

            #mlflow.set_tag("exp_id", experiment_id)
            mlflow.set_tag("exp_name", 'first')

            mlflow.sklearn.log_model(self.model, "model")
            #
            # settings ={
            #     'name': 'sets',
            #     'model': 'KNeighbours'
            # }
            #mlflow.log_artifact(settings)

        return f1
