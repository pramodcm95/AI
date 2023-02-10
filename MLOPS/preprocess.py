import numpy as np
from sklearn.cluster import DBSCAN
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')


class PreProcess:
    def __init__(self):
        self.data_pipeline = None

    def load_data(self, train=True, inference=False):
        """

        :return: dataframe, features, labels
        """
        if train:
            df = pd.read_csv(
                "data/historical_sensor_data.csv",
                sep=','
            )
            logging.info("Training Data loaded successfully")
        elif inference:
            df = pd.read_csv(
                "data/latest_sensor_data.csv",
                sep=','
            )
            logging.info("Inference Data loaded successfully")
        else:
            raise Exception("Please input either train or inference")

        # Features
        X = df[['sensor_1', 'sensor_2']].values
        try:
            # Labels
            y = df[['label']].values
            y = y.reshape(-1)
        except:
            # Labels
            y = None

        return df, X, y

    def noise_detector(self, X):
        """

        :param X: features
        """
        # run DBSCAN on the dataset
        dbscan = DBSCAN(eps=0.3, min_samples=10)
        labels = dbscan.fit_predict(X)

        # count the number of noise points
        noise_points = np.sum(labels == -1)

        # calculate the noise percentage
        noise_percentage = noise_points / X.shape[0] * 100
        logging.info(f"Noise in data is about {noise_percentage}% as detected by DBSCAN")

    def _detect_Zscore(self, data):
        """

        :param data: a column data
        :return: index of outliers in given data
        """

        threshold = 3
        mean = np.mean(data)
        std = np.std(data)

        z_scores = [(y - mean) / std for y in data]
        return np.where(np.abs(z_scores) > threshold)

    # detect outliers
    def detect_outliers(self, df):
        """

        :param df:  Training dataframe
        """
        for col in df[['sensor_1', 'sensor_2']].columns:
            outlier_count = len(self._detect_Zscore(df[col])) - 1
            logging.info(f"Outliers count in column {col} is {outlier_count} ")

    def data_imbalance_check(self, df):
        """

        :param df: Training dataframe
        """
        col_1_share = df["label"].value_counts()[0] * 100 / df["label"].value_counts().sum()
        col_2_share = df["label"].value_counts()[1] * 100 / df["label"].value_counts().sum()
        logging.info(
            f"share of class 1 and class 2 in training data are {col_1_share}% and {col_2_share}% respectively")
        logging.info(f"No imbalance of data detected")

    def preprocess_data(self, X, train=True, inference=False):
        """

        :param train: True if data needs for training
        :param inference: True if data needs for testing
        :param X: Features to preprocess
        :return: features after preprocessing
        """
        """Preprocess the data by standardizing it."""
        if train:
            self.data_pipeline = Pipeline([
                ('std_scaler', StandardScaler())
            ])
            X_scaled = self.data_pipeline.fit_transform(X)
        elif inference:
            if self.data_pipeline is None:
                raise Exception("Please perform training first to infer")
            X_scaled = self.data_pipeline.transform(X)
        """
        Generally a data pipeline should be used here like:

        num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='interpolate')),
        ('add_variables', NewVariablesAdder()),
        ('std_scaler', StandardScaler())
        ])

        data_pipeline = ColumnTransformer([
            ('numerical', num_pipeline, num_vars),
            ('categorical', OneHotEncoder(), cat_vars),
        ])

        Since we do not have any missing data, categorical variables and only numerical data, the above scaling is enough
        """

        return X_scaled

    def execute(self):
        df, features, labels = self.load_data()

        # pre processing verification
        self.noise_detector(features)

        # detect outlier
        self.detect_outliers(df)

        # Data balance check
        self.data_imbalance_check(df)

        # data pre-process
        features_scaled = self.preprocess_data(features, train=True)

        return df, features, labels, features_scaled
