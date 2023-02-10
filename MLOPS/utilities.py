import time
import pandas as pd
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
import joblib
import os


class BestClassifier:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        # Define the classifiers to be compared
        self.classifiers = [RandomForestClassifier(), SVC(), KNeighborsClassifier(), GaussianNB(), GaussianProcessClassifier(1.0 * RBF(1.0))]
        self.classifier_names = ['Random Forest', 'SVM', 'KNN', 'GNB', "GP"]
        rf_params = {'n_estimators': [10, 100, 1000], 'max_depth': [None, 5, 10, 20]}
        svm_params = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
        knn_params = {'n_neighbors': [1, 5, 10, 50]}
        gaussian_params = {'var_smoothing': [1e-9]}
        gp_params = {}
        self.param_grids = [rf_params, svm_params, knn_params, gaussian_params, gp_params]

    def _train_classifier(self, clf, X_train, y_train, X_test, y_test):
        """

        :param clf: current model
        :param X_train: Training features
        :param y_train: Training label
        :param X_test: Testing features
        :param y_test: testing label
        :return:
        """
        start = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # end = time.time()
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        end = time.time()
        return accuracy, f1, end - start

    def _grid_search_classifier(self, clf, X_train, y_train, params):
        """

        :param clf: current model
        :param X_train: Training features
        :param y_train: Training label
        :param params:
        :return:
        """
        grid_search = GridSearchCV(clf, params, cv=5)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_

    def execute(self):
        """

        :return: results of hyperparameter search and model selection sorted according to best evaluation metrics
        """
        logging.info(f'Training multiple models to look for best classification model for given data')
        # Train and evaluate each classifier
        results = []
        for clf, name, params in zip(self.classifiers, self.classifier_names, self.param_grids):
            best_clf = self._grid_search_classifier(clf, self.X_train, self.y_train, params)
            accuracy, f1, exec_time = self._train_classifier(best_clf, self.X_train, self.y_train, self.X_test,
                                                            self.y_test)
            results.append((name, accuracy, f1, exec_time, best_clf))

        # Select the classifier with the best f1 score
        best_classifier = sorted(results, key=lambda x: x[2], reverse=True)[0]
        results = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1 Score', 'Time Taken', 'Estimator'])
        # sort based on highest F1 score and least time taken
        results = results.sort_values(by=['F1 Score', 'Time Taken'], ascending=[False, True])
        results.to_csv("results.csv")
        logging.info(f'Best classifier: {results.iloc[0][0]}')
        logging.info(f'Accuracy: {results.iloc[0][1]}')
        return results


def visualize(df):
    """

    :param df: data to visualize
    """
    # folder to save plots
    directory = "plots"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Plot a heatmap to visualize the missing values
    sns.heatmap(df.isnull(), cbar=False)
    plt.savefig(os.path.join(directory, 'mssing_data_heat_map_training.png'))

    # Plot histograms for all columns
    df.hist(bins=20, figsize=(20, 15))
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'hist_training.png'))

    # Plot a boxplot for each column
    df.boxplot(figsize=(20, 5))
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'box_plot_training.png'))

    # Plot a scatter plot matrix
    sns.pairplot(df)
    plt.savefig(os.path.join(directory, 'pair_plot_training.png'))

    # Plot a heatmap of the correlation matrix
    sns.heatmap(df.corr(), annot=True)
    plt.savefig(os.path.join(directory, 'correlation_training.png'))


def save_model(model, name):
    joblib.dump(model, name)


def load_model(name):
    model = joblib.load(name)
    return model
