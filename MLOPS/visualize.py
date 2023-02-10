import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import logging

directory = "plots"
if not os.path.exists(directory):
    os.makedirs(directory)


class Visualize:
    def __init__(self, X, X_train, X_test, y_train, y_test):
        self.X = X
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        h = 0.02  # meshgrid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        self.xx, self.yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        self.cm = plt.cm.PiYG
        self.cm_bright = ListedColormap(["#FF0000", "#00ff5e"])

    def vis_dataset(self):
        """

        :param xx:
        :param yy:
        :param X_train: Training features
        :param y_train: Training label
        :param X_test: Testing features
        :param y_test: testing label
        :return:
        """

        # just plot the dataset first
        plt.figure(figsize=(10, 8))
        self.cm = plt.cm.PiYG
        self.m_bright = ListedColormap(["#FF0000", "#00ff5e"])
        plt.title("Input data")

        # Plot the training points
        plt.scatter(
            self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=self.cm_bright, edgecolors="k"
        )
        # Plot the testing points
        plt.scatter(
            self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, marker='x', cmap=self.cm_bright, alpha=1
        )
        plt.xlim(self.xx.min(), self.xx.max())
        plt.ylim(self.yy.min(), self.yy.max())
        plt.xticks(())
        plt.yticks(())

        plt.savefig(os.path.join(directory, 'train_dataset_plot.png'))

    def vis_classifier(self, clf, score):
        """


        :param clf: model
        :param score: F1 score
        """
        self.Z = clf.predict_proba(np.c_[self.xx.ravel(), self.yy.ravel()])[:, 1]
        self.Z = self.Z.reshape(self.xx.shape)
        fig, ax = plt.subplots(figsize=(10, 10))

        ax.contourf(self.xx, self.yy, self.Z, cmap=self.cm, alpha=0.8)

        # Plot the training points
        ax.scatter(
            self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=self.cm_bright, edgecolors="k"
        )
        # Plot the testing points
        ax.scatter(
            self.X_test[:, 0],
            self.X_test[:, 1],
            c=self.y_test,
            cmap=self.cm_bright,
            edgecolors="k",
            alpha=0.6,
        )

        ax.set_xlim(self.xx.min(), self.xx.max())
        ax.set_ylim(self.yy.min(), self.yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        ax.set_title("Classifier output")
        ax.text(
            self.xx.max() - 0.3,
            self.yy.min() + 0.3,
            (f"score = {score:.2f}").lstrip("0"),
            size=15,
            horizontalalignment="right",
        )
        fig.savefig(os.path.join(directory, 'classifier_decision_training.png'))

    def vis_inference(self, X_inference, y_pred):
        """

        :param X_inference: Inference data
        :param y_pred: best model predictions on inference data
        :return:
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        ax.contourf(self.xx, self.yy, self.Z, cmap=self.cm, alpha=0.8)

        # Plot the inference points
        ax.scatter(
            X_inference[:, 0], X_inference[:, 1], marker="x", c=y_pred, cmap=self.cm_bright
        )

        ax.set_xlim(self.xx.min(), self.xx.max())
        ax.set_ylim(self.yy.min(), self.yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

        ax.set_title("Classifier inference output")
        fig.savefig(os.path.join(directory, 'classifier_decision_inference.png'))
        logging.info(f"Performance plot of best model is stored in plots/classifier_decision_inference.png")
