import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
from sklearn import datasets
from sklearn.svm import SVC


class SVM:
    def __init__(self):
        self.INDENT = 0.01
        self.iris = datasets.load_iris()
        self.clf = SVC(kernel='linear')
        self.sepal = self.iris.data[:, :2]
        self.petal = self.iris.data[:, 2:]

    def create_grid(self, X):
        x_min = X[:, 0].min() - 1
        x_max = X[:, 0].max() + 1
        y_min = X[:, 1].min() - 1
        y_max = X[:, 1].max() + 1
        x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, self.INDENT),
                                     np.arange(y_min, y_max, self.INDENT))
        return x_grid, y_grid

    def split_irises(self, X):
        x_grid, y_grid = self.create_grid(X)
        Z = self.clf.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
        Z = Z.reshape(x_grid.shape)
        plt.contourf(x_grid, y_grid, Z, alpha=0.3)

    def classify_irises(self, name):
        if name == 'sepal':
            X = self.iris.data[:, :2]
            plt.xlabel('Sepal length')
            plt.ylabel('Sepal width')
        elif name == 'petal':
            X = self.iris.data[:, 2:]
            plt.xlabel('Petal length')
            plt.ylabel('Petal width')
        else:
            print('Not found name')
            return

        self.clf.fit(X, self.iris.target)
        self.split_irises(X)
        plt.scatter(X[:, 0], X[:, 1], c=self.iris.target)

        handles = [
            patches.Patch(color='purple', label='setosa'),
            patches.Patch(color='green', label='versicolor'),
            patches.Patch(color='yellow', label='viriginica'),
        ]
        plt.legend(handles=handles)

    def set_new_iris(self, x, y):
        new_iris = self.clf.predict([[x, y]])
        plt.scatter(x, y, c='red')
        return self.iris.target_names[new_iris][0]


if __name__ == '__main__':
    o = SVM()
    o.classify_irises('sepal')
    x = 5.1
    y = 4.9
    print(f'({x}, {y}) -> {o.set_new_iris(x, y)}')
    plt.show()
