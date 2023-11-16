import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import io
import os

# Загрузка из Интернета данных, запись их в объект DataFrame и вывод на печать
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
response = requests.get(url, verify=False)
data = response.content.decode('utf-8')
df = pd.read_csv(io.StringIO(data), header=None)
print('Данные об ирисах')
print(df.to_string())

# Создание папки, если она не существует
output_dir = 'C:\\CSV'
os.makedirs(output_dir, exist_ok=True)

# Сохранение файла CSV
output_file = os.path.join(output_dir, 'Iris.csv')
df.to_csv(output_file)

# ... остальной код ...

X = df.iloc[0:100, [0, 2]].values
Y = df.iloc[0:100, 4].values
Y = np.where(Y == 'Iris-setosa', -1, 1)

class Perceptron(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

    def fit(self, X, Y):
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, Y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

perceptron = Perceptron(eta=0.1, n_iter=10)
perceptron.fit(X, Y)

# Первые 50 элементов обучающей выборки (строки 0-50, столбцы 0, 1)
plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='щетинистый')
# Следующие 50 элементов обучающей выборки (строки 50-100, столбцы 0, 1)
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='разноцветный')

plt.xlabel('Длина чашелистика')
plt.ylabel('Длина лепестка')
plt.legend(loc='upper left')
plt.show()

# Тренировка
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, Y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Число случаев ошибочной классификации')
plt.show()

il=[5.5, 1.6]
i2=[6.4, 4.5]
Rl = ppn.predict(il)
R2 = ppn.predict(i2)
print('Rl=', Rl,' R2=', R2)

if Rl == 1:
    print('Rl = Вид Iris versicolor')
else:
    print('Rl = Вид Iris setosa')

# Визуализация разделительной границы
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # Настроить генератор маркеров и палитру
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Вывести поверхность решения
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Показать образцы классов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                    c=cmap(idx), marker=markers[idx], label=cl)

plot_decision_regions(X, Y, classifier=ppn)
plt.xlabel('Длина чашелистика, см')
plt.ylabel('Длина лепестка, см')
plt.legend(loc='upper left')
plt.show()









