import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import io
import os

# Загрузка данных из Интернета, запись их в объект DataFrame и вывод на печать
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
response = requests.get(url, verify=False)  # Запрос на получение данных по указанному URL
data = response.content.decode('utf-8')  # Декодирование полученных данных в формате utf-8
df = pd.read_csv(io.StringIO(data), header=None)  # Создание объекта DataFrame для хранения данных
print('Данные об ирисах')
print(df.to_string())  # Вывод данных об ирисах на печать

# Создание папки, если она не существует
output_dir = 'C:\\CSV'
os.makedirs(output_dir, exist_ok=True)  # Создание папки CSV или проверка, если она уже существует

# Сохранение файла CSV
output_file = os.path.join(output_dir, 'Iris.csv')  # Формирование пути к файлу CSV
df.to_csv(output_file)  # Запись данных в файл CSV

# Подготовка данных для обучения модели
X = df.iloc[0:100, [0, 2]].values  # Выбор первых 100 строк и столбцов 0 и 2 для признаков X (длина чашелистика и пестика)
Y = df.iloc[0:100, 4].values  # Выбор первых 100 строк и столбца 4 для целевой переменной Y (виды ирисов)
Y = np.where(Y == 'Iris-setosa', -1, 1)  # Преобразование меток классов в числовые значения (-1 для Iris-setosa и 1 для остальных видов ирисов)

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

    def fit(self, X, Y):
        """
        Осуществляет подгонку параметров модели перцептрона к тренировочным данным.

        Параметры:
        X : numpy.ndarray
            Матрица признаков тренировочных данных.
        Y : numpy.ndarray
            Вектор целевой переменной тренировочных данных.

        Возвращает:
        self : Perceptron
            Объект класса Perceptron.
        """
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
        """
        Рассчитывает взвешенную сумму входных данных.

        Параметры:
        X : numpy.ndarray
            Матрица входных данных.

        Возвращает:
        numpy.ndarray
            Взвешенная сумма входных данных.
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Прогнозирует класс на основе входных данных.

        Параметры:
        X : numpy.ndarray
            Матрица входных данных.

        Возвращает:
        numpy.ndarray
            Прогнозируемый класс (1 или -1).
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

perceptron = Perceptron(eta=0.1, n_iter=10)
perceptron.fit(X, Y)

# Визуализация первых 50 элементов обучающей выборки (строки 0-49, столбцы 0 и 1)
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='щетинистый')
# Визуализация следующих 50 элементов обучающей выборки (строки 50-99, столбцы 0 и 1)
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

il = [2.5, 1.6]
i2 = [6.4, 5.5]
Rl = ppn.predict(il)
R2 = ppn.predict(i2)
print('Rl=', Rl, ' R2=', R2)

if Rl == 1:
    print('Rl = Вид Iris versicolor')
else:
    print('Rl = Вид Iris setosa')

# Визуализация разделительной границы
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """
    Функция для визуализации разделительной границы классификации.

    Параметры:
    X : numpy.ndarray
        Матрица признаков.
    y : numpy.ndarray
        Вектор целевых переменных.
    classifier : object
        Обученная модель классификатора.
    resolution : float, optional (default=0.02)
        Разрешение сетки для построения разделительной границы.

    Возвращает:
    None
    """
    # Настройка маркеров и палитры
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue')
    cmap = ListedColormap(colors)

    # Вывод поверхности решения
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Отображение образцов классов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                    color=cmap(idx), marker=markers[idx], label=cl)

plot_decision_regions(X, Y, classifier=ppn)
plt.xlabel('Длина чашелистика, см')
plt.ylabel('Длина лепестка, см')
plt.legend(loc='upper left')
plt.show()

