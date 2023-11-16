
# Модуль sigmoid
import numpy as np


# Функция активации: f(x) = 1/(1+e^(-x))
def sigmoid(x):
    b = 5
    if x >= b:
        return 1/(1+np.exp(-x))
    else:
        return 1/(1+np.exp(-x))

# Создание класса "Нейрон" 
class Neuron: 
    def __init__ (self, w): 
        self.w = w
    def y (self, x): # Сумматор
        s = np.dot(self.w, x) # Суммируем входы
        return sigmoid(s) # функция активации

#Xi = np.array( [1, 1, 0, 0]) # Задание значений входам
Xi = np.array( [0, 0, 0, 0]) # Задание значений входам
Wi = np.array( [5, 4, 3, 1]) # Веса входных сенсоров
n = Neuron (Wi) # Создание объекта из класса Neuron
#print ('Sl= ', n.y(Xi)) # Обращение к нейрону
#Xi = np.array([5, 6]) # Веса входных сенсоров
#print ('S2= ', n.y(Xi)) # Обращение к нейрону
print('Y= ', n.y(Xi))




