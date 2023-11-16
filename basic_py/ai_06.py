# Дельта-правило

# Сеть училась распозновать 5ть на картинке
# 1. В сеть подали 5 и она её узнала, значит дельта ошибки (=) равна 0 и соответствено ничего делать не нужно.
# 2. В сеть подали 5, но она её не узнала и выдала друое значение, в этом случае мы прибавляем вес, который мы определили в скорости обучения = коэффициент
# 3. В сеть подали, например 7, но она сказала что это 5ть, в этом случае мы вычитаем вес, который мы определили в скорости обучения = коэффициент. 
# 4. В сеть например, подали 7 и сеть прявильно определила, что это не 5ть, значит дельта ошибки (=) равна 0 и соответствено ничего делать не нужно.

import random
# Коэффициент при Х
k = random.uniform(-5, 5)

# Свободный член урованения прямой - c
c = random.uniform(-5, 5)
print('Начальная прямая линия: ', k, '* x + ', c)

rate = 0.0001 # скорость обучения = значение на которое меняется вес связи в соответствии с дельта-правилом


# Набор точек X:Y
data = {22: 150, 23: 155, 24: 160, 25: 162, 26: 171, 27: 174, 28: 180, 29: 183, 30: 189, 31: 192}


# Расчёт Y
def proceed(x):
    return x * k + c

# Тренеровка сети
for i in range (100000):
    # Получить случайную X-координату точки
    x = random.choice(list(data.keys()))
    # Получить соответствующую Y-координату точки
    true_result = data [x]
    # Получить ответ сети
    out = proceed(x)
    # Считаем ошибку сети
    delta = true_result - out
    # Меняем вес при x в соответствии с дельта-правилом
    k += delta * rate * x
    # Меняем вес при постоянном входе в соответствии с дельта-правилом
    c += delta * rate

# Вывод данных начальной прямой линии

print('Готовая прямая: Y = ', k, '* x + ', c)






