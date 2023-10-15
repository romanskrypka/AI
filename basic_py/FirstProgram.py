
from math import pi
print(pi)




def MyPrint(d, r):
    print ('В функции мы выполнили печать ', d, 'переменньх x и y')
    print ('Результат Z=', r)

#print ( "Привет Python") 


x = 2
y = 3
z = x + y
#print(x, y)
#print(z)
#MyPrint ('сложения', z)
z = y - x
#print(z)
#MyPrint ('вычитания', z)
z = x * y
#print(z)
#MyPrint ('умножения', z)
z = x / y
#print(z)
#MyPrint ('деления', z)
z = x ** y # возведение в степень
#print(z)
#MyPrint ('возведения в степень', z)

def f_sum(a, b):
    result = a + b
    return result

s = f_sum(2, 3)
#print(s)

array = [10, 11, 12, 13, 14, "Это текст"]
#print(array[0])
#print(len(array))

x = 12
"""
if (x < 10):
    x = x/2
else:
    x = x * 2
print(x)
"""
xwhile = 1
"""
while xwhile <= 100: #Цикл while повторяет необходимые команды до тех пор, пока не остается истинным некоторое условие.  # noqa: E501
    print ("Квадрат числа " + str(xwhile) + " равен " + str(xwhile**2))
    xwhile = xwhile + 1

    
xfor = 1
for xfor in range(1, 100): # Мы используем ключевое слово for для создания цикла. 
    #Далее мы указываем, что хотим повторить определенные действия для всех xfor в диапазоне от 1 до 100.
    # Функция range (1, 101) создает массив из 100 чисел, начиная с 1 и заканчивая 100.
    print("Квадрат числа " + str(xfor) + " равен " + str(xfor**2))
"""  # noqa: E501
"""
for i in [1, 10, 100, 1000]: # Имеем конкретный массив 
    print(i * 2)
"""

# Класс кошки 
class Cat:
    Name_Class = "Кошки"

    # Действия, которые надо выполнять при создании оббъекта "Кошка"
    def __init__(self, wool_color, eyes_color, name):
        self.wool_color = wool_color
        self.eyes_color = eyes_color
        self.name = name

    # Мурлыкать 
    def purr (self): 
        print("Myppp!") 
    # Шипеть 
    def hiss (self): 
        print("Шшшш 1 ") 
    # Царапаться 
    def scrabble(self): 
        print("Цап-царап!")

my_cat = Cat('Цвет шерсти', 'Цвет глаз', 'Кличка')
"""
my_cat = Cat('Белая', 'Зелёные', 'Мурка')
print("Наименование класса -", my_cat.Name_Class)
print("Boт наша кошка:")
print("Цвeт шерсти -", my_cat.wool_color)
print("Цвeт глаз -", my_cat.eyes_color)
print("Кличка -", my_cat.name)
"""

"""
my_cat = Cat('Белая', 'Зелёные', 'Мурка')
my_cat.name = "Васька"
my_cat.wool_color = "Черный"
print("Наименование класса -", my_cat.Name_Class)
print("Boт наша кошка:")
print("Цвeт шерсти -", my_cat.wool_color)
print("Цвeт глаз -", my_cat.eyes_color)
print("Кличка -", my_cat.name)
my_cat .purr ()
"""

class Car (object):
    # Наименование класса
    Narne_class = "Автомобили"

    def __init__(self, brand,  weight, power):
        self.brand = brand # Марка автомобиля
        self.weight = weight # Вес автомобиля
        self.power = power # Мощность двигателя
    
    # Метод двигаться прямо
    def drive(self):
        # Здесь команды двигаться прямо
        print("Поехали, двигаться прямо!")
    
    # Метод повернуть на право
    def righ(self):
        # Здесь команды повернуть руль направо
        print("Едем, поварачиваем руль на право!")
    
    # Метод повернуть на лево
    def left(self):
        # Здесь команды повернуть руль налево
        print("Едем, поварачиваем руль налево!")
    
    # Метод тормозить
    def brake(self):
        # Здесь команды нажатия на педать тормоза
        print("Стоп, активируем тормоз!")

    # Метод подать звуковой сигнал
    def beep(self):
        # Здесь команды подачи звукового сигнала
        print("Подан звуковой сигнал")

"""
my_car = Car("Мерседес", 1200, 250)
print('Параметры автомобиля, созданного из класса - ', my_car.Narne_class)
print ('Марка (модель) - ', my_car.brand)
print ('Вес (кг) - ', my_car.weight)
print('Мощность двигателя (лс) - ', my_car.power)
my_car.drive() # Двигается прямо
my_car.righ() # Поворачиваем направо
my_car.drive() # Двигается прямо
my_car.left() # Поворачиваем налево
my_car.drive() # Двигается прямо
my_car.beep() # Подаем звуковой сигнал
my_car.brake() # Тормози


from math import factorial  # noqa: E402

result = factorial(10)
print(result)
"""


