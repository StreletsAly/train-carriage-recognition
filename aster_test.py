import argparse

from os import walk # Импорт функции для обхода директорий
import pandas as pd # Импорт библиотеки для работы с таблицами
import csv # Импорт модуля для работы с CSV-файлами

from model2 import NumberOcrModel # Импорт класса модели OCR из файла model2

# Создание парсера аргументов командной строки
commandLineArgumentsParser = argparse.ArgumentParser()

# Добавление аргументов командной строки
commandLineArgumentsParser.add_argument('-i', '--input', help='Input image folder', type=str, default='test_images/')
commandLineArgumentsParser.add_argument('-o', '--output', help='Output csv-file', type=str, default='results.csv')
commandLineArgumentsParser.add_argument('-d', '--detection-weights', help='Path to YOLO detection model weights', type=str, default='models/best(6).pt')

# Разбор аргументов
commandLineArguments = commandLineArgumentsParser.parse_args()


if __name__ == "__main__":
    images = [] # Список для хранения имён файлов изображений
    results = [] # Список для хранения результатов обработки

    # Инициализация модели OCR
    model = NumberOcrModel(
        detection_model=commandLineArguments.detection_weights, # Путь к модели детекции
        rec_model='damo/cv_convnextTiny_ocr-recognition-general_damo', # Путь к модели распознавания текста
        angle_rec_model='SVTR-base', # Путь к модели распознавания углов поворота текста
    )

    # Рекурсивное чтение всех файлов в указанной папке
    for (dirpath, dirnames, filenames) in walk(commandLineArguments.input):
        images.extend(filenames) # Добавление имён файлов в список

    # Обработка каждого изображения
    for image in images:
        result = model.predict(commandLineArguments.input + image) # Предсказание с использованием OCR модели
        if result:
            results.append(result) # Сохранение результата, если он не пустой

    # Запись результатов в CSV-файл
    with open(commandLineArguments.output, 'w', encoding='UTF8') as f:
        fields = ('filename', 'type', 'number', 'is_correct') # Заголовки столбцов
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n', delimiter=';')
        writer.writeheader() # Запись заголовков
        for result in results:
            writer.writerow(result[0]) # Запись строк в CSV

    # Чтение и вывод CSV-файла с результатами
    file = pd.read_csv(commandLineArguments.output)
    print(file) # Вывод содержимого файла