import argparse # Модуль для обработки аргументов командной строки

from os import walk # Модуль для обхода файловой системы
import pandas as pd # Библиотека для работы с табличными данными
import csv # Модуль для работы с CSV-файлами

from model import NumberOcrModel # Импорт класса NumberOcrModel из модуля model

# Создание объекта для обработки аргументов командной строки
commandLineArgumentsParser = argparse.ArgumentParser()

# Добавление аргументов
commandLineArgumentsParser.add_argument('-i', '--input', help='Input image folder', type=str, default='test_images/') # Путь к папке с изображениями
commandLineArgumentsParser.add_argument('-o', '--output', help='Output csv-file', type=str, default='results.csv') # Путь к выходному CSV-файлу
commandLineArgumentsParser.add_argument('-d', '--detection-weights', help='Path to YOLO detection model weights', type=str, default='models/best(6).pt') # Путь к весам модели детекции YOLO
commandLineArgumentsParser.add_argument('-r', '--rect-weights', help='Path to rectification model weights', type=str, default='moran.pth') # Путь к весам модели для ректификации (MORAN)

# Парсинг аргументов
commandLineArguments = commandLineArgumentsParser.parse_args()


if __name__ == "__main__":
    # Список для хранения имен изображений
    images = []
    # Список для хранения результатов
    results = []

    # Создание экземпляра модели NumberOcrModel с переданными параметрами
    model = NumberOcrModel(
        detection_model=commandLineArguments.detection_weights, # Весы для YOLO
        rec_model='damo/cv_convnextTiny_ocr-recognition-general_damo', # Модель для распознавания текста
        angle_rec_model='MASTER', # Модель для распознавания углов текста
        moran_model=commandLineArguments.rect_weights # Весы для модели ректификации
    )

    # Обход всех файлов в указанной папке с изображениями
    for (dirpath, dirnames, filenames) in walk(commandLineArguments.input):
        images.extend(filenames) # Добавление всех найденных файлов в список images

    # Обработка каждого изображения
    for image in images:
        result = model.predict(commandLineArguments.input + image) # Предсказание с использованием модели
        if result:
            results.append(result) # Если результат не пустой, добавить его в список results

    # Запись результатов в CSV-файл
    with open(commandLineArguments.output, 'w', encoding='UTF8') as f:
        fields = ('filename', 'type', 'number', 'is_correct') # Поля CSV-файла
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n', delimiter=';') # Создание объекта для записи
        writer.writeheader() # Запись заголовков
        for result in results:
            writer.writerow(result[0]) # Запись строк с результатами
    # Чтение содержимого CSV-файла с помощью pandas
    file = pd.read_csv(commandLineArguments.output)