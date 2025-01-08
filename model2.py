import os
import re
from cgi import maxlen
from os import walk

import cv2
import torch
from ultralytics import YOLO
from mmocr.apis import MMOCRInferencer
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torchvision.transforms.functional as F

from vaild_function import is_valid

# Константы для путей сохранения результатов
DETECTION_SAVE_PATH = './yolo_detections/results/crops/number/'
MODEL_RESULT_PATH = './model_result/results.csv'

# Типы бинаризации для обработки изображений
BIN_TYPES = {
    'ADAPTIVE_THRESH_GAUSSIAN_C': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    'ADAPTIVE_THRESH_MEAN_C': cv2.ADAPTIVE_THRESH_MEAN_C,
    'THRESH_OTSU': cv2.THRESH_OTSU
}


class NumberOcrModel:
    def __init__(self, detection_model, rec_model, angle_rec_model):
        '''
        Инициализация модели.

        :param detection_model: путь к модели детекции (YOLO)
        :param rec_model: модель распознавания текста (например, ModelScope)
        :param angle_rec_model: модель для исправления угла текста (например, MMOCR)
        '''
        self.detection_model = YOLO(detection_model) # Инициализация модели YOLO
        self.rec_model = pipeline(Tasks.ocr_recognition, model=rec_model) # Модель распознавания текста
        self.angle_rec_model = MMOCRInferencer(rec=angle_rec_model) # Модель исправления угла текста
        self.tfms = self.angle_rec_model.textrec_inferencer.pipeline # Трансформации для предобработки изображений

        # Переменные для хранения промежуточных результатов
        self.detection_result = None
        self.img_path = None
        self.image_name = None
        self.bin_type = None
        self.rec_result = None

        self.prepare_model() # Подготовка модели к использованию

    def prepare_model(self):
        '''
        Перенос модели YOLO на GPU, если доступно.
        '''
        if torch.cuda.is_available():
            self.detection_model.to('cuda')

    def rectificate(self, crop_path):
        '''
        Исправление искажения текста на изображении

        :param crop_path: путь к вырезанному изображению
        '''
        d = {k: [v] for k, v in self.tfms(crop_path).items()} # Применение трансформаций
        pr = self.angle_rec_model.textrec_inferencer.model.data_preprocessor(d) # Предобработка данных
        rectCrop = self.angle_rec_model.textrec_inferencer.model.preprocessor(pr['inputs']) # Исправление угла
        F.to_pil_image(rectCrop[0].data.cpu().mul_(0.5).add_(0.5)).save(crop_path) # Сохранение исправленного изображения

    def preprocess(self, imagePath, imageName, binPrep):
        '''
        Предобработка изображения: детекция объектов и бинаризация.

        :param imagePath: путь к исходному изображению
        :param imageName: имя изображения
        :param binPrep: тип бинаризации
        :return: результаты детекции и путь к вырезанному изображению
        '''
        # Выполнение детекции объектов на изображении
        detectionResult = self.detection_model.predict(imagePath, save=True, save_crop=True,
                                                        project='yolo_detections', name='results', verbose=False)

        # Поиск папки с результатами детекции
        allDirectories = os.listdir('./yolo_detections')
        maxLength = len(max(allDirectories, key=len))
        dataDirectory = sorted([x for x in allDirectories if len(x) == maxLength])[-1]
        directoryPath = f'./yolo_detections/{dataDirectory}/crops/number/'

        # Сбор всех вырезанных изображений
        crops = []
        for (dirpath, dirnames, filenames) in walk(directoryPath):
            crops.extend(filenames)

        # Выбор изображения с максимальной шириной (если несколько объектов)
        if len(crops) > 1:
            images = [cv2.imread(directoryPath + img) for img in crops]
            cropsWidth = [img.shape[1] for img in images]
            cropImageName = crops[cropsWidth.index(max(cropsWidth))]
        else:
            cropImageName = imageName

        # Применение бинаризации, если указано
        if detectionResult and binPrep:
            image = cv2.imread(directoryPath + cropImageName)
            blurImage = cv2.GaussianBlur(image, (1, 1), 0) # Размытие изображения
            binImage = cv2.adaptiveThreshold(blurImage, 255, BIN_TYPES[binPrep], cv2.THRESH_BINARY_INV, 29, -4) # Бинаризация
            cv2.imwrite(directoryPath + cropImageName, binImage) # Сохранение бинаризованного изображения


        return detectionResult, directoryPath + cropImageName

    def recognize(self, image_name, crop_image_path, detected_data):
        '''
        Распознавание текста на изображении.

        :param image_name: имя изображения
        :param crop_image_path: путь к вырезанному изображению
        :param detected_data: результаты детекции
        :return: список с результатами распознавания
        '''
        # Если детекция не обнаружила объектов, возвращаем пустой результат
        if not detected_data[0] or len(detected_data[0].cpu().numpy()) == 0:
            return [
                {
                    'filename': image_name,
                    'type': 0,
                    'number': 0,
                    'is_correct': 0,
                }
            ]
        self.rectificate(crop_image_path) # Исправление угла текста

        # Распознавание текста двумя разными моделями
        result_angle_rec = self.angle_rec_model(crop_image_path)
        result_rec = self.rec_model(crop_image_path)

        # Извлечение чисел из результатов распознавания
        num_angle_rec = re.sub(r'[^0-9]', '', result_angle_rec['predictions'][0]['rec_texts'][0])  # Aster
        num_rec = re.sub(r'[^0-9]', '', result_rec['text'][0])  # ModelScope

        # Логика выбора наиболее подходящего результата
        if not num_rec:
            num_sub = num_angle_rec
        elif num_angle_rec:
            if len(num_rec) >= 8:
                num_sub = num_rec[:8]
            if len(num_rec) < 8 and len(num_angle_rec) >= len(num_rec):
                num_sub = num_angle_rec[:8]
            if 8 > len(num_rec) > len(num_angle_rec):
                num_sub = num_rec
        else:
            num_sub = num_rec

        # Формирование результата
        result = [{
            'filename': image_name,
            'type': int(num_sub != None),
            'number': (0, num_sub)[num_sub != None and num_sub != ''],
            'is_correct': is_valid(num_sub),
        }]

        return result

    def predict(self, imagePath, bin_prep=None):
        '''
        Основной метод для выполнения предсказания.

        :param imagePath: путь к изображению
        :param bin_prep: тип бинаризации (если требуется)
        :return: результаты распознавания текста
        '''
        imageName = os.path.basename(os.path.normpath(imagePath)) # Получение имени файла из пути
        detectedData, cropImagePath = self.preprocess(imagePath, imageName, bin_prep) # Предобработка изображения
        return self.recognize(imageName, cropImagePath, detectedData) # Распознавание текста
