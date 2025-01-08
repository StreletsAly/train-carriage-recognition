import os
import re
from cgi import maxlen
from os import walk

import cv2 # Библиотека для обработки изображений
import torch # Фреймворк для машинного обучения
import torchvision.transforms.functional as F # Функции для преобразования изображений
from torch.autograd import Variable # Для создания автоградируемых переменных
from ultralytics import YOLO # Для работы с моделью YOLO
from mmocr.apis import MMOCRInferencer # API для работы с MMOCR
from modelscope.pipelines import pipeline # Для работы с pipeline ModelScope
from modelscope.utils.constant import Tasks # Константы задач ModelScope
import torchvision.transforms.functional as F
from torchvision.transforms import ToPILImage # Преобразование изображений в PIL
from collections import OrderedDict # Упорядоченный словарь
from PIL import Image # Работа с изображениями в формате PIL

from vaild_function import is_valid # Импорт функции для проверки валидности номера
from MORAN_v2.models.moran import MORAN # Импорт модели MORAN
import MORAN_v2.tools.utils as utils # Инструменты для работы с MORAN
import MORAN_v2.tools.dataset as dataset # Преобразования изображений для MORAN


'''
Конфигурация моделей:
    detection_mode: './models/custom_yolov8x.pt'  # YOLO для детекции
    rec_model: damo/cv_convnextTiny_ocr-recognition-general_damo  # ModelScope для распознавания текста
    angle_rec_model: Aster  # MMOCR для распознавания углов текста
'''

DETECTION_SAVE_PATH = './yolo_detections/results/crops/number/' # Путь сохранения вырезанных объектов
MODEL_RESULT_PATH = './model_result/results.csv' # Путь сохранения результатов работы модели

# Типы бинаризации для предобработки изображений
BIN_TYPES = {
    'ADAPTIVE_THRESH_GAUSSIAN_C': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    'ADAPTIVE_THRESH_MEAN_C': cv2.ADAPTIVE_THRESH_MEAN_C,
    'THRESH_OTSU': cv2.THRESH_OTSU
}


class NumberOcrModel:
    """
        Основной класс для распознавания номеров. Содержит функции для:
        - Инициализации моделей детекции и распознавания.
        - Предобработки изображений.
        - Распознавания текста.
        - Ректификации изображений с использованием MORAN.
        """
    def __init__(self, detection_model, rec_model, angle_rec_model, moran_model):
        # Инициализация модели детекции YOLO
        self.detection_model = YOLO(detection_model)
        # Инициализация модели распознавания текста из ModelScope
        self.rec_model = pipeline(Tasks.ocr_recognition, model=rec_model)
        # Инициализация модели распознавания текста из ModelScope
        self.angle_rec_model = MMOCRInferencer(rec=angle_rec_model)

        # Алфавит для MORAN (включает цифры, буквы и специальный символ "$")
        alphabetForMORAN = '0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$'
        self.cuda_flag = False # Флаг использования CUDA

        # Инициализация MORAN модели
        if torch.cuda.is_available(): # Если доступна CUDA, использовать её
            self.cuda_flag = True
            self.rectificator = MORAN(1, len(alphabetForMORAN.split(':')), 256, 32, 100, BidirDecoder=True, CUDA=self.cuda_flag)
            self.rectificator = self.rectificator.cuda()
            state_dict = torch.load(moran_model)
        else:  # Использование CPU
            self.rectificator = MORAN(1, len(alphabetForMORAN.split(':')), 256, 32, 100, BidirDecoder=True, inputDataType='torch.FloatTensor', CUDA=self.cuda_flag)
            state_dict = torch.load(moran_model, map_location='cpu')

        # Переименование слоев MORAN модели для совместимости
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "") # Удаление префикса "module."
            MORAN_state_dict_rename[name] = v

        self.rectificator.load_state_dict(MORAN_state_dict_rename) # Загрузка весов модели
        for p in self.rectificator.parameters():
            p.requires_grad = False # Отключение градиентов
        self.rectificator.eval() # Перевод модели в режим инференса

        # Инициализация конвертера и трансформера для MORAN
        self.converter = utils.strLabelConverterForAttention(alphabetForMORAN, ':')
        self.transformer = dataset.resizeNormalize((100, 32))

        self.detection_result = None
        self.img_path = None
        self.image_name = None
        self.bin_type = None
        self.rec_result = None

        self.prepare_model() # Подготовка модели для работы

    def prepare_model(self):
        # Перенос модели детекции на GPU, если доступно
        if torch.cuda.is_available():
            self.detection_model.to('cuda')

    def rectificate(self, crop_path):
        """
                Ректификация изображения с помощью MORAN.
                """
        imagesInGrayscaleMode = Image.open(crop_path).convert('L') # Открытие изображения в режиме градаций серого
        imagesInGrayscaleMode = self.transformer(imagesInGrayscaleMode) # Изменение размера изображения

        if self.cuda_flag:
            imagesInGrayscaleMode = imagesInGrayscaleMode.cuda() # Перенос на GPU
        imagesInGrayscaleMode = imagesInGrayscaleMode.view(1, *imagesInGrayscaleMode.size())
        imagesInGrayscaleMode = Variable(imagesInGrayscaleMode)

        # Ректификация изображения
        rectifiedImage = self.rectificator.rectify(imagesInGrayscaleMode, test=True)
        # Сохранение выровненного изображения
        F.to_pil_image(rectifiedImage[0].data.cpu().mul_(0.5).add_(0.5)).save(crop_path)


    def preprocess(self, image_path, image_name, bin_prep):
        """
        Предобработка изображения: детекция объекта и бинаризация.
        """
        # Детекция объектов с помощью YOLO
        detectionResult = self.detection_model.predict(image_path, save=True, save_crop=True,
                                                        project='yolo_detections', name='results', verbose=False)
        # Поиск директории с результатами
        allDirectories = os.listdir('./yolo_detections')
        maxLength = len(max(allDirectories, key=len))
        dataDirectory = sorted([x for x in allDirectories if len(x) == maxLength])[-1]
        directoryPath = f'./yolo_detections/{dataDirectory}/crops/number/'

        # Выбор самого большого по ширине кадра
        croppedImages = []
        for (dirpath, dirnames, filenames) in walk(directoryPath):
            croppedImages.extend(filenames)

        if len(croppedImages) > 1:
            images = [cv2.imread(directoryPath + img) for img in croppedImages]
            cropsWidth = [img.shape[1] for img in images]
            crop_image_name = croppedImages[cropsWidth.index(max(cropsWidth))]
        else:
            crop_image_name = image_name

        # Бинаризация изображения
        if detectionResult and bin_prep:
            image = cv2.imread(directoryPath + crop_image_name)
            blurImage = cv2.GaussianBlur(image, (1, 1), 0) # Размытие
            binImage = cv2.adaptiveThreshold(blurImage, 255, BIN_TYPES[bin_prep], cv2.THRESH_BINARY_INV, 29, -4)
            cv2.imwrite(directoryPath + crop_image_name, binImage)

        return detectionResult, directoryPath + crop_image_name

    def recognize(self, imagePath, cropImagePath, detectedData):
        """
        Распознавание текста на изображении.
        """
        # Если ничего не обнаружено, вернуть пустой результат
        if not detectedData[0] or len(detectedData[0].cpu().numpy()) == 0:
            return [
                {
                    'filename': imagePath,
                    'type': 0,
                    'number': 0,
                    'is_correct': 0,
                }
            ]
        self.rectificate(cropImagePath) # Ректификация изображения

        # Распознавание текста с помощью двух моделей
        result_MMOCR = self.angle_rec_model(cropImagePath) # MMOCR
        result_ModelScope = self.rec_model(cropImagePath) # ModelScope

        # Обработка результата
        num_MMOCR = re.sub(r'[^0-9]', '', result_MMOCR['predictions'][0]['rec_texts'][0])  # aster
        num_ModelScope = re.sub(r'[^0-9]', '', result_ModelScope['text'][0])  # model scope

        # Логика выбора результата
        if not num_ModelScope:
            num_sub = num_MMOCR
        elif num_MMOCR:
            if len(num_ModelScope) >= 8:
                num_sub = num_ModelScope[:8]
            if len(num_ModelScope) < 8 and len(num_MMOCR) >= len(num_ModelScope):
                num_sub = num_MMOCR[:8]
            if 8 > len(num_ModelScope) > len(num_MMOCR):
                num_sub = num_ModelScope
        else:
            num_sub = num_ModelScope
        result = [{
            'filename': imagePath,
            'type': int(num_sub != None),
            'number': int(num_sub) if num_sub != None and num_sub != '' else 0,
            'is_correct': is_valid(num_sub), # Проверка валидности номера
        }]

        return result

    def predict(self, imagePath, binPrep=None):
        """
        Основная функция для выполнения детекции, предобработки и распознавания.
        """
        imageName = os.path.basename(os.path.normpath(imagePath))  # Извлечение имени файла
        detectedData, croppedImagePath = self.preprocess(imagePath, imageName, binPrep)
        return self.recognize(imagePath, croppedImagePath, detectedData)
