import torch.nn as nn
from MORAN_v2.models.morn import MORN
from MORAN_v2.models.asrn_res import ASRN

# Основная модель MORAN, которая включает в себя два подмодуля: MORN и ASRN
class MORAN(nn.Module):

    def __init__(self, nc, nclass, nh, targetH, targetW, BidirDecoder=False, 
    	inputDataType='torch.cuda.FloatTensor', maxBatch=256, CUDA=True):
        # Инициализация класса MORAN
        super(MORAN, self).__init__()
        # Инициализация подмодуля MORN для предварительной обработки изображений
        self.MORN = MORN(nc, targetH, targetW, inputDataType, maxBatch, CUDA)
        # Инициализация подмодуля ASRN для распознавания текста
        self.ASRN = ASRN(targetH, nc, nclass, nh, BidirDecoder, CUDA)

    # Метод для корректировки (обработки) изображений через MORN
    def rectify(self, x, test=False):
        # Применяем MORN для обработки входных данных
        return self.MORN(x, test)

    # Основной метод forward для прохождения данных через ASRN после обработки через MORN
    def forward(self, x_rectified, length, text, text_rev, test=False):
        # Прогоняем откорректированные изображения через ASRN для распознавания текста
        return self.ASRN(x_rectified, length, text, text_rev, test)
