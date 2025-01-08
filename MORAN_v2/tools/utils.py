import torch
import torch.nn as nn
from torch.autograd import Variable
import collections

class strLabelConverterForAttention(object):
    """Конвертер между строками и метками для использования в модели с вниманием.

        ЗАМЕЧАНИЕ:
            Добавляет символ `EOS` (конец строки) в алфавит для работы механизма внимания.

        Аргументы:
            alphabet (str): строка, содержащая набор возможных символов.
            ignore_case (bool, по умолчанию=True): игнорировать ли регистр символов.
        """

    def __init__(self, alphabet, sep):
        self._scanned_list = False # Флаг, указывающий, была ли выполнена проверка текста
        self._out_of_list = '' # Строка для хранения символов вне алфавита
        self._ignore_case = True # Игнорировать ли регистр символов
        self.sep = sep # Разделитель для алфавита
        self.alphabet = alphabet.split(sep) # Разделение алфавита на список символов

        # Создание словаря символов с индексами
        self.dict = {}
        for i, item in enumerate(self.alphabet):
            self.dict[item] = i

    def scan(self, text):
        """Проверяет текст на наличие символов, не входящих в алфавит.

                Аргументы:
                    text (list[str]): список строк для проверки.

                Возвращает:
                    tuple[str]: проверенные строки.
                """
        text_tmp = text
        text = []
        for i in range(len(text_tmp)):
            text_result = ''
            for j in range(len(text_tmp[i])):
                chara = text_tmp[i][j].lower() if self._ignore_case else text_tmp[i][j]
                if chara not in self.alphabet:
                    if chara in self._out_of_list:
                        continue
                    else:
                        # Запись неизвестных символов в файл.
                        self._out_of_list += chara
                        file_out_of_list = open("out_of_list.txt", "a+")
                        file_out_of_list.write(chara + "\n")
                        file_out_of_list.close()
                        print('" %s " is not in alphabet...' % chara)
                        continue
                else:
                    text_result += chara
            text.append(text_result)
        text_result = tuple(text)
        self._scanned_list = True
        return text_result

    def encode(self, text, scanned=True):
        """Кодирует строки в тензоры.

        Аргументы:
            text (str или list[str]): текст для кодирования.
            scanned (bool): использовать ли предварительную проверку текста.

        Возвращает:
            tuple[torch.LongTensor, torch.LongTensor]: кодированные символы и их длины.
        """
        self._scanned_list = scanned
        if not self._scanned_list:
            text = self.scan(text)

        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.LongTensor(text), torch.LongTensor(length))

    def decode(self, t, length):
        """Декодирует тензоры обратно в строки.

        Аргументы:
            t (torch.IntTensor): закодированные символы.
            length (torch.IntTensor): длины строк.

        Возвращает:
            str или list[str]: декодированные строки.
        """
        if length.numel() == 1: # Если длина одна (одна строка)
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            return ''.join([self.alphabet[i] for i in t])
        else:
            # Если длина указывает на несколько строк (батч-режим)
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.LongTensor([l])))
                index += l
            return texts

class averager(object):
    """Класс для вычисления среднего значения для `torch.Variable` и `torch.Tensor`."""

    def __init__(self):
        self.reset() # Инициализация переменных

    def add(self, v):
        """Добавляет значение для вычисления среднего.

                Аргументы:
                    v (Variable или Tensor): данные для добавления.
                """
        if isinstance(v, Variable):
            count = v.data.numel() # Число элементов в данных
            v = v.data.sum() # Сумма элементов
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count # Увеличение общего числа элементов
        self.sum += v # Увеличение общей суммы

    def reset(self):
        """Сбрасывает накопленные значения."""
        self.n_count = 0
        self.sum = 0

    def val(self):
        """Возвращает среднее значение.

                Возвращает:
                    float: среднее значение.
                """
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def loadData(v, data):
    """Загружает данные в тензор, изменяя его размер при необходимости.

        Аргументы:
            v (Tensor): тензор для изменения.
            data (Tensor): данные для копирования.
        """
    major, _ = get_torch_version()

    if major >= 1: # Для версий PyTorch 1.0 и выше
        v.resize_(data.size()).copy_(data)
    else: # Для более старых версий
        v.data.resize_(data.size()).copy_(data)

def get_torch_version():
    """Определяет версию PyTorch и возвращает её в виде целых чисел.
    Возвращает:
        tuple[int, int]: основные и второстепенные версии PyTorch.
    """
    torch_version = str(torch.__version__).split(".")
    return int(torch_version[0]), int(torch_version[1])
