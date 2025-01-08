import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import sampler
import lmdb
import six
import sys
from PIL import Image
import numpy as np

# Класс для работы с датасетом в формате LMDB
class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, reverse=False, alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):
        # Открытие базы данных LMDB
        self.env = lmdb.open(
            root,
            max_readers=1, # Максимальное количество читателей
            readonly=True, # Только для чтения
            lock=False,    # Без блокировки
            readahead=False,  # Отключение предзагрузки
            meminit=False)   # Без инициализации памяти

        # Проверка, что база данных открыта успешно
        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        # Чтение информации о количестве примеров из LMDB
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode())) # Получаем число примеров в датасете
            self.nSamples = nSamples

        # Сохранение переданных параметров
        self.transform = transform
        self.alphabet = alphabet
        self.reverse = reverse

    def __len__(self):
        # Возвращаем количество примеров в датасете
        return self.nSamples

    def __getitem__(self, index):
        # Проверка индекса
        assert index <= len(self), 'index range error'
        index += 1 # Индексация с 1

        # Чтение изображения из базы данных LMDB
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index # Ключ для изображения
            imgbuf = txn.get(img_key.encode()) # Чтение изображения

            # Преобразование байтового потока в изображение
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L') # Открытие изображения в оттенках серого
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1] # Если изображение повреждено, пропустить и перейти к следующему

            # Чтение метки (текста) из базы данных
            label_key = 'label-%09d' % index # Ключ для метки
            label = str(txn.get(label_key.encode()).decode('utf-8')) # Чтение и декодирование метки

            # Фильтрация метки, оставляя только символы, которые есть в алфавите
            label = ''.join(label[i] if label[i].lower() in self.alphabet else '' 
                for i in range(len(label)))
            if len(label) <= 0:
                return self[index + 1] # Если метка пуста, пропустить

            # Если флаг reverse включен, инвертируем метку
            if self.reverse:
                label_rev = label[-1::-1] # Реверс строки
                label_rev += '$' # Добавляем символ конца строки
            label += '$' # Добавляем символ конца строки

            # Применяем трансформации к изображению, если они заданы
            if self.transform is not None:
                img = self.transform(img)

        # Возвращаем кортеж с изображением и меткой (или инвертированной меткой, если reverse=True)
        if self.reverse:
            return (img, label, label_rev)
        else:
            return (img, label)

# Класс для нормализации и изменения размера изображения
class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size # Новый размер изображения
        self.interpolation = interpolation # Способ интерполяции (по умолчанию билинейная)
        self.toTensor = transforms.ToTensor() # Преобразование изображения в тензор

    def __call__(self, img):
        # Изменяем размер изображения и преобразуем в тензор
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5) # Нормализация: вычитание 0.5 и деление на 0.5
        return img

# Класс для случайной выборки последовательностей из датасета
class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source) # Количество примеров в датасете
        self.batch_size = batch_size # Размер батча


    def __len__(self):
        return self.num_samples

    def __iter__(self):
        # Количество полных батчей
        n_batch = len(self) // self.batch_size
        # Оставшиеся элементы, которые не входят в полный батч
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        # Генерация случайных индексов для каждого батча
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size) # Случайное начало батча
            batch_index = random_start + torch.arange(0, self.batch_size) # Индексы для батча
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # Обработка хвоста (если есть)
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)
