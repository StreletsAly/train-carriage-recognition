import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import numpy.random as npr

# Определение класса fracPickup, который наследуется от nn.Module
class fracPickup(nn.Module):

    # Конструктор класса
    def __init__(self, CUDA=True):
        super(fracPickup, self).__init__() # Вызов конструктора родительского класса nn.Module
        self.cuda = CUDA # Флаг использования GPU (если доступен)

    # Определение прямого прохода (forward pass) через слой
    def forward(self, x):
        # Получение формы входного тензора
        x_shape = x.size()
        # Убедимся, что тензор имеет 4 измерения (batch, channel, height, width)
        assert len(x_shape) == 4
        # Убедимся, что высота равна 1 (работаем с изображением высотой 1 пиксель)
        assert x_shape[2] == 1

        # Количество итераций для изменения сетки (можно варьировать)
        fracPickup_num = 1

        # Создание списка координат для высоты и ширины
        h_list = 1. # Высота фиксирована
        # Создание линейной сетки координат для ширины от -1 до 1
        w_list = np.arange(x_shape[3])*2./(x_shape[3]-1)-1

        # Цикл для случайного изменения координат сетки
        for i in range(fracPickup_num):
            # Генерация случайного индекса в пределах длины w_list
            idx = int(npr.rand()*len(w_list))
            # Проверка, чтобы индекс не выходил за пределы (игнорируем крайние точки)
            if idx <= 0 or idx >= x_shape[3]-1:
                continue
            # Генерация случайного коэффициента beta (определяет степень изменения)
            beta = npr.rand()/4.
            # Пересчет значений соседних элементов с учетом beta
            value0 = (beta*w_list[idx] + (1-beta)*w_list[idx-1])
            value1 = (beta*w_list[idx-1] + (1-beta)*w_list[idx])
            # Обновляем значения в списке
            w_list[idx-1] = value0
            w_list[idx] = value1

        # Генерация 2D сетки координат (meshgrid) с помощью новых w_list и h_list
        grid = np.meshgrid(
                w_list,  # по ширине
                h_list,  # по высоте
                indexing='ij'
            )
        # Объединяем сетку по последней оси
        grid = np.stack(grid, axis=-1)
        # Меняем порядок осей: [height, width, 2] -> [width, height, 2]
        grid = np.transpose(grid, (1, 0, 2))
        # Добавляем размерность batch (для обработки нескольких изображений)
        grid = np.expand_dims(grid, 0)
        # Клонируем сетку для всех изображений в batch
        grid = np.tile(grid, [x_shape[0], 1, 1, 1])
        # Преобразуем numpy массив в тензор PyTorch
        grid = torch.from_numpy(grid).type(x.data.type())
        # Переносим тензор на GPU, если активен флаг cuda
        if self.cuda:
            grid = grid.cuda()
        # Оборачиваем сетку в Variable (тензор без градиента)
        self.grid = Variable(grid, requires_grad=False)

        # Применяем grid_sample для выборки значений из x по сетке координат
        x_offset = nn.functional.grid_sample(x, self.grid)

        # Возвращаем результат выборки (модифицированное изображение)
        return x_offset
