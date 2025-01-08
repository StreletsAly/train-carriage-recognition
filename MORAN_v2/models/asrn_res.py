import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from MORAN_v2.models.fracPickup import fracPickup

# Двухнаправленная LSTM (Bidirectional LSTM) для обработки последовательностей
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        # LSTM с двумя направлениями (bidirectional=True)
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.3)
        # Линейный слой для преобразования выходных данных LSTM в финальные выходные данные
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        # Прогоняем вход через LSTM
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size() # Размеры выходных данных: (time_steps, batch_size, hidden_size)
        t_rec = recurrent.view(T * b, h) # Изменяем форму для линейного слоя

        # Прогоняем через линейный слой для получения предсказания
        output = self.embedding(t_rec)
        output = output.view(T, b, -1) # Возвращаем форму (time_steps, batch_size, output_size)

        return output

# Ячейка внимания (Attention Cell), используется для фокусировки внимания на разных частях входа
class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings=128, CUDA=True):
        super(AttentionCell, self).__init__()
        # Линейный слой для преобразования входных данных в скрытое состояние
        self.i2h = nn.Linear(input_size, hidden_size,bias=False)
        # Линейный слой для преобразования предыдущего скрытого состояния
        self.h2h = nn.Linear(hidden_size, hidden_size)
        # Линейный слой для вычисления весов внимания (score)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        # GRU-ячейка для обновления скрытого состояния
        self.rnn = nn.GRUCell(input_size+num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_embeddings = num_embeddings
        self.fracPickup = fracPickup(CUDA=CUDA)

    def forward(self, prev_hidden, feats, cur_embeddings, test=False):
        # Получаем размерности данных
        nT = feats.size(0) # Количество временных шагов
        nB = feats.size(1) # Размер батча
        nC = feats.size(2) # Количество каналов
        hidden_size = self.hidden_size

        # Преобразуем признаки с помощью линейного слоя
        feats_proj = self.i2h(feats.view(-1,nC))
        prev_hidden_proj = self.h2h(prev_hidden).view(1,nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(-1, hidden_size)
        # Вычисляем вес внимания для каждого временного шага
        emition = self.score(F.tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT,nB)

        # Вычисляем коэффициенты внимания (alpha)
        alpha = F.softmax(emition, 0) # Применяем softmax для нормализации по времени

        if not test:
            # Применяем fracPickup, чтобы "выбирать" части признаков с максимальным вниманием
            alpha_fp = self.fracPickup(alpha.transpose(0,1).contiguous().unsqueeze(1).unsqueeze(2)).squeeze()
            context = (feats * alpha_fp.transpose(0,1).contiguous().view(nT,nB,1).expand(nT, nB, nC)).sum(0).squeeze(0) # nB * nC
            if len(context.size()) == 1:
                context = context.unsqueeze(0)
            context = torch.cat([context, cur_embeddings], 1)
            # Обновляем скрытое состояние с помощью GRU
            cur_hidden = self.rnn(context, prev_hidden)
            return cur_hidden, alpha_fp
        else:
            # Для тестирования, без fracPickup
            context = (feats * alpha.view(nT,nB,1).expand(nT, nB, nC)).sum(0).squeeze(0) # nB * nC
            if len(context.size()) == 1:
                context = context.unsqueeze(0)
            context = torch.cat([context, cur_embeddings], 1)
            cur_hidden = self.rnn(context, prev_hidden)
            return cur_hidden, alpha

# Механизм внимания (Attention), использующий AttentionCell
class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_embeddings=128, CUDA=True):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_embeddings, CUDA=CUDA)
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Линейный слой для получения окончательных предсказаний
        self.generator = nn.Linear(hidden_size, num_classes)
        self.char_embeddings = Parameter(torch.randn(num_classes+1, num_embeddings)) # Векторные представления символов
        self.num_embeddings = num_embeddings
        self.num_classes = num_classes
        self.cuda = CUDA


    def forward(self, feats, text_length, text, test=False):

        nT = feats.size(0) # Количество временных шагов
        nB = feats.size(1) # Размер батча
        nC = feats.size(2) # Количество каналов
        hidden_size = self.hidden_size
        input_size = self.input_size
        assert(input_size == nC)
        assert(nB == text_length.numel())

        num_steps = text_length.data.max()
        num_labels = text_length.data.sum()

        if not test:

            # Подготавливаем таргеты для обучения
            targets = torch.zeros(nB, num_steps+1).long()
            if self.cuda:
                targets = targets.cuda()
            start_id = 0

            for i in range(nB):
                targets[i][1:1+text_length.data[i]] = text.data[start_id:start_id+text_length.data[i]]+1
                start_id = start_id+text_length.data[i]
            targets = Variable(targets.transpose(0,1).contiguous())

            output_hiddens = Variable(torch.zeros(num_steps, nB, hidden_size).type_as(feats.data))
            hidden = Variable(torch.zeros(nB,hidden_size).type_as(feats.data))

            for i in range(num_steps):
                cur_embeddings = self.char_embeddings.index_select(0, targets[i])
                hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings, test)
                output_hiddens[i] = hidden

            # Переорганизуем выходы и генерируем окончательные вероятности
            new_hiddens = Variable(torch.zeros(num_labels, hidden_size).type_as(feats.data))
            b = 0
            start = 0

            for length in text_length.data:
                new_hiddens[start:start+length] = output_hiddens[0:length,b,:]
                start = start + length
                b = b + 1

            probs = self.generator(new_hiddens)
            return probs

        else:

            hidden = Variable(torch.zeros(nB,hidden_size).type_as(feats.data))
            targets_temp = Variable(torch.zeros(nB).long().contiguous())
            probs = Variable(torch.zeros(nB*num_steps, self.num_classes))
            if self.cuda:
                targets_temp = targets_temp.cuda()
                probs = probs.cuda()

            for i in range(num_steps):
                cur_embeddings = self.char_embeddings.index_select(0, targets_temp)
                hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings, test)
                hidden2class = self.generator(hidden)
                probs[i*nB:(i+1)*nB] = hidden2class
                _, targets_temp = hidden2class.max(1)
                targets_temp += 1

            probs = probs.view(num_steps, nB, self.num_classes).permute(1, 0, 2).contiguous()
            probs = probs.view(-1, self.num_classes).contiguous()
            probs_res = Variable(torch.zeros(num_labels, self.num_classes).type_as(feats.data))
            b = 0
            start = 0

            for length in text_length.data:
                probs_res[start:start+length] = probs[b*num_steps:b*num_steps+length]
                start = start + length
                b = b + 1

            return probs_res

# Блок остаточных связей для ResNet (Residual Block)
class Residual_block(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super(Residual_block, self).__init__()
        self.downsample = None
        flag = False
        # Если stride больше 1, добавляем downsampling
        if isinstance(stride, tuple):
            if stride[0] > 1:
                self.downsample = nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1),nn.BatchNorm2d(c_out, momentum=0.01))
                flag = True
        else:
            if stride > 1:
                self.downsample = nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1),nn.BatchNorm2d(c_out, momentum=0.01))
                flag = True
        if flag:
            self.conv1 = nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1),
                                    nn.BatchNorm2d(c_out, momentum=0.01))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, 0),
                                    nn.BatchNorm2d(c_out, momentum=0.01))
        self.conv2 = nn.Sequential(nn.Conv2d(c_out, c_out, 3, 1, 1),
                                   nn.BatchNorm2d(c_out, momentum=0.01))  
        self.relu = nn.ReLU()

    def forward(self,x):
        residual = x 
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(residual + conv2)

# Архитектура ResNet для извлечения признаков из изображений
class ResNet(nn.Module):
    def __init__(self,c_in):
        super(ResNet,self).__init__()
        self.block0 = nn.Sequential(nn.Conv2d(c_in, 32, 3, 1, 1),nn.BatchNorm2d(32, momentum=0.01))
        self.block1 = self._make_layer(32, 32, 2, 3)
        self.block2 = self._make_layer(32, 64, 2, 4)
        self.block3 = self._make_layer(64, 128, (2,1), 6)
        self.block4 = self._make_layer(128, 256, (2,1), 6)
        self.block5 = self._make_layer(256, 512, (2,1), 3)

    def _make_layer(self,c_in,c_out,stride,repeat=3):
        layers = []
        layers.append(Residual_block(c_in, c_out, stride))
        for i in range(repeat - 1):
            layers.append(Residual_block(c_out, c_out, 1))
        return nn.Sequential(*layers)

    def forward(self,x):
        block0 = self.block0(x)
        block1 = self.block1(block0)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        return block5

# Основная модель ASRN для распознавания текста
class ASRN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, BidirDecoder=False, CUDA=True):
        super(ASRN, self).__init__()
        assert imgH % 16 == 0, 'imgH must be a multiple of 16'

        # CNN для извлечения признаков
        self.cnn = ResNet(nc)

        # RNN для обработки признаков как последовательности
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh),
            )

        self.BidirDecoder = BidirDecoder
        # Двусторонний декодер
        if self.BidirDecoder:
            self.attentionL2R = Attention(nh, nh, nclass, 256, CUDA=CUDA)
            self.attentionR2L = Attention(nh, nh, nclass, 256, CUDA=CUDA)
        else:
            self.attention = Attention(nh, nh, nclass, 256, CUDA=CUDA)

        # Инициализация весов
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out', a=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def forward(self, input, length, text, text_rev, test=False):
        # Извлечение признаков из изображения с помощью CNN
        conv = self.cnn(input)
        
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1).contiguous()  # Преобразование формы для LSTM

        # Обработка через RNN
        rnn = self.rnn(conv)

        # Если используем двусторонний декодер
        if self.BidirDecoder:
            outputL2R = self.attentionL2R(rnn, length, text, test)
            outputR2L = self.attentionR2L(rnn, length, text_rev, test)
            return outputL2R, outputR2L
        else:
            output = self.attention(rnn, length, text, test)
            return output
