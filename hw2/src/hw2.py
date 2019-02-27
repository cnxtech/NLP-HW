# ========================================================================
# Copyright 2019 Emory University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import os
from typing import List, Tuple
import torch.utils.data as Data
from elit.component import Component
from elit.embedding import FastText
from src.util import tsv_reader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class MyDataset(Data.Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels



    def __getitem__(self, index):  # 返回的是tensor
        sent = self.sentences[index]
        target = self.labels[index]
        sent = torch.from_numpy(np.array(sent)).float().unsqueeze(0)
        target = torch.from_numpy(np.array(target))
        # print(sent.shape)

        return sent, target


    def __len__(self):
        return len(self.sentences)


class Net(nn.Module):
    def __init__(self, num_classes=5):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(4,50), stride=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(58,1), stride=1))
        self.fc = nn.Linear(128 * 1, num_classes)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(-1,  128 * 1)
        out = self.fc(out)
        # out = self.dropout(out)
        return out

#
# class Net(nn.Module):
#     def __init__(self, num_classes=5):
#         super(Net, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(1, 10, kernel_size=(3,25), stride=1),
#             # nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2,2), stride=1))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(10, 100, kernel_size=(3,25), stride=1),
#             # nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(56,1), stride=1))
#         self.fc = nn.Linear(100, num_classes)
#
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.reshape(-1, 100)
#         out = self.fc(out)
#         return out



class SentimentAnalyzer(Component):
    def __init__(self, resource_dir: str, embedding_file='fasttext-50-180614.bin'):
        """
        Initializes all resources and the model.
        :param resource_dir: a path to the directory where resource files are located.
        """
        self.vsm = FastText(os.path.join(resource_dir, embedding_file))
        # print(os.path.join(resource_dir, 'sst.trn.tsv'))
        # self.train = pd.read_csv(os.path.join(resource_dir, 'sst.trn.tsv'), sep='\t')

        # TODO: to be filled.
        self.net = Net()
        self.resource_dir = os.environ.get('RESOURCE')

    def pad_x(self,trn_xs):
        max_len = 0
        trn_xs = list(trn_xs)
        for i in trn_xs:
            if len(i) > max_len:
                max_len = len(i)

        max_len = 61

        for i in range(len(trn_xs)):
            if len(trn_xs[i]) <= max_len:
                # temp = np.zeros(50)
                temp = [np.zeros(50) for _ in range(max_len - len(trn_xs[i]))]
                trn_xs[i] = trn_xs[i] + temp
            else:
                trn_xs[i] = trn_xs[i][0:max_len]
        trn_xs = tuple(trn_xs)
        return trn_xs


    def load1(self, model_path: str, **kwargs):
        """
        Load the pre-trained model.
        :param model_path:
        :param kwargs:
        """
        # TODO: to be filled
        model_name = kwargs['name']
        dir =os.path.join(model_path,model_name)
        model = torch.load(dir)
        # print(model)
        return model


    def load(self, model_path: str, **kwargs):
        """
        Load the pre-trained model.
        :param model_path:
        :param kwargs:
        """
        # TODO: to be filled
        # model_name = kwargs['name']+'.plk'
        model = Net()
        dir = model_path
        model.load_state_dict(torch.load(dir))
        # print(model)
        return model

    def save(self, model_path: str, **kwargs):
        """
        Saves the current model to the path.
        :param model_path:
        :param kwargs:
        """
        # TODO: to be filled
        model = self.net
        dir = model_path
        # the_model.state_dict()
        torch.save(model.state_dict(), dir)


    def save1(self, model_path: str, **kwargs):

        model = kwargs['model']
        model_name = kwargs['name']
        dir = os.path.join(model_path, model_name)

        torch.save(model,dir)

    def get_tensor_data(self, dev_xs):
        result = []
        for i in dev_xs:
            x = np.array(i)
            # x = torch.from_numpy(x)
            result.append(x)
        result = np.array(result)
        dev_xs = torch.FloatTensor(result)
        dev_xs = dev_xs.unsqueeze(1)
        return  dev_xs

    def plot_fig(self, x1,x2, epoch):
        fig, ax = plt.subplots()
        ax.plot(x1, label='training')
        ax.plot(x2, label='validation')
        ax.set(xlabel='epoch', ylabel='accuracy',
               title='model accuracy')
        ax.grid()

        name = 'model'+ str(epoch)+'.png'
        dir = os.path.join(self.resource_dir,name)
        plt.legend()
        fig.savefig(dir)
        # plt.show()


    def train(self, trn_data: List[Tuple[int, List[str]]], dev_data: List[Tuple[int, List[str]]], *args, **kwargs):
        """
        Trains the model.
        :param trn_data: the training data.
        :param dev_data: the development data.
        :param args:
        :param kwargs:
        :return:
        """
        trn_ys, trn_xs = zip(*[(y, self.vsm.emb_list(x)) for y, x in trn_data])

        trn_xs = self.pad_x(trn_xs)

        dev_ys, dev_xs = zip(*[(y, self.vsm.emb_list(x)) for y, x in dev_data])

        dev_xs = self.pad_x(dev_xs)


        train_data = MyDataset(trn_xs, trn_ys)
        vali_data = MyDataset(dev_xs, dev_ys)


        dev_xs = self.get_tensor_data(dev_xs)
        dev_ys = list(dev_ys)


        train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
        # vali_loader = Data.DataLoader(dataset=vali_data)


        # TODO: to be filled

        net = self.net
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.MultiLabelSoftMarginLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

        total_epoch = 22

        train_acc = []
        vali_acc = []
        for epoch in range(total_epoch):

            for i, data in enumerate(train_loader):

                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics

                # if i % 20 == 19:  # print every 2000 mini-batches

            vali_output = net(dev_xs)
            vali_pred_y = torch.max(vali_output, 1)[1].data.numpy()
            vali_accuracy = float((vali_pred_y == dev_ys).astype(int).sum()) / float(len(dev_ys))

            train_output = net(inputs)
            train_pred_y = torch.max(train_output, 1)[1].data.numpy()
            train_accuracy = float((train_pred_y == labels.tolist()).astype(int).sum()) / float(len(labels))

            # print('epoch:', epoch, 'of', total_epoch ,'| train loss: %.4f' % loss.data.numpy(),'| validation accuracy: %.4f' % vali_accuracy)
            print('epoch:', epoch, 'of', total_epoch ,'| train accuracy: %.4f' % train_accuracy,'| validation accuracy: %.4f' % vali_accuracy)

            train_acc.append(train_accuracy)
            vali_acc.append(vali_accuracy)

        self.plot_fig(train_acc,vali_acc, epoch)
        self.net = net

        self.save(os.path.join(self.resource_dir, 'hw2-model'))



    def decode(self, data: List[Tuple[int, List[str]]], **kwargs) -> List[int]:
        """
        :param data:
        :param kwargs:
        :return: the list of predicted labels.
        """
        xs = [self.vsm.emb_list(x) for _, x in data]
        xs = self.pad_x(xs)
        inputs = self.get_tensor_data(xs)
        model = self.load(os.path.join(self.resource_dir, 'hw2-model'))
        outputs = model(inputs)
        pred_y = torch.max(outputs, 1)[1].data.numpy()
        return pred_y

        # TODO: to be filled

    def evaluate(self, data: List[Tuple[int, List[str]]], **kwargs) -> float:
        """
        :param data:
        :param kwargs:
        :return: the accuracy of this model.
        """
        gold_labels = [y for y, _ in data]
        auto_labels = self.decode(data)
        total = correct = 0
        for gold, auto in zip(gold_labels, auto_labels):
            if gold == auto:
                correct += 1
            total += 1
        # print(100.0 * correct / total)
        return 100.0 * correct / total


if __name__ == '__main__':
    resource_dir = os.environ.get('RESOURCE')
    # resource_dir = '../res/'
    sentiment_analyzer = SentimentAnalyzer(resource_dir)
    trn_data = tsv_reader(resource_dir, 'sst.trn.tsv')
    dev_data = tsv_reader(resource_dir, 'sst.dev.tsv')
    tst_data = tsv_reader(resource_dir, 'sst.tst.tsv')
    sentiment_analyzer.train(trn_data, dev_data)
    sentiment_analyzer.evaluate(tst_data)
    sentiment_analyzer.save(os.path.join(resource_dir, 'hw2-model'))
    sentiment_analyzer.load(os.path.join(resource_dir, 'hw2-model'))
