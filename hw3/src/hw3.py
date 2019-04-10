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
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple
from elit.component import Component
from elit.embedding import FastText
from elit.eval import ChunkF1
from src.util import tsv_reader
from keras.utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
from keras.models import Model, load_model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, \
    Flatten, concatenate

class NamedEntityRecognizer(Component):
    def __init__(self, resource_dir: str, embedding_file='fasttext-50-180614.bin'):
        """
        Initializes all resources and the model.
        :param resource_dir: a path to the directory where resource files are located.
        """
        self.vsm = FastText(os.path.join(resource_dir, embedding_file))
        self.resource_dir = resource_dir

        trn_data = self.format_data(tsv_reader(resource_dir, 'conll03.eng.trn.tsv'))
        dev_data = self.format_data(tsv_reader(resource_dir, 'conll03.eng.dev.tsv'))
        tst_data = self.format_data(tsv_reader(resource_dir, 'conll03.eng.tst.tsv'))

        token_dic = {}
        for sentences in trn_data + dev_data + tst_data:
            for words in sentences:
                token = words[0]
                token_dic[token] = True

        tokens = list(token_dic.keys())
        tokens_emb = self.vsm.emb_list(tokens)

        trn_sentence = self.get_char_inform(trn_data)
        dev_sentence = self.get_char_inform(dev_data)
        tst_sentence = self.get_char_inform(tst_data)

        ## parepare labe and words
        label_set = set()
        words = {}
        for dataset in [trn_sentence, dev_sentence, tst_sentence]:
            for sentence in dataset:
                for token, char, label in sentence:
                    if label != 'XX':
                        label_set.add(label)
                        words[token.lower()] = True

        ## label index
        label_idx = {}
        for label in label_set:
            label_idx[label] = len(label_idx)
        self.label_idx = label_idx

        ## case index and case embedding
        case_idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                    'contains_digit': 6, 'PADDING_TOKEN': 7}
        self.case_embeddings = np.identity(len(case_idx), dtype='float32')
        self.case_idx = case_idx

        ## word to index and word embedding
        word_idx = {}
        word_embeddings = []

        df = pd.DataFrame([tokens, tokens_emb])
        combine_embeddings = df.T.values.tolist()

        # for line in combine_embeddings:
        for i in range(len(combine_embeddings)):
            split = combine_embeddings[i]
            word = split[0]

            if len(word_idx) == 0:  
                word_idx["PADDING_TOKEN"] = len(word_idx)
                vector = np.zeros(len(split[1]))  
                word_embeddings.append(vector)
                word_idx["UNKNOWN_TOKEN"] = len(word_idx)
                vector = np.random.uniform(-0.25, 0.25, len(split[1]))
                word_embeddings.append(vector)

            if split[0].lower() in words:
                vector = np.array([float(num) for num in split[1]])
                word_embeddings.append(vector)
                word_idx[split[0]] = len(word_idx)

        self.word_idx = word_idx
        self.word_embeddings = np.array(word_embeddings)

        ## char index
        char_idx = {"PADDING": 0, "UNKNOWN": 1}
        for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
            char_idx[c] = len(char_idx)

        self.char_idx = char_idx

        ## prepare dataset
        train_set = self.padding(self.get_embedded_data(trn_sentence, word_idx, label_idx, case_idx, char_idx))
        dev_set = self.padding(self.get_embedded_data(dev_sentence, word_idx, label_idx, case_idx, char_idx))
        test_set = self.padding(self.get_embedded_data(tst_sentence, word_idx, label_idx, case_idx, char_idx))

        self.idx2Label = {v: k for k, v in label_idx.items()}
        self.train_batch, self.train_batch_len = self.get_batch(train_set)
        self.dev_batch, self.dev_batch_len = self.get_batch(dev_set)
        self.test_batch, self.test_batch_len = self.get_batch(test_set)

    def get_word_embd(self):
        return self.word_embeddings

    def get_case_emb(self):
        return self.case_embeddings

    def get_char2index(self):
        return self.char_idx

    def prepare_data(self, data):
        data = self.format_data(data)
        sentences = self.get_char_inform(data)
        dataset = self.padding(
            self.get_embedded_data(sentences, self.word_idx, self.label_idx, self.case_idx, self.char_idx))
        batch, _ = self.get_batch(dataset)
        return batch

    def get_model(self):
        word_embeddings = self.word_embeddings
        case_embeddings = self.case_embeddings
        char_idx = self.char_idx
        label_idx = self.label_idx

        words_input = Input(shape=(None,), dtype='int32', name='words_input')
        words = Embedding(input_dim=word_embeddings.shape[0], output_dim=word_embeddings.shape[1],
                          weights=[word_embeddings],
                          trainable=False)(words_input)

        casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
        casing = Embedding(output_dim=case_embeddings.shape[1], input_dim=case_embeddings.shape[0],
                           weights=[case_embeddings],
                           trainable=False)(casing_input)

        character_input = Input(shape=(None, 52,), name='char_input')
        embed_char_out = TimeDistributed(
            Embedding(len(char_idx), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
            name='char_embedding')(character_input)

        dropout = Dropout(0.5)(embed_char_out)
        conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1))(
            dropout)
        maxpool_out = TimeDistributed(MaxPooling1D(52))(conv1d_out)
        char = TimeDistributed(Flatten())(maxpool_out)
        char = Dropout(0.5)(char)

        output = concatenate([words, casing, char])
        output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
        output = TimeDistributed(Dense(len(label_idx), activation='softmax'))(output)
        model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
        model.summary()
        return model

    def get_casing(self, word, case_tag):
        casing = 'other'

        num_digit = 0
        for char in word:
            if char.isdigit():
                num_digit += 1

        digit_frac = num_digit / float(len(word))
        if word.isdigit():
            casing = 'numeric'
        elif digit_frac > 0.5:
            casing = 'mainly_numeric'
        elif word.islower():
            casing = 'allLower'
        elif word.isupper():
            casing = 'allUpper'
        elif word[0].isupper():
            casing = 'initialUpper'
        elif num_digit > 0:
            casing = 'contains_digit'

        return case_tag[casing]

    def get_batch(self, data):
        l = []
        for i in data:
            l.append(len(i[0]))
        l = set(l)
        batches = []
        batch_len = []
        z = 0
        for i in l:
            for batch in data:
                if len(batch[0]) == i:
                    batches.append(batch)
                    z += 1
            batch_len.append(z)
        return batches, batch_len


    def get_embedded_data(self, sentences, word_idx, label_idx, case_idx, char_idx):
        unknownIdx = word_idx['UNKNOWN_TOKEN']
        paddingIdx = word_idx['PADDING_TOKEN']
        dataset = []
        wordCount = 0
        unknownWordCount = 0

        for sentence in sentences:
            wordIndices = []
            caseIndices = []
            charIndices = []
            labelIndices = []
            for word, char, label in sentence:
                wordCount += 1
                if word in word_idx:
                    wordIdx = word_idx[word]
                elif word.lower() in word_idx:
                    wordIdx = word_idx[word.lower()]
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1
                charIdx = []
                for x in char:
                    charIdx.append(char_idx[x])
                # Get the label and map to int
                wordIndices.append(wordIdx)
                caseIndices.append(self.get_casing(word, case_idx))
                charIndices.append(charIdx)
                if label != 'XX':
                    labelIndices.append(label_idx[label])

            dataset.append([wordIndices, caseIndices, charIndices, labelIndices])

        return dataset

    def get_mini_batch(self, dataset, batch_len):
        start = 0
        for i in batch_len:
            tokens = []
            caseing = []
            char = []
            labels = []
            data = dataset[start:i]
            start = i
            for dt in data:
                t, c, ch, l = dt
                l = np.expand_dims(l, -1)
                tokens.append(t)
                caseing.append(c)
                char.append(ch)
                labels.append(l)
            yield np.asarray(labels), np.asarray(tokens), np.asarray(caseing), np.asarray(char)

    def get_char_inform(self, Sentences):
        for i, sentence in enumerate(Sentences):
            for j, data in enumerate(sentence):
                chars = [c for c in data[0]]
                Sentences[i][j] = [data[0], chars, data[1]]
        return Sentences

    def padding(self, Sentences):
        maxlen = 52
        for sentence in Sentences:
            char = sentence[2]
            for x in char:
                maxlen = max(maxlen, len(x))
        for i, sentence in enumerate(Sentences):
            Sentences[i][2] = pad_sequences(Sentences[i][2], 52, padding='post')
        return Sentences

    def load(self, model_path: str, **kwargs):
        """
        Load the pre-trained model.
        :param model_path:
        :param kwargs:
        """
        # TODO: to be filled
        self.model = load_model(model_path)
        return self.model


    def save(self, model_path: str, **kwargs):
        """
        Saves the current model to the path.
        :param model_path:
        :param kwargs:
        """
        # TODO: to be filled
        self.model.save(model_path)

    def tuple2list(self, data):
        res = []
        for el in data:
            res += el
        return res

    def format_data(self, data):
        temp1 = []
        for ele in data:
            token = ele[0]
            label = ele[1]
            temp2 = []
            for i in range(len(token)):
                temp2.append([label[i], token[i]])
            temp1.append(temp2)
        return temp1

    def train(self, trn_data: List[Tuple[List[str], List[str]]], dev_data: List[Tuple[List[str], List[str]]], *args,
              **kwargs):
        """
        Trains the model.
        :param trn_data: the training data.
        :param dev_data: the development data.
        :param args:
        :param kwargs:
        :return:
        """

        self.model = self.get_model()

        epochs = 80
        for epoch in range(epochs):
            print("Epoch %d/%d" % (epoch, epochs))
            a = Progbar(len(self.train_batch_len))
            for i, batch in enumerate(self.get_mini_batch(self.train_batch, self.train_batch_len)):
                labels, tokens, casing, char = batch
                self.model.train_on_batch([tokens, casing, char], labels)
                a.update(i)
            a.update(i + 1)
            print(' ')

        # model.save("hw3-model")
        save_data = [self.word_embeddings, self.case_embeddings, self.idx2Label, self.word_idx, self.word_idx,
                     self.label_idx, self.case_idx, self.char_idx]

        with open(os.path.join(resource_dir, 'pickle'), 'wb') as handle:
            pickle.dump(save_data, handle)

    def dev_evaluate(self, model, data):

        correctLabels = []
        predLabels = []
        b = Progbar(len(data))
        for i, data1 in enumerate(data):
            tokens, casing, char, labels = data1
            tokens = np.asarray([tokens])
            casing = np.asarray([casing])
            char = np.asarray([char])
            pred = model.predict([tokens, casing, char], verbose=False)[0]
            pred = pred.argmax(axis=-1)  # Predict the classes
            correctLabels.append(labels)
            predLabels.append(pred)
            b.update(i)
        b.update(i + 1)

        label_pred = []
        for sentence in predLabels:
            label_pred.append([self.idx2Label[element] for element in sentence])
        label_correct = []
        for sentence in correctLabels:
            label_correct.append([self.idx2Label[element] for element in sentence])

        acc = ChunkF1()
        for pred, label in zip(label_pred, label_correct):
            acc.update(pred, label)

        print(float(acc.get()[1]))

    def decode(self, data: List[Tuple[List[str], List[str]]], **kwargs) -> List[List[str]]:
        """
        :param data:
        :param kwargs:
        :return: the list of predicted labels.
        """

        with open(os.path.join(self.resource_dir, 'pickle'), 'rb') as handle:
            save_data = pickle.load(handle)

        self.word_embeddings, self.case_embeddings, self.idx2Label, self.word_idx, self.word_idx, self.label_idx, self.case_idx, self.char_idx = save_data

        dataset = self.prepare_data(data)
        model = self.load(os.path.join(resource_dir, 'hw3-model'))

        correctLabels = []
        predLabels = []
        b = Progbar(len(dataset))
        for i, data1 in enumerate(dataset):
            tokens, casing, char, labels = data1
            tokens = np.asarray([tokens])
            casing = np.asarray([casing])
            char = np.asarray([char])
            pred = model.predict([tokens, casing, char], verbose=False)[0]
            pred = pred.argmax(axis=-1)  # Predict the classes
            correctLabels.append(labels)
            predLabels.append(pred)
            b.update(i)
        b.update(i + 1)
        label_pred = []

        for sentence in predLabels:
            label_pred.append([self.idx2Label[element] for element in sentence])
        label_correct = []
        for sentence in correctLabels:
            label_correct.append([self.idx2Label[element] for element in sentence])
        return label_pred, label_correct

    def evaluate(self, data: List[Tuple[List[str], List[str]]], **kwargs) -> float:
        """
        :param data:
        :param kwargs:
        :return: the accuracy of this model.
        """

        labels = [y for y, _ in data]
        preds, labels = self.decode(data)
        # print(preds)
        # print(labels)

        acc = ChunkF1()
        for pred, label in zip(preds, labels):
            acc.update(pred, label)

        # print(float(acc.get()[1]))
        return float(acc.get()[1])


if __name__ == '__main__':
    resource_dir = os.environ.get('RESOURCE')
    # resource_dir = '../res'
    sentiment_analyzer = NamedEntityRecognizer(resource_dir)
    trn_data = tsv_reader(resource_dir, 'conll03.eng.trn.tsv')
    dev_data = tsv_reader(resource_dir, 'conll03.eng.dev.tsv')
    tst_data = tsv_reader(resource_dir, 'conll03.eng.tst.tsv')
    sentiment_analyzer.train(trn_data, dev_data)
    # sentiment_analyzer.evaluate(dev_data)
    sentiment_analyzer.evaluate(tst_data)
    sentiment_analyzer.save(os.path.join(resource_dir, 'hw3-model'))
