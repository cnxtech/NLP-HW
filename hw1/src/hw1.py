# ========================================================================
# Copyright 2019 ELIT
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
import csv
import os
import glob
import numpy as np
from typing import List, Any
import pandas as pd
from elit.component import Component

__author__ = "Gary Lai, Jinho D. Choi"


class HashtagSegmenter(Component):

    def __init__(self, resource_dir: str):
        """
        :param resource_dir: a path to the directory where resource files are located.
        """
        # initialize the n-grams
        ngram_filenames = glob.glob(os.path.join(resource_dir, '[1-6]gram.txt'))
        # TODO: initialize resources

        ngram_filenames.sort()
        df1 = pd.read_csv(ngram_filenames[0], sep='\t', names=['count', 'term'])
        df2 = pd.read_csv(ngram_filenames[1], sep='\t', names=['count', 'term'])

        # clean data
        df2['term'] = df2['term'].map(lambda x: x.lower())
        df2['term1'] = df2['term'].map(lambda x: x.split()[0])
        df2['term2'] = df2['term'].map(lambda x: x.split()[1])

        df_temp = df2[['count', 'term1', 'term2']]
        df21 = df_temp.groupby('term1').sum()['count'].reset_index()

        df2['term'] = df2['term'].map(lambda x: tuple(x.split()))
        df2 = df2[['count', 'term']]

        # # compute probability
        df1['prob'] = df1['count'] / sum(df1['count'])
        # df21['prob'] = df21['count']/sum(df1['count'])
        # df2['prob'] = df2['count']/sum(df1['count'])

        # to dictionary
        self.dict1 = df1.set_index('term').to_dict()['prob']
        self.dict21 = df21.set_index('term1').to_dict()['count']
        self.dict2 = df2.set_index('term').to_dict()['count']

    def get_uni_prob(self, string):
        if string in self.dict1.keys():
            return self.dict1[string]
        else:
            return 0

    def get_bi_count(self, string):
        if string in self.dict2.keys():
            return self.dict2[string]
        else:
            return 0

    def get_bi_uni_count(self, string):
        if string in self.dict21.keys():
            return self.dict21[string]
        else:
            return 0

    def strip_string(self, string, loc):
        return string[:loc], string[loc:]

    def seg_word(self, string):
        segmentation, frequency = self.dp_seg_word(string, {})
        return ' '.join(segmentation), frequency

    def dp_seg_word(self, string, temp_result):
        
        if string in temp_result:
            return temp_result[string]

        init_prob = self.get_uni_prob(string)
        if len(string) <= 1:
            temp_result[string] = [string], init_prob
            return temp_result[string]

        best_seg = [string]
        best_score = init_prob

        for i in range(1, len(string)):
            a, b = self.strip_string(string, i)
            seg_a, score_a = self.dp_seg_word(a, temp_result)
            seg_b, score_b = self.dp_seg_word(b, temp_result)
            new_bigram = tuple([seg_a[-1], seg_b[0]])
            prior = max(self.get_bi_uni_count(seg_a[-1]), 1)
            new_bigram_prob = max(1, self.get_bi_count(new_bigram)) / prior
            score = score_a * score_b * new_bigram_prob
            if score > best_score:
                best_score = score
                best_seg = seg_a + seg_b

        temp_result[string] = best_seg, best_score
        return temp_result[string]

    def decode(self, hashtag: str, **kwargs) -> List[str]:
        """
        :param hashtag: the input hashtag starting with `#` (e.g., '#helloworld').
        :param kwargs:
        :return: the list of tokens segmented from the hashtag (e.g., ['hello', 'world']).
        """
        # TODO: update the following code.

        tag = hashtag[1:].lower()
        segmentation, _ = self.dp_seg_word(tag, {})

        return segmentation

    def evaluate(self, data: Any, **kwargs):
        pass  # NO NEED TO UPDATE

    def load(self, model_path: str, **kwargs):
        pass  # NO NEED TO UPDATE

    def save(self, model_path: str, **kwargs):
        pass  # NO NEED TO UPDATE

    def train(self, trn_data, dev_data, *args, **kwargs):
        pass  # NO NEED TO UPDATE


if __name__ == '__main__':
    resource_dir = os.environ.get('RESOURCE')
    segmenter = HashtagSegmenter(resource_dir)
    total = correct = 0

    with open(os.path.join(resource_dir, 'hashtags.csv')) as fin:
        reader = csv.reader(fin)
        for row in reader:
            hashtag = row[0]
            gold = row[1]
            # auto = ' '.join(segmenter.decode("#helloworld"))
            auto = ' '.join(segmenter.decode(hashtag))

            print('%s -> %s | %s' % (hashtag, auto, gold))
            if gold == auto:
                correct += 1
            total += 1

    print('%5.2f (%d/%d)' % (100.0 * correct / total, correct, total))
