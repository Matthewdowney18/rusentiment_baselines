import csv
import os
import json
from collections import defaultdict, Counter
import functools
from pathlib import Path
import vecto.embeddings
import random

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from nltk import word_tokenize, TweetTokenizer

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

from matplotlib import pyplot

class RNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, weight_decay, lr):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.drop = torch.nn.Dropout(0.5)
        self.lstm = torch.nn.LSTMCell(input_size, hidden_size)
        self.hidden_scale = torch.nn.Linear(input_size, hidden_size)
        self.input_scale = torch.nn.Linear(hidden_size, hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.i2h = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size+input_size,
                                self.hidden_size),
                torch.nn.Sigmoid())
        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.normalize = torch.nn.BatchNorm1d(hidden_size)
        self.h2o = torch.nn.Sequential(
        #    torch.nn.Linear(self.hidden_size, self.hidden_size),
        #    torch.nn.ELU(),
        #    torch.nn.Linear(self.hidden_size, self.hidden_size),
        #    torch.nn.ELU(),
        #    torch.nn.Linear(self.hidden_size, self.hidden_size),
        #    torch.nn.ELU(),
            torch.nn.Linear(self.hidden_size, output_size))

        self.criterion = torch.nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr,
                                    weight_decay=weight_decay)
        self.metadata = dict()
        self.metadata['input_size'] = input_size
        self.metadata['hidden_size'] = hidden_size
        self.metadata['output_size'] = output_size
        self.metadata['weight_decay'] = weight_decay
        self.metadata['lr'] = lr
        self.metadata['description'] = 'a rnn that has a dropout layer, ' \
                                       'calculates the hidden ' \
                                       'state with with a linear, elu layer. ' \
                                       'the ' \
                                       'new hidden state is put through a nn (' \
                                       'linear softmax ) for ' \
                                       'the output. loss:NLLL loss, ' \
                                       'optimizer = SGD, 5 epoch'

    def forward(self, input, hidden, state):
        for element in input:
            element = self.drop(element)
            element = self.input_scale(element)
            hidden = self.hidden_scale(hidden)
            hidden = self.sigmoid(hidden + element)
            #importance = F.softmax(self.importance(element), dim = -1)
            #combined = torch.cat((element, hidden), 1)
            #hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = F.log_softmax(output, dim=-1)
        return output, hidden, state

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def backward(self, output, target):
        self.optimizer.zero_grad()
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        # Add parameters' gradients to their values, multiplied by learning rate
        # for p in classifier.parameters():
        #    p.data.add_(-learning_rate, p.grad.data)
        return loss

class Data:
    def __init__(self):
        self.metadata = dict()
        self.scaler = StandardScaler()

    def load_data_from_dataset(self, mode, labels_mode, embedding_file,
                                  data_file):
        samples_train, labels_train, samples_test, labels_test, \
        metadata = create_data(data_file, mode, labels_mode)
        self.metadata['dataset'] = metadata

        print(f'Data train: {len(samples_train)}')
        print(f'Labels train: {Counter(labels_train)}')
        self.metadata['data train'] = len(samples_train)
        self.metadata['labels train'] = Counter(labels_train)

        embeddings = load_embeddings(embedding_file)
        embeddings.normalize()
        print(f'Word embeddings: {len(embeddings.vocabulary.lst_words)}')
        self.metadata['embedding'] = embeddings.metadata

        label_encoder = LabelEncoder()
        label_encoder.fit(labels_train)
        print(f'Labels: {label_encoder.classes_}')
        self.labels = label_encoder.classes_

        train_examples = self.create_data_dictionary(
            samples_train, labels_train, embeddings)
        train_examples = self.add_target(train_examples, label_encoder)
        #print(f'Train data: {len(self.X_train)}, {self.y_train.shape}')
        self.metadata['Train data'] = len(train_examples)

        test_examples = self.create_data_dictionary(
            samples_test, labels_test, embeddings)
        test_examples = self.add_target(test_examples, label_encoder)
        #print(f'Test data: {len(self.X_test)}, {self.y_test.shape}')
        self.metadata['Test data'] = [len(test_examples)]

        return train_examples, test_examples

    def create_data_dictionary(self, samples, labels, word_embeddings):
        embeddings_dim = len(word_embeddings.matrix[0])
        nb_samples = len(samples)
        examples = dict()

        nb_empty = int(0)

        example_num = 0
        for i, sample in enumerate(samples):
            tokens = sample.split(' ')
            tokens_embeddings = [word_embeddings.get_vector(t) for t in tokens
                                 if
                                 word_embeddings.has_word(t)]
            length = len(tokens_embeddings)

            if length > 0:
                sequence = torch.zeros((length, 1, embeddings_dim),
                                       dtype=torch.float)
                for j in range(0, length):
                    line = torch.from_numpy(tokens_embeddings[j])
                    sequence[j][0] = line
                    if line[0] != 0:
                        self.scaler.fit(line.view(-1,1))

                examples[example_num] = {"text":sample, "label":labels[i],
                               "sequence":sequence}
                example_num +=1
            if len(tokens_embeddings) == 0:
                nb_empty += 1

        print(f'Empty samples: {nb_empty}')

        self.metadata['Empty samples'] = nb_empty

        return examples

    def add_target(self, examples, label_encoder):
        for i in examples:
            label = [str(examples[i]['label'])]
            target = label_encoder.transform(label)
            examples[i]['target'] = torch.tensor(target)
        return examples

def load_embeddings(embedding_file):
    try:
        embeddings = vecto.embeddings.load_from_dir(embedding_file)

    except EOFError:
        print(f'Cannot load: {filename}')
        embeddings = None
    return embeddings


def create_data(data_file, mode, labels_mode):
    metadata = dict()
    data_base_filename = data_file.joinpath(
        'RuSentiment/rusentiment_random_posts.csv')
    data_posneg_filename = data_file.joinpath(
        'RuSentiment/rusentiment_preselected_posts.csv')
    data_test_filename = data_file.joinpath(
        'RuSentiment/rusentiment_test.csv')

    samples_base_train, labels_base_train = load_data(
        data_base_filename)
    samples_posneg_train, labels_posneg_train = load_data(
        data_posneg_filename)
    samples_test, labels_test = load_data(data_test_filename)

    print(
        f'Data base: {len(samples_base_train)}, '
        f'{len(labels_base_train)}')
    print(f'Data posneg: {len(samples_posneg_train)},'
          f' {len(labels_posneg_train)}')
    print(f'Data test: {len(samples_test)}, {len(labels_test)}')
    print(f'Labels: {len(set(labels_base_train))},'
          f' {len(set(labels_base_train))}, {len(set(labels_test))}')

    metadata['mode'] = mode
    metadata['labels_mode'] = labels_mode
    metadata['Data base'] =  [len(samples_base_train),
        len(labels_base_train)]
    metadata['Data posneg'] = [len(samples_posneg_train),
        len(labels_posneg_train)]
    metadata['Labels'] = [len(set(labels_base_train)),
        len(set(labels_base_train)), len(set(labels_test))]

    if mode == 'base':
        samples_train = samples_base_train
        labels_train = labels_base_train
    elif mode == 'posneg':
        samples_train = samples_base_train + samples_posneg_train
        labels_train = labels_base_train + labels_posneg_train
    elif mode == 'pos':
        target_class = 'positive'
        target_samples = \
            [s for s, l in
             zip(samples_posneg_train, labels_posneg_train)
             if l == target_class]
        target_labels = [target_class] * len(target_samples)
        samples_train = samples_base_train + target_samples
        labels_train = labels_base_train + target_labels
    elif mode == 'neg':
        target_class = 'negative'
        target_samples = \
            [s for s, l in
             zip(samples_posneg_train, labels_posneg_train)
             if l == target_class]
        target_labels = [target_class] * len(target_samples)
        samples_train = samples_base_train + target_samples
        labels_train = labels_base_train + target_labels
    elif mode == 'neutral':
        target_class = 'neutral'
        target_samples = \
            [s for s, l in
             zip(samples_posneg_train, labels_posneg_train)
             if l == target_class]
        target_labels = [target_class] * len(target_samples)
        samples_train = samples_base_train + target_samples
        labels_train = labels_base_train + target_labels
    elif mode == 'posneg_only':
        samples_train = samples_posneg_train
        labels_train = labels_posneg_train
    elif mode == 'replace':
        nb_replace = len(samples_posneg_train)
        samples_base_train, labels_base_train = \
            shuffle(samples_base_train, labels_base_train)
        samples_train = samples_base_train[
                        :-nb_replace] + samples_posneg_train
        labels_train = labels_base_train[
                       :-nb_replace] + labels_posneg_train
    elif mode == 'debug':
        nb_samples_debug = 100
        samples_train = samples_base_train[:nb_samples_debug]
        labels_train = labels_base_train[:nb_samples_debug]
    elif mode == 'sample':
        nb_sample = len(samples_posneg_train)
        samples_base_train, labels_base_train = shuffle(
            samples_base_train, labels_base_train)
        samples_train = samples_base_train[:nb_sample]
        labels_train = labels_base_train[:nb_sample]
    elif mode == 'sample_posneg':
        nb_samples_by_classes = Counter(labels_posneg_train)

        samples_train = []
        labels_train = []
        for target_class, target_counts in \
                nb_samples_by_classes.most_common():
            base_samples_of_target_class = [
                s for s, l in zip(samples_base_train, labels_base_train)
                if l == target_class]
            shuffle(base_samples_of_target_class)
            base_samples_of_target_class = \
                base_samples_of_target_class[:target_counts]

            samples_train.extend(base_samples_of_target_class)
            labels_train.extend(
                [target_class] * len(base_samples_of_target_class))
    else:
        raise ValueError(f'Mode {mode} is unknown')

    if labels_mode == 'base':
        pass
    elif labels_mode == 'neg':
        labels_train = ['rest' if lbl != 'negative' else lbl for lbl in
                        labels_train]
        labels_test = ['rest' if lbl != 'negative' else lbl for lbl in
                       labels_test]
    elif labels_mode == 'pos':
        labels_train = ['rest' if lbl != 'positive' else lbl for lbl in
                        labels_train]
        labels_test = ['rest' if lbl != 'positive' else lbl for lbl in
                       labels_test]
    else:
        raise ValueError(f'Labels mode {labels_mode} is unknown')

    return samples_train, labels_train, samples_test, labels_test, metadata


def load_data(data_file):
    tokenizer = TweetTokenizer()

    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)

        samples = []
        labels = []
        for row in reader:
            text = row['text']
            label = row['label']

            text_tokenized = tokenizer.tokenize(text)

            text_joined = ' '.join(text_tokenized)

            samples.append(text_joined)
            labels.append(label)
    return samples, labels

def plot(total_loss):
    pyplot.plot(total_loss)
    pyplot.title('loss')
    pyplot.xlabel('sample')
    pyplot.ylabel('loss')
    pyplot.show()


def score_model(classifier, examples, data):
    y_pred = torch.zeros(len(examples))
    y_true = torch.zeros(len(examples))
    for i in examples:
        hidden = classifier.init_hidden()
        state = classifier.init_hidden()
        sequence = examples[i]['sequence']
        input = []
        for element in sequence:
            input.append(torch.FloatTensor(data.scaler.transform(element)))

        output, hidden, state = classifier(input, hidden, state)

            #output = sum(outputs) / len(outputs)

        _, y_pred[i] = torch.max(output, 1)
        y_true[i] = examples[i]['target']
        print(f'example: {i} predicted: {y_pred[i]} actual:{y_true[i]} '
              f'output:{output}')
        #print(labels[])
    labels = data.labels
    if len(set(labels)) == 2:
        average = 'binary'
        pos_label = int(np.argwhere(labels != 'rest'))
    else:
        average = 'weighted'
        pos_label = 1

    results = dict()
    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["f1"] = f1_score(y_true, y_pred, average=average,
                             pos_label=pos_label)
    results["precision"] = precision_score(y_true, y_pred, average=average,
                                 pos_label=pos_label)
    results["recall"] = recall_score(y_true, y_pred, average=average,
                            pos_label=pos_label)
    return results

def train(classifier, train_examples, data, num_iterations):
    last_100_loss = [0] * 100
    total_loss = []
    keys = list(train_examples.keys())
    for epoch in range(0, num_iterations):
        random.shuffle(keys)
        for i, key in enumerate(keys):
            hidden = classifier.init_hidden()
            state = classifier.init_hidden()

            sequence = train_examples[key]['sequence']
            input = list()
            for element in sequence:
                input.append(torch.FloatTensor(data.scaler.transform(element)))

            output, hidden, state = classifier(input, hidden, state)

            # output = sum(outputs)/len(outputs)
            target = train_examples[key]['target']
            loss = classifier.backward(output, target)

            k = i % 100
            last_100_loss[k] = loss.item()
            average_loss = sum(last_100_loss) / len(last_100_loss)
            if k == 0:
                print(epoch, i, average_loss, '|', len(sequence), loss.item(),
                      target, output)
            total_loss.append(average_loss)
    plot(total_loss)
    return classifier

def save_json(results, filename):
    basedir = os.path.dirname(
        '/home/mattd/projects/tmp/sentiment/results/')
    os.makedirs(basedir, exist_ok=True)
    info = json.dumps(results, ensure_ascii=False, indent=4,
                      sort_keys=False)
    file = open('/home/mattd/projects/tmp/sentiment/results/'
                '' + filename + '.json', 'w')
    file.write(info)
    file.close()


def main():
    results = dict()
    # Directory with data files (data_base.csv, data_posneg.csv, etc)
    data_file = Path('./')

    # Path to a folder that has embedding file within
    embedding_file = '/home/mattd/projects/tmp/sentiment/fasttext/'

    data = Data()
    train_examples, test_examples = data.load_data_from_dataset('posneg',
                                                        'base', embedding_file,
                                  data_file)

    #scaler = StandardScaler()
    #scaler.fit(data.X_train)
    #scaler.fit(data.X_test)

    classifier = RNN(300, 300, 5, .2, .001)
    #criterion = torch.nn.NLLLoss()
    #learning_rate = .01

    epoch = 5

    classifier = train(classifier, train_examples, data, epoch)

    result = score_model(classifier, test_examples, data)

    results['result'] = result
    results['data'] = data.metadata
    results['classifier'] = classifier.metadata
    results['classifier']['epoch'] = epoch
    print(results)

    save_json(results, 'result25')


if __name__ == '__main__':
    main()