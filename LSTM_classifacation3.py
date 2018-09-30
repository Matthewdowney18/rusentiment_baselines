import csv
import os
import json
import pandas as pd
from collections import defaultdict, Counter
import functools
from pathlib import Path
import vecto.embeddings
import random
import time

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from nltk import word_tokenize, TweetTokenizer

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence

from matplotlib import pyplot

class RNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, weight_decay,
                 lr, weights):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.drop = torch.nn.Dropout(0.5)

        self.GRU = torch.nn.GRU(input_size, hidden_size, bidirectional=True,
                                dropout=.5)
        self.h2o = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*2, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size))

        self.criterion = torch.nn.NLLLoss(weight=weights)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay)

        self.metadata = dict()
        self.metadata['input_size'] = input_size
        self.metadata['hidden_size'] = hidden_size
        self.metadata['output_size'] = output_size
        self.metadata['weight_decay'] = weight_decay
        self.metadata['lr'] = lr
        # self.metadata['description'] = get_description()


    def forward(self, input, hidden, state):
        output, hidden = self.GRU(input, hidden)
        hidden = output.view(self.hidden_size*2, -1)
        hidden = hidden[:, -1]
        self.drop(hidden)
        output = self.h2o(hidden)
        output = F.log_softmax(output, dim=-1)
        return output.view(1, 5), hidden, state

    def init_hidden(self, sequence_length):
        return torch.zeros(2, sequence_length, self.hidden_size)

    def backward(self, output, target):
        self.optimizer.zero_grad()
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        # Add parameters' gradients to their values, multiplied by learning rate
        #for p in self.parameters():
        #    p.data.add_(-.0009, p.grad.data)
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

    def get_weight(self):
        weight = torch.Tensor([7.75, 2.44, 4.6, 6.89, 9.25])
        return weight


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
        nb_samples_debug = 1000
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

def plot_loss(total_loss, cv):
    pyplot.figure(0)
    pyplot.plot(total_loss, 'r-', cv, 'b-')
    pyplot.title('loss')
    pyplot.xlabel('sample')
    pyplot.ylabel('loss')
    pyplot.show()

def plot_accuracy(accuracy):
    df = pd.DataFrame(accuracy)
    pyplot.figure(1)
    pyplot.plot(df)
    pyplot.title('accuracy')
    pyplot.xlabel('')
    pyplot.ylabel('accuracy')

def plot_confusion(confusion_matrix):
    pyplot.imshow(confusion_matrix)
    pyplot.title('confusion')
    pyplot.xlabel('sample')
    pyplot.ylabel('loss')
    pyplot.show()

def get_accuracy(targets, outputs):
    results = dict()
    outputs = torch.max(outputs, 1)
    for i in range(0, 5):
        class_outputs = torch.zeros(targets.size())
        class_targets = torch.zeros(targets.size())
        for j, output in enumerate(outputs[1]):
            if output == i:
                class_outputs[j] = 1
            if targets[j] == i:
                class_targets[j] = 1
        results['class ' + str(i)] = accuracy_score(class_targets,
                                                    class_outputs)
    return results

def forward_pass(classifier, sequence):
    packed_sequence = pack_sequence(sequence)

    hidden = classifier.init_hidden(len(sequence))
    state = classifier.init_hidden(2)

    output, hidden, state = classifier(packed_sequence, hidden, state)
    return classifier, output

def cross_validation(classifier, examples, size):
    keys = list(examples.keys())
    random.shuffle(keys)
    total_loss = list()
    for i, key in enumerate(keys[:size]):
        sequence = examples[key]['sequence']
        output = forward_pass(classifier, sequence)[1]
        y_true = examples[key]['target']
        loss = classifier.criterion(output, y_true)
        total_loss.append(loss.item())
    return (sum(total_loss)/len(total_loss))


def score_model(classifier, examples, data):
    y_pred = torch.zeros(len(examples))
    y_true = torch.zeros(len(examples))
    for i in examples:
        sequence = examples[i]['sequence']
        _, output = forward_pass(classifier, sequence)

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

    confusion = confusion_matrix(y_true, y_pred)
    plot_confusion(confusion)

    results = dict()
    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["f1"] = f1_score(y_true, y_pred, average=average,
                             pos_label=pos_label)
    results["precision"] = precision_score(y_true, y_pred, average=average,
                                 pos_label=pos_label)
    results["recall"] = recall_score(y_true, y_pred, average=average,
                            pos_label=pos_label)
    return results


def train(classifier, train_examples, test_examples, num_iterations,
          test_interval=100):
    interval_loss = [0] * test_interval
    interval_outputs = torch.zeros(test_interval, 5)
    interval_targets = torch.zeros(test_interval, 1)
    total_loss = list()
    cv_loss = list()
    total_accuracy = list()
    keys = list(train_examples.keys())
    for epoch in range(0, num_iterations):
        random.shuffle(keys)
        for i, key in enumerate(keys):
            sequence = train_examples[key]['sequence']
            start = time.clock()

            classifier, output = forward_pass(classifier, sequence)
            target = train_examples[key]['target']
            loss = classifier.backward(output, target)
            end = time.clock()

            t = (end - start)

            k = i % test_interval
            interval_loss[k] = loss.item()
            interval_outputs[k] = output
            interval_targets[k] = target
            average_loss = sum(interval_loss) / len(interval_loss)

            if k == 0 and i > test_interval - 1:

                validation_loss = cross_validation(
                    classifier, test_examples, 50)
                cv_loss.append(validation_loss)

                print(epoch, i, t, average_loss, validation_loss, '|',
                      len(sequence), loss.item(), target, output)
                accuracy = get_accuracy(interval_targets, interval_outputs)
                print(accuracy)
                total_accuracy.append(accuracy)

                total_loss.append(average_loss)

    plot_accuracy(total_accuracy)
    plot_loss(total_loss, cv_loss)
    return classifier

def get_description():
    file = open()

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
    #embedding_file = '/home/mattd/pycharm/sentiment/fasttext/'

    data = Data()
    train_examples, test_examples = data.load_data_from_dataset(
        'posneg', 'base', embedding_file, data_file)

    weights = data.get_weight()
    classifier = RNN(300, 600, 5, 0.005, .0005, weights)

    epoch = 3

    classifier = train(classifier, train_examples, test_examples, epoch, 50)

    result = score_model(classifier, test_examples, data)

    results['result'] = result
    results['data'] = data.metadata
    results['classifier'] = classifier.metadata
    results['classifier']['epoch'] = epoch
    print(results)

    save_json(results, 'result38')


if __name__ == '__main__':
    main()