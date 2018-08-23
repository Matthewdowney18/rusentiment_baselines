import csv
import re
from collections import defaultdict, Counter
import functools
from pathlib import Path
import vecto.embeddings

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

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = torch.nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = torch.nn.Linear(200, output_size)
        self.i2i = torch.nn.Linear(hidden_size,
                                   200)
        self.elu = torch.nn.ELU()
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2i(hidden)
        output = self.elu(output)
        output = self.i2o(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class Data:
    def __init__(self):
        self.metadata = dict()
        self.scaler = StandardScaler()

    def load_data_from_dataset(self, mode, labels_mode, embedding_file,
                                  data_file):
        samples_train, labels_train, samples_test, labels_test, \
        metadata = create_data(data_file, mode, labels_mode)
        self.metadata['dataset'] = metadata

        self.metadata['mode'] = mode
        self.metadata['labels_mode'] = labels_mode

        print(f'Data train: {len(samples_train)}')
        print(f'Labels train: {Counter(labels_train)}')

        embeddings = load_embeddings(embedding_file)
        print(f'Word embeddings: {len(embeddings.vocabulary.lst_words)}')
        self.metadata['embedding'] = embeddings.metadata

        label_encoder = LabelEncoder()
        label_encoder.fit(labels_train)
        print(f'Labels: {label_encoder.classes_}')
        self.metadata['labels'] = label_encoder.classes_


        train_examples = self.create_data_matrix_embeddings(
            samples_train, labels_train, embeddings)
        train_examples = self.add_target(train_examples, label_encoder)
        #print(f'Train data: {len(self.X_train)}, {self.y_train.shape}')
        #self.metadata['Train data'] = [len(self.X_train), self.y_train.shape]

        test_examples = self.create_data_matrix_embeddings(
            samples_test, labels_test, embeddings)
        test_examples = self.add_target(test_examples, label_encoder)
        #print(f'Test data: {len(self.X_test)}, {self.y_test.shape}')
        #self.metadata['Test data'] = [len(self.X_test), self.y_test.shape]

        return train_examples, test_examples

    def create_data_matrix_embeddings(self, samples, labels, word_embeddings):
        embeddings_dim = len(word_embeddings.matrix[0])
        nb_samples = len(samples)
        examples = dict()

        nb_empty = int(0)

        example_num = 0
        for i, sample in enumerate(samples):
            sequence = torch.zeros((100, 1, embeddings_dim), dtype=torch.float)
            tokens = sample.split(' ')
            tokens_embeddings = [word_embeddings.get_vector(t) for t in tokens
                                 if
                                 word_embeddings.has_word(t)][:100]
            if len(tokens_embeddings) > 0:
                for j in range(0, len(tokens_embeddings)):
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
        nb_samples_debug = 2000
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
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.show()


def score_model(classifier, examples, data):
    y_pred = torch.zeros(len(examples))
    y_true = torch.zeros(len(examples))
    for i in examples:
        hidden = classifier.initHidden()
        sequence = examples[i]['sequence']
        outputs = []
        for j in range(sequence.size()[0]):
            line = torch.FloatTensor(data.scaler.transform(sequence[j]))
            output, hidden = classifier(line, hidden)
            outputs.append(output)

            #output = sum(outputs) / len(outputs)

        _, y_pred[i] = torch.max(output, 1)
        y_true[i] = examples[i]['target']
        print(f'example: {i} predicted: {y_pred[i]} actual:{y_true[i]}')
        #print(labels[])
    labels = data.metadata['labels']
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



def main():
    results = dict
    # Directory with data files (data_base.csv, data_posneg.csv, etc)
    data_file = Path('./')

    # Path to a folder that has embedding file within
    embedding_file = '/home/mattd/projects/tmp/sentiment/fasttext/'

    data = Data()
    train_examples, test_examples = data.load_data_from_dataset('debug',
                                                              'base', embedding_file,
                                  data_file)

    #scaler = StandardScaler()
    #scaler.fit(data.X_train)
    #scaler.fit(data.X_test)

    classifier = RNN(300, 500, 5)
    #criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01,
                                weight_decay=3)
    #learning_rate = .01
    total_loss = []
    for epoch in range(0,1):
        for i in train_examples:
            hidden = classifier.initHidden()

            optimizer.zero_grad()

            sequence = train_examples[i]['sequence']
            outputs = []
            for j in range(sequence.size()[0]):
                if sequence[j][0][0] == 0 and sequence[j][0][1] == 0:
                    break
                line = torch.FloatTensor(data.scaler.transform(sequence[j]))
                output, hidden = classifier(line, hidden)
                outputs.append(output)

            #output = sum(outputs)/len(outputs)
            target = train_examples[i]['target']
            loss = criterion(output, target)
            total_loss.append(loss.item())

            average_loss = sum(total_loss)/len(total_loss)
            print(i, loss.item(), average_loss, target, output)
            if i % 99 == 1:
                total_loss = []
            loss.backward()

            # Add parameters' gradients to their values, multiplied by learning rate
            #for p in classifier.parameters():
            #    p.data.add_(-learning_rate, p.grad.data)
            optimizer.step()
        plot(total_loss)
        result = score_model(classifier, test_examples,
                             data)
        #results['result'] = result
        #results['metadata'] = data.metadata
        print(result)


if __name__ == '__main__':
    main()
