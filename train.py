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
import torch.nn.functional as F

from matplotlib import pyplot


class SmallDeepNet(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_size, nb_classes):
        super().__init__()

        self.hidden_size = hidden_size
        self.nb_classes = nb_classes
        self.embedding_dim = embedding_dim

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(self.hidden_size, self.nb_classes),
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.5,
                                         momentum=0.9)

    def forward(self, inputs):
        logits = self.classifier(inputs)
        outputs = F.softmax(logits, dim=-1)

        return outputs

    def backward(self, y_pred, target):
        loss = self.criterion(y_pred, target)

        # Zero the gradients
        self.optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        self.optimizer.step()
        return loss

class Data:
    def __init__(self, mode, labels_mode, embedding_file, data_file):
        self.metadata = dict()
        self.mode = mode
        self.labels_mode = labels_mode
        self.embedding_file = embedding_file
        self.data_file = data_file

    def load_data_from_embeddings(self):
        samples_train, labels_train, samples_test, labels_test = \
            self.create_training_data()

        self.metadata['mode'] = self.mode
        self.metadata['labels_mode'] = self.labels_mode

        print(f'Data train: {len(samples_train)}')
        print(f'Labels train: {Counter(labels_train)}')

        embeddings = self.load_embeddings()
        print(f'Word embeddings: {len(embeddings.vocabulary.lst_words)}')

        label_encoder = LabelEncoder()
        label_encoder.fit(labels_train)
        print(f'Labels: {label_encoder.classes_}')
        self.metadata['labels'] = label_encoder.classes_

        X_train = self.create_data_matrix_embeddings(samples_train,
                                                       embeddings)
        y_train = label_encoder.transform(labels_train)
        print(f'Train data: {X_train.shape}, {y_train.shape}')

        X_test = self.create_data_matrix_embeddings(samples_test, embeddings)
        y_test = label_encoder.transform(labels_test)
        print(f'Train data: {X_test.shape}, {y_test.shape}')
        return X_train, y_train, X_test, y_test, label_encoder


    def load_embeddings(self):
        try:
            embeddings = vecto.embeddings.load_from_dir(self.embedding_file)

        except EOFError:
            print(f'Cannot load: {filename}')
            embeddings = None
        return embeddings


    def load_data(self, data_file):
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

    def create_data_matrix_embeddings(self, samples, word_embeddings):
        embeddings_dim = len(word_embeddings.matrix[0])
        nb_samples = len(samples)
        X = np.zeros((nb_samples, embeddings_dim), dtype=np.float32)

        nb_empty = 0
        for i, sample in enumerate(samples):
            tokens = sample.split(' ')
            tokens_embeddings = [word_embeddings.get_vector(t) for t in tokens
                                 if
                                 word_embeddings.has_word(t)]
            if len(tokens_embeddings) > 0:
                mean_embeddings = np.mean(tokens_embeddings, axis=0)
                X[i] = mean_embeddings
            else:
                nb_empty += 1

        print(f'Empty samples: {nb_empty}')

        self.metadata['Empty samples'] = nb_empty

        return X

    def create_training_data(self):
        data_base_filename = self.data_file.joinpath(
            'RuSentiment/rusentiment_random_posts.csv')
        data_posneg_filename = self.data_file.joinpath(
            'RuSentiment/rusentiment_preselected_posts.csv')
        data_test_filename = self.data_file.joinpath(
            'RuSentiment/rusentiment_test.csv')

        samples_base_train, labels_base_train = self.load_data(
            data_base_filename)
        samples_posneg_train, labels_posneg_train = self.load_data(
            data_posneg_filename)
        samples_test, labels_test = self.load_data(data_test_filename)

        print(
            f'Data base: {len(samples_base_train)}, '
            f'{len(labels_base_train)}')
        print(f'Data posneg: {len(samples_posneg_train)},'
              f' {len(labels_posneg_train)}')
        print(f'Data test: {len(samples_test)}, {len(labels_test)}')
        print(f'Labels: {len(set(labels_base_train))},'
              f' {len(set(labels_base_train))}, {len(set(labels_test))}')

        if self.mode == 'base':
            samples_train = samples_base_train
            labels_train = labels_base_train
        elif self.mode == 'posneg':
            samples_train = samples_base_train + samples_posneg_train
            labels_train = labels_base_train + labels_posneg_train
        elif self.mode == 'pos':
            target_class = 'positive'
            target_samples = \
                [s for s, l in
                 zip(samples_posneg_train, labels_posneg_train)
                 if l == target_class]
            target_labels = [target_class] * len(target_samples)
            samples_train = samples_base_train + target_samples
            labels_train = labels_base_train + target_labels
        elif self.mode == 'neg':
            target_class = 'negative'
            target_samples = \
                [s for s, l in
                 zip(samples_posneg_train, labels_posneg_train)
                 if l == target_class]
            target_labels = [target_class] * len(target_samples)
            samples_train = samples_base_train + target_samples
            labels_train = labels_base_train + target_labels
        elif self.mode == 'neutral':
            target_class = 'neutral'
            target_samples = \
                [s for s, l in
                 zip(samples_posneg_train, labels_posneg_train)
                 if l == target_class]
            target_labels = [target_class] * len(target_samples)
            samples_train = samples_base_train + target_samples
            labels_train = labels_base_train + target_labels
        elif self.mode == 'posneg_only':
            samples_train = samples_posneg_train
            labels_train = labels_posneg_train
        elif self.mode == 'replace':
            nb_replace = len(samples_posneg_train)
            samples_base_train, labels_base_train = \
                shuffle(samples_base_train, labels_base_train)
            samples_train = samples_base_train[
                            :-nb_replace] + samples_posneg_train
            labels_train = labels_base_train[
                           :-nb_replace] + labels_posneg_train
        elif self.mode == 'debug':
            nb_samples_debug = 2000
            samples_train = samples_base_train[:nb_samples_debug]
            labels_train = labels_base_train[:nb_samples_debug]
        elif self.mode == 'sample':
            nb_sample = len(samples_posneg_train)
            samples_base_train, labels_base_train = shuffle(
                samples_base_train, labels_base_train)
            samples_train = samples_base_train[:nb_sample]
            labels_train = labels_base_train[:nb_sample]
        elif self.mode == 'sample_posneg':
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

        if self.labels_mode == 'base':
            pass
        elif self.labels_mode == 'neg':
            labels_train = ['rest' if lbl != 'negative' else lbl for lbl in
                            labels_train]
            labels_test = ['rest' if lbl != 'negative' else lbl for lbl in
                           labels_test]
        elif self.labels_mode == 'pos':
            labels_train = ['rest' if lbl != 'positive' else lbl for lbl in
                            labels_train]
            labels_test = ['rest' if lbl != 'positive' else lbl for lbl in
                           labels_test]
        else:
            raise ValueError(f'Labels mode {labels_mode} is unknown')

        return samples_train, labels_train, samples_test, labels_test


def plot(total_loss):
    pyplot.plot(total_loss)
    pyplot.title('loss')
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.show()


def score_model(model, X, y_true, labels):
    _, y_pred = torch.max(model.forward(X), 1)

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
    # Directory with data files (data_base.csv, data_posneg.csv, etc)
    DATA_DIR = Path('./')

    # Path to a folder that has embedding file within
    WORD_VECTORS_FILENAME = '/home/mattd/projects/tmp/sentiment/fasttext/'

    data = Data('base', 'base', WORD_VECTORS_FILENAME, DATA_DIR)

    X_train, y_train, X_test, y_test, label_encoder = \
        data.load_data_from_embeddings()

    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.fit(X_test)

    X_train = torch.tensor(scaler.transform(X_train), requires_grad = False)
    y_train = torch.tensor(y_train, requires_grad = False)
    X_test = torch.tensor(scaler.transform(X_test), requires_grad=False)
    y_test = torch.tensor(y_test, requires_grad=False)

    classifier = SmallDeepNet(len(X_train[0,:]), 100, 5)

    epoch = range(1, 500)
    total_loss = list()
    for i in epoch:
        output = classifier(X_train)

        loss = classifier.backward(output, y_train)

        total_loss.append(float(loss.item()))

        # print('epoch: ', epoch, ' loss: ', loss.item()))
        if str(i)[-1] == str(0):
            print('[%d] loss: %.3f' %
                (i, loss.item()))
    plot(total_loss)

    # net = NeuralNetClassifier(
    #     SmallDeepNet(
    #         embedding_dim=X_train.shape[1],
    #         hidden_size=100,
    #         nb_classes=int(max(y_train)) + 1
    #     ),
    #     max_epochs=50,
    #     lr=0.01,
    #     device='cuda',
    #     verbose=1,
    # )
    #
    # models = [
    #     LogisticRegression(),
    #     # LinearSVC(),
    #     # GradientBoostingClassifier(),
    #     net,
    # ]
    # results = []
    # for model in models:
    #     model.fit(X_train, y_train)  # , sample_weight=sample_weight
    #

    result = [score_model(classifier, X_test, y_test, label_encoder.classes_)]
    print(result)


if __name__ == '__main__':
    main()
