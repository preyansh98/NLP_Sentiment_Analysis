import nltk
import numpy as np
import sklearn

DATA_ENCODING = "ISO-8859-1"
TRAIN_TEST_SPLIT = 0.75 #define ratio of dataset used for training vs testing

def read_data():
    pos_lines = []
    neg_lines = []

    with open('./rt-polaritydata/rt-polarity.pos', 'r', encoding= DATA_ENCODING) as file_reader:
        pos_lines = list(map(lambda x: x.replace("\n", "").strip(), file_reader.readlines()))

    with open('./rt-polaritydata/rt-polarity.neg', 'r', encoding = DATA_ENCODING) as file_reader:
        neg_lines = list(map(lambda x: x.replace("\n", "").strip(), file_reader.readlines()))

    sentences = []

    for line in pos_lines:
        sentences.append([line, 1])

    for line in neg_lines:
        sentences.append([line, 0])

    N = len(sentences)
    inds = np.random.permutation(N)

    N_train = int(TRAIN_TEST_SPLIT * N)

    x_train, y_train = [sentences[i][0] for i in inds[:N_train]], [sentences[i][1] for i in inds[:N_train]]
    x_test, y_test = [sentences[i][0] for i in inds[N_train:]], [sentences[i][1] for i in inds[N_train:]]

    return ([x_train, y_train], [x_test, y_test])

def preprocess(x_train, y_train, N_train):
    pass

def __init__():
    # first read training and test data
    training_data, test_data = read_data()

    x_train, y_train = training_data
    x_test, y_test = test_data

    N_train, N_test = len(x_train), len(x_test)

    # preprocess training data
    preprocess(x_train, y_train, N_train)

__init__()