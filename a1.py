import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn import svm
import string
import re
import random

DATA_ENCODING = "ISO-8859-1"
TRAIN_TEST_SPLIT = 0.75 #define ratio of dataset used for training vs testing

def read_data():
    pos_lines = []
    neg_lines = []

    with open('./rt-polaritydata/rt-polarity.pos', 'r', encoding= DATA_ENCODING) as file_reader:
        pos_lines = list(map(lambda x: x.replace("\n", "").strip(), file_reader.readlines()))

    with open('./rt-polaritydata/rt-polarity.neg', 'r', encoding = DATA_ENCODING) as file_reader:
        neg_lines = list(map(lambda x: x.replace("\n", "").strip(), file_reader.readlines()))

    sentences = [[line,1] for line in pos_lines] + [[line,0] for line in neg_lines]

    N = len(sentences)
    inds = np.random.permutation(N)

    N_train = int(TRAIN_TEST_SPLIT * N)

    x_train, y_train = [sentences[i][0] for i in inds[:N_train]], [sentences[i][1] for i in inds[:N_train]]
    x_test, y_test = [sentences[i][0] for i in inds[N_train:]], [sentences[i][1] for i in inds[N_train:]]

    return ([x_train, y_train], [x_test, y_test])

### Preprocess to feature vectors

stopwords_set = set(stopwords.words('english'))

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def common_preprocessing(text):
    # lowercase and remove special chars
    text = text.lower()
    text = re.sub("\\W", " ", text)

    # tokenize text
    words = word_tokenize(text)

    # filter out stopwords and punctuation
    filtered_words = [w for w in words if not w in stopwords_set and w not in string.punctuation]

    return filtered_words

# define a preprocessor with stemmer
def stemmer_preprocessor(text):
    filtered_words = common_preprocessing(text)

    # stem words
    stemmed_words = [stemmer.stem(word = word) for word in filtered_words]
    return ' '.join(stemmed_words)

# define a preprocessor with lemmatizer
def lemmatizer_preprocessor(text):
    filtered_words = common_preprocessing(text)

    # lemmatize words
    lemmatized_words = [lemmatizer.lemmatize(word = word) for word in filtered_words]
    return ' '.join(lemmatized_words)

def preprocess_train(x_train, y_train, N_train):
    # min_df=2 -> ignore words that appear in less than 2 samples
    cv = CountVectorizer(min_df = 2, preprocessor=lemmatizer_preprocessor)
    x_traincv = cv.fit_transform(x_train)
    
    feature_vector = x_traincv.toarray()
    feature_vector_words = cv.get_feature_names()

    return feature_vector, feature_vector_words, cv

def preprocess_test(x_test, cv):
    x_testcv = cv.transform(x_test)
    feature_vector = x_testcv.toarray()
    
    return feature_vector
    
##### CLASSIFIERS #######

# Naive-Bayes Classifier
class NB_Classifier:

    def fit(self, x, y, feature_names):
        dataset = []

        for (train_no, feature_vec) in enumerate(x):
            featureset = dict()
            label = y[train_no]

            for i in range(len(feature_vec)):
                featureset[feature_names[i]] = x[train_no][i]

            dataset.append((featureset, label))

        self.dataset = dataset
        self.nbc = NaiveBayesClassifier.train(dataset)

        return self

    def predict(self, x_test, feature_names):
        dataset_pred = []

        # format x_test to featureset
        for (test_no, feature_vec) in enumerate(x_test):
            featureset = dict()

            for i in range(len(feature_vec)):
                featureset[feature_names[i]] = x_test[test_no][i]

            dataset_pred.append(featureset)
        
        result = self.nbc.classify_many(dataset_pred)
        return result

# Logistic Regression Classifier
class LR_Classifier:

    def __init__(self):
        model = LogisticRegression(random_state=0, C=2)
        self.model = model
        return

    def fit(self, x, y):
        self.x = x
        self.y = y

        self.model.fit(x,y)
        
        return self
    
    def predict(self, x_test):
        return self.model.predict(x_test)

# SVM Classifier
class SVM_Classifier:

    def __init__(self):
        model = svm.LinearSVC()
        self.model = model
        return

    def fit(self, x, y):
        self.x = x
        self.y = y

        self.model.fit(x,y)
        return self

    def predict(self, x_test):
        return self.model.predict(x_test)

def run_nb_classifier(feature_vector_train, y_train, feature_vector_words, feature_vector_test, y_test):
    nbc = NB_Classifier()
    nbc.fit(feature_vector_train, y_train, feature_vector_words)
    predictions = nbc.predict(feature_vector_test, feature_vector_words)

    return calculate_accuracy(predictions, y_test)

def run_lr_classifier(feature_vector_train, y_train, feature_vector_words, feature_vector_test, y_test):
    logreg = LogisticRegression()
    logreg.fit(feature_vector_train, y_train)
    predictions = logreg.predict(feature_vector_test)

    return calculate_accuracy(predictions, y_test)

def run_svm_classifier(feature_vector_train, y_train, feature_vector_words, feature_vector_test, y_test):
    linear_svm = SVM_Classifier()
    linear_svm.fit(feature_vector_train, y_train)
    predictions = linear_svm.predict(feature_vector_test)

    return calculate_accuracy(predictions, y_test)

def run_randombaseline_classifier(feature_vector_test, y_test):

    # choose positive (1) or negative (0) with equal probability
    predictions = [random.randint(0,1) for i in range(len(feature_vector_test))]

    return calculate_accuracy(predictions, y_test)

def run_perceptron_classifier(feature_vector_train, y_train, feature_vector_words, feature_vector_test, y_test):
    perceptron = Perceptron(tol=1e-3,random_state=0)
    perceptron.fit(feature_vector_train, y_train)
    predictions = perceptron.predict(feature_vector_test)

    return calculate_accuracy(predictions, y_test)

def calculate_accuracy(predictions, y_test):
    num_correct = 0
    num_sampled = len(predictions)

    for (test_no, prediction) in enumerate(predictions):
        if (prediction == y_test[test_no]):
            num_correct += 1

    accuracy = num_correct/num_sampled
    return accuracy

def __init__():
    # first read training and test data
    training_data, test_data = read_data()

    x_train, y_train = training_data
    x_test, y_test = test_data

    N_train, N_test = len(x_train), len(x_test)

    # preprocess training + testing data
    feature_vector_train, feature_vector_words, cv = preprocess_train(x_train, y_train, N_train)
    feature_vector_test = preprocess_test(x_test, cv)

    # run nb_classifier
    nb_accuracy = run_nb_classifier(feature_vector_train, y_train, feature_vector_words, feature_vector_test, y_test)
    print("Accuracy for NB is ",nb_accuracy)

    # run logreg classifier
    lgreg_accuracy = run_lr_classifier(feature_vector_train, y_train, feature_vector_words, feature_vector_test, y_test)
    print("Accuracy for LogReg is ", lgreg_accuracy)

    # run linear svm classifier
    linsvm_accuracy = run_svm_classifier(feature_vector_train, y_train, feature_vector_words, feature_vector_test, y_test)
    print("Accuracy for Linear SVM is ", linsvm_accuracy)

    # run random baseline 
    random_baseline = run_randombaseline_classifier(feature_vector_test, y_test)
    print("Accuracy for Random Baseline is ", random_baseline)

    # run perceptron
    perceptron_accuracy = run_perceptron_classifier(feature_vector_train, y_train, feature_vector_words, feature_vector_test, y_test)
    print("Accuracy for Perceptron is ", perceptron_accuracy)

__init__()