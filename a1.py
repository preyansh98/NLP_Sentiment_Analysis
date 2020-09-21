import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import string
import re

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

def preprocess(x_train, y_train, N_train):
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

    # min_df=2 -> ignore words that appear in less than 2 samples
    cv = CountVectorizer(min_df = 2, preprocessor=lemmatizer_preprocessor)
    x_traincv = cv.fit_transform(x_train)
    
    feature_vector = x_traincv.toarray()
    print(feature_vector.shape)

    return feature_vector
    

def __init__():
    # first read training and test data
    training_data, test_data = read_data()

    x_train, y_train = training_data
    x_test, y_test = test_data

    N_train, N_test = len(x_train), len(x_test)

    # preprocess training data
    feature_vector = preprocess(x_train, y_train, N_train)

__init__()