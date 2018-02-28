import re
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from scipy import io
from sklearn import svm

def get_vocablist():
    vocabulary = []
    with open('vocab.txt') as f:
        for line in f:
            idx, word = line.split('\t')
            vocabulary.append(word.strip())
    return vocabulary

def split(delimiters, string, maxsplit=0):
    pattern = '|'.join(map(re.escape, delimiters))
    return re.split(pattern, string, maxsplit)

def process_email(email_contents):
    vocab_list = get_vocablist()
    email_contents = email_contents.lower()
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    words = split(""" @$/#.-:&*+=[]?!(){},'">_<;%\n\r""", email_contents)
    word_indices = []
    stemmer = SnowballStemmer("english")
    for word in words:
        word = re.sub('[^a-zA-Z0-9]', '', word)
        if word == '':
            continue
        word = stemmer.stem(word)
        if word in vocab_list:
            idx = vocab_list.index(word)
            word_indices.append(idx)
    return word_indices

def email_features(word_indices):
    n = 1899
    x = np.zeros((n, 1))
    x[word_indices] = 1
    return x

if __name__ == "__main__":

    # ==================== Part 1: Email Preprocessing ====================
    print ('Preprocessing sample email (emailSample1.txt)...')
    file = open('emailSample1.txt', 'r')
    file_contents = file.read().replace('\n', '')

    word_indices = process_email(file_contents)
    print ('Word Indices:', word_indices)

    # ==================== Part 2: Feature Extraction ====================
    print ('Extracting features from sample email (emailSample1.txt)...')
    features = email_features(word_indices)
    print ('Length of feature vector:', len(features))
    print ('Number of non-zero entries:', np.sum(features > 0))

    # =========== Part 3: Train Linear SVM for Spam Classification ========
    # Load the Spam Email dataset
    train_data = io.loadmat('spamTrain.mat')
    X_train = train_data['X']
    y_train = train_data['y'].ravel()

    print ('Training Linear SVM (Spam Classification)...')
    C = 1
    clf = svm.LinearSVC(C=C)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    print ('Training Accuracy:', np.mean(y_pred_train == y_train) * 100)

    # =================== Part 4: Test Spam Classification ================
    # Load the test dataset
    test_data = io.loadmat('spamTest.mat')
    X_test = test_data['Xtest']
    y_test = test_data['ytest'].ravel()
    print ('Evaluating the trained Linear SVM on a test set...')
    y_pred_test = clf.predict(X_test)
    print ('Test Accuracy:', np.mean(y_pred_test == y_test) * 100)

    # ================= Part 5: Top Predictors of Spam ====================
    coef = clf.coef_.ravel()
    idx = coef.argsort()[::-1]
    vocab_list = get_vocablist()

    print ('Top predictors of spam:')
    for i in range(15):
        print ("{0:<15s} ({1:f})".format(vocab_list[idx[i]], coef[idx[i]]))

    # =================== Part 6: Try Your Own Emails =====================
    file_name = open('spamSample2.txt', 'r')
    file_contents = file_name.read().replace('\n', '')

    word_indices = process_email(file_contents)
    x = email_features(word_indices)
    p = clf.predict(x.T)
    print ('Processed', file_name, '\nSpam Classification:', p)
    print ('(1 indicates spam, 0 indicates not spam)')
