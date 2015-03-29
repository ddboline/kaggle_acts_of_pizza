#!/usr/bin/python

import os

import matplotlib
matplotlib.use('Agg')
import pylab as pl
import numpy as np
import pandas as pd
import re

from bs4 import BeautifulSoup

from collections import defaultdict

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

def review_to_wordlist(review, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        from nltk.corpus import stopwords

        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

def clean_review_function(review):
    global master_word_dict, number_of_rows
    list_of_words = review_to_wordlist(review, remove_stopwords=False)
    return ' '.join(list_of_words)


def load_data():
    train_df = pd.read_json('train.json')
    subm_df = pd.read_csv('sampleSubmission.csv')
    test_df = pd.read_json('test.json')

    YCOL = u'requester_received_pizza'

    XCOLS_KEEP = [u'requester_account_age_in_days_at_request', u'requester_days_since_first_post_on_raop_at_request', u'requester_number_of_comments_at_request', u'requester_number_of_comments_in_raop_at_request', u'requester_number_of_posts_at_request', u'requester_number_of_posts_on_raop_at_request', u'requester_number_of_subreddits_at_request', u'requester_upvotes_minus_downvotes_at_request', u'requester_upvotes_plus_downvotes_at_request', u'unix_timestamp_of_request', u'unix_timestamp_of_request_utc']

    XCOLS_TOSS = [u'number_of_downvotes_of_request_at_retrieval', u'number_of_upvotes_of_request_at_retrieval', u'post_was_edited', u'request_number_of_comments_at_retrieval', u'request_text', u'requester_account_age_in_days_at_retrieval', u'requester_days_since_first_post_on_raop_at_retrieval', u'requester_number_of_comments_at_retrieval', u'requester_number_of_comments_in_raop_at_retrieval', u'requester_number_of_posts_at_retrieval', u'requester_number_of_posts_on_raop_at_retrieval', u'requester_upvotes_minus_downvotes_at_retrieval', u'requester_upvotes_plus_downvotes_at_retrieval', u'requester_user_flair', u'request_id', u'giver_username_if_known']

    train_df = train_df.drop(labels=XCOLS_TOSS, axis=1)

    use_text_data = True

    if use_text_data:
        clean_train_review = train_df['request_text_edit_aware'].apply(clean_review_function)
        clean_test_review = test_df['request_text_edit_aware'].apply(clean_review_function)
        clean_train_title = train_df['request_title'].apply(clean_review_function)
        clean_test_title = test_df['request_title'].apply(clean_review_function)

        #for df in train_df, test_df:
            #for c in 'request_text_edit_aware', 'request_title':
                #print c, df[c].shape

        nfeatures=1000
        print 'nfeatures', nfeatures
        vectorizer = CountVectorizer(analyzer='word', tokenizer=None,  preprocessor=None, stop_words=None, max_features=nfeatures)
        train_review_features = vectorizer.fit_transform(clean_train_review).toarray()
        test_review_features = vectorizer.transform(clean_test_review).toarray()
        train_title_features = vectorizer.transform(clean_train_title).toarray()
        test_title_features = vectorizer.transform(clean_test_title).toarray()

        print 'shape0', train_review_features.shape, test_review_features.shape, train_title_features.shape, test_title_features.shape

    train_df = train_df.drop(labels=['request_text_edit_aware', 'request_title'], axis=1)
    test_df = test_df.drop(labels=['request_text_edit_aware', 'request_title'], axis=1)

    for df in train_df, test_df:
        #df['request_text_edit_aware'] = df['request_text_edit_aware'].map(review_to_wordlist).map(len)
        #df['request_title'] = df['request_title'].map(review_to_wordlist).map(len)
        df['requester_subreddits_at_request'] = df['requester_subreddits_at_request'].map(len)
        df['requester_username'] = df['requester_username'].map(len)
        df['requester_account_age_in_days_at_request'] = df['requester_account_age_in_days_at_request'].astype(np.int64)
        df['requester_days_since_first_post_on_raop_at_request'] = df['requester_days_since_first_post_on_raop_at_request'].astype(np.int64)

    ytrain = train_df['requester_received_pizza'].astype(np.int64).values

    train_df = train_df.drop(labels=['requester_received_pizza'], axis=1)

    for c in train_df.columns:
        if train_df[c].dtype == np.int64:
            train_df[c] = train_df[c].astype(np.float64)
            test_df[c] = test_df[c].astype(np.float64)

    if use_text_data:
        print 'shape1', train_df.values[:,2:].shape, train_review_features.shape, train_title_features.shape

        xtrain = np.hstack([train_df.values[:,2:], train_review_features, train_title_features])
        xtest = np.hstack([test_df.values[:,2:], test_review_features, test_title_features])
        ytest = test_df.values[:,1]
    else:
        xtrain = train_df.values[:,2:]
        xtest = test_df.values[:,2:]
        ytest = test_df.values[:,1]

    print 'shape2', xtrain.shape, ytrain.shape, xtest.shape, ytest.shape

    return xtrain, ytrain, xtest, ytest

def score_model(model, xtrain, ytrain):
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(xtrain,
                                                                     ytrain,
                                                                     test_size=0.4, random_state=randint)
    model.fit(xTrain, yTrain)
    ytpred = model.predict(xTest)
    print 'roc', roc_auc_score(yTest, ytpred)
    print 'score', model.score(xTest, yTest)
    return roc_auc_score(yTest, ytpred)

def compare_models(xtraindata, ytraindata):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.lda import LDA
    from sklearn.qda import QDA

    classifier_dict = {
                #'linSVC': LinearSVC(),
                #'kNC5': KNeighborsClassifier(),
                #'kNC6': KNeighborsClassifier(6),
                #'SVC': SVC(kernel="linear", C=0.025),
                #'DT': DecisionTreeClassifier(max_depth=5),
                #'RF200': RandomForestClassifier(n_estimators=200, n_jobs=-1),
                'RF400gini': RandomForestClassifier(n_estimators=400, n_jobs=-1),
                'RF400entropy': RandomForestClassifier(n_estimators=400, n_jobs=-1, criterion='entropy'),
                #'RF800': RandomForestClassifier(n_estimators=800, n_jobs=-1),
                #'RF1000': RandomForestClassifier(n_estimators=1000, n_jobs=-1),
                'Ada': AdaBoostClassifier(),
                #'SVClin': SVC(kernel='linear'),
                #'SVCpoly': SVC(kernel='poly'),
                #'SVCsigmoid': SVC(kernel='sigmoid'),
                'Gauss': GaussianNB(),
                'LDA': LDA(),
                #'QDA': QDA(),
                'SVC': SVC(),
               }

    results = {}
    ytrain_vals = []
    ytest_vals = []
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(xtraindata,
                                                                     ytraindata,
                                                                     test_size=0.4, random_state=randint)
    scale = StandardScaler()
    xTrain = scale.fit_transform(xTrain)
    xTest = scale.transform(xTest)

    for name, model in sorted(classifier_dict.items()):
        model.fit(xTrain, yTrain)
        ytrpred = model.predict(xTrain)
        ytpred = model.predict(xTest)
        results[name] = roc_auc_score(yTest, ytpred)
        ytrain_vals.append(ytrpred)
        ytest_vals.append(ytpred)
        print name, results[name], ytest_vals[-1]
    print '\n\n\n'

    print 'shape3', xTrain.shape, xTest.shape, ytrain_vals[0].shape, ytest_vals[0].shape
    xTrain = np.hstack([xTrain]+[y.reshape(xTrain.shape[0],1) for y in ytrain_vals])
    xTest = np.hstack([xTest]+[y.reshape(xTest.shape[0],1) for y in ytest_vals])

    print '\n\n\n'
    model = RandomForestClassifier(n_estimators=400, n_jobs=-1)
    model.fit(xTrain, yTrain)
    ytpred = model.predict(xTest)
    print 'RF400', roc_auc_score(yTest, ytpred)

def prepare_submission(model, xtrain, ytrain, xtest, ytest):
    model.fit(xtrain, ytrain)
    ytest2 = model.predict(xtest)
    request_id = ytest

    df = pd.DataFrame({'request_id': request_id, 'requester_received_pizza': ytest2}, columns=('request_id','requester_received_pizza'))
    df.to_csv('submission.csv', index=False)

    return

if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = load_data()

    print 'shapes', xtrain.shape, ytrain.shape, xtest.shape, ytest.shape

    compare_models(xtrain, ytrain)
    #model = Pipeline([('scale', StandardScaler()),
                      #('rf800', RandomForestClassifier(n_estimators=800, n_jobs=-1))])
    #print 'score', score_model(model, xtrain, ytrain)
    ##print model.feature_importances_
    #prepare_submission(model, xtrain, ytrain, xtest, ytest)
