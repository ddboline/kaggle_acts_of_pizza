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
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)


def load_data():
    train_df = pd.read_json('train.json')
    test_df = pd.read_json('test.json')
    subm_df = pd.read_csv('sampleSubmission.csv')

    #train_df = train_df.drop(labels=['giver_username_if_known', 'request_id'], axis=1)
    #test_df = test_df.drop(labels=['giver_username_if_known', 'request_id'], axis=1)
    
    #print train_df.shape, test_df.shape
    #print '\n'.join(train_df.columns)
    #print '\n\n%s' % '\n'.join(test_df.columns)
    #print '\n\n%s' % '\n'.join(subm_df.columns)
    
    #print ['%s' % c for c in train_df.columns if c not in test_df.columns]
    
    YCOL = u'requester_received_pizza'

    XCOLS_KEEP = [u'requester_account_age_in_days_at_request', u'requester_days_since_first_post_on_raop_at_request', u'requester_number_of_comments_at_request', u'requester_number_of_comments_in_raop_at_request', u'requester_number_of_posts_at_request', u'requester_number_of_posts_on_raop_at_request', u'requester_number_of_subreddits_at_request', u'requester_upvotes_minus_downvotes_at_request', u'requester_upvotes_plus_downvotes_at_request', u'unix_timestamp_of_request', u'unix_timestamp_of_request_utc']
    
    XCOLS_TOSS = [u'number_of_downvotes_of_request_at_retrieval', u'number_of_upvotes_of_request_at_retrieval', u'post_was_edited', u'request_number_of_comments_at_retrieval', u'request_text', u'requester_account_age_in_days_at_retrieval', u'requester_days_since_first_post_on_raop_at_retrieval', u'requester_number_of_comments_at_retrieval', u'requester_number_of_comments_in_raop_at_retrieval', u'requester_number_of_posts_at_retrieval', u'requester_number_of_posts_on_raop_at_retrieval', u'requester_upvotes_minus_downvotes_at_retrieval', u'requester_upvotes_plus_downvotes_at_retrieval', u'requester_user_flair']
    
    train_df = train_df.drop(labels=XCOLS_TOSS, axis=1)
    
    for df in train_df, test_df:
        df['request_text_edit_aware'] = df['request_text_edit_aware'].map(review_to_wordlist).map(len)
        df['request_title'] = df['request_title'].map(review_to_wordlist).map(len)
        df['requester_subreddits_at_request'] = df['requester_subreddits_at_request'].map(len)
        df['requester_username'] = df['requester_username'].map(len)
        df['requester_account_age_in_days_at_request'] = df['requester_account_age_in_days_at_request'].astype(np.int64)
        df['requester_days_since_first_post_on_raop_at_request'] = df['requester_days_since_first_post_on_raop_at_request'].astype(np.int64)

    ytrain = train_df['requester_received_pizza'].astype(np.int64).values
    print len(ytrain), ytrain.sum()
    
    train_df = train_df.drop(labels=['requester_received_pizza'], axis=1)
    
    print train_df.shape, test_df.shape
    print train_df.columns
    print test_df.columns
    
    xtrain = train_df.values[:,2:]
    xtest = test_df.values[:,2:]
    ytest = test_df.values[:,1]
    
    print xtrain.shape, ytrain.shape, xtest.shape, ytest.shape
    print xtrain
    print ytrain
    print xtest
    print ytest
    
    return xtrain, ytrain, xtest, ytest

def score_model(model, xtrain, ytrain):
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    xTrain, xTest, yTrain, yTest = cross_validation.train_test_split(xtrain,
                                                                     ytrain,
                                                                     test_size=0.4, random_state=randint)
    model.fit(xTrain, yTrain)
    
    return model.score(xTest, yTest)

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
                'RF200': RandomForestClassifier(n_estimators=200, n_jobs=-1),
                'RF400': RandomForestClassifier(n_estimators=400, n_jobs=-1),
                'RF800': RandomForestClassifier(n_estimators=800, n_jobs=-1),
                'RF1000': RandomForestClassifier(n_estimators=1000, n_jobs=-1),
                #'Ada': AdaBoostClassifier(),
                #'Gauss': GaussianNB(),
                #'LDA': LDA(),
                #'QDA': QDA(),
                #'SVC2': SVC(),
                }

    results = {}
    for name, model in classifier_dict.items():
        print name
        results[name] = score_model(model, xtraindata, ytraindata)
        print name, results[name]
    print '\n\n\n'
    for name, result in sorted(results.items(), key=lambda x: x[1]):
        print name, result


def prepare_submission(model, xtrain, ytrain, xtest, ytest):
    model.fit(xtrain, ytrain)
    ytest2 = model.predict(xtest)
    ids = ytest
    
    df = pd.DataFrame({'id': ids, 'hand': ytest2}, columns=('id','hand'))
    df.to_csv('submission.csv', index=False)
    
    return

if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = load_data()
    
    #pca = PCA(n_components=4)
    #x_pca = np.vstack([xtrain, xtest])
    #print x_pca.shape
    #pca.fit(xtrain)
    
    #xtrain = pca.transform(xtrain)
    #xtest = pca.transform(xtest)
    
    compare_models(xtrain, ytrain)
    #model = RandomForestClassifier(n_estimators=400, n_jobs=-1)
    #print 'score', score_model(model, xtrain, ytrain)
    #print model.feature_importances_
    #prepare_submission(model, xtrain, ytrain, xtest, ytest)
    