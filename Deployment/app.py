from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from sklearn.ensemble import VotingClassifier
import gensim
from sklearn.externals import joblib
import nltk
from sklearn import metrics

app = Flask(__name__)

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():

    df_train = pd.read_csv('train-dataset.csv')
    df_test = pd.read_csv('hold-out.csv')
    df_test = df_test[~(df_test['comment'].isnull())]
    X_train = df_train['comment']
    X_test = df_test['comment'][:10000]

    y_train = df_train['offensive']
    y_test = df_test['offensive'][:10000]

    tokenized_train = [nltk.word_tokenize(t) for t in X_train]
    tokenized_test = [nltk.word_tokenize(t) for t in X_test]
    num_features = 256

    w2v_model = gensim.models.Word2Vec(tokenized_train , size = num_features, window = 150 , min_count=10 , sample=1e-3 , workers= 16)
    w2v_model.save('w2v')
    w2v_model = gensim.models.Word2Vec.load('w2v')

    def averaged_word2vec_vectorizer (corpus , model , num_features):
        vocabulary = set(model.wv.index2word)

        def average_word_vectors (words, model , vocabulary , num_features):
            feature_vector = np.zeros((num_features) , dtype = 'float64')
            nwords = 0

            for word in words :
                if word in vocabulary :
                    nwords += 1
                    feature_vector = np.add(feature_vector , model.wv[word])
            if nwords :
                feature_vector = np.divide(feature_vector , nwords)
            return feature_vector
        features = [average_word_vectors(tokenized_sentence , model , vocabulary , num_features) for tokenized_sentence in corpus]

        return np.array(features)

    avg_wv_train_features = averaged_word2vec_vectorizer (corpus = tokenized_train , model = w2v_model , num_features= num_features)
    avg_wv_test_features = averaged_word2vec_vectorizer (corpus = tokenized_test , model = w2v_model , num_features = num_features)

    lr = LogisticRegression(penalty='l2', max_iter=500, C=1, solver='lbfgs')

    smote_w2v_pipeline = make_pipeline_imb(SMOTE(sampling_strategy= .95 , k_neighbors = 40 , kind = 'borderline2') , lr )

    smote_w2v_model = smote_w2v_pipeline.fit(avg_wv_train_features , y_train )

    smote_w2v_predict = smote_w2v_model.predict(avg_wv_test_features)

    metrics.recall_score(y_test , smote_w2v_predict)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        tokenized = [nltk.word_tokenize(t) for t in data]
        avg_wv_features = averaged_word2vec_vectorizer (corpus = tokenized , model = w2v_model , num_features= num_features)
        my_prediction = smote_w2v_model.predict(avg_wv_features)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug= True , host="0.0.0.0", port=80)
