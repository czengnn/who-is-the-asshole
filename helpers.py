import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import string
import contractions
import spacy
import pickle 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, IncrementalPCA

def clean_lil_bit(text):
    text = text.lower()
    # remove new line, *, links
    text = text.replace('\n',' ').replace('*','')
    text = re.sub('(https?\S+|www.\S+)','', text)
    text = re.sub('\[?\(?aita\]?\)?', '', text)
    text = re.sub('\[?\(?wibta\]?\)?', '', text)
    text = re.sub('\s(bf)\s', ' boyfriend ', text)
    text = re.sub('\s(gf)\s', ' girlfriend ', text)
    text = contractions.fix(text)
    return text

sp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def remove_punc(s):
    s = re.sub('[%s]' % re.escape(string.punctuation), '', s)
    s = re.sub('[‘’“”…]', '', s)
    s = re.sub('\w*\d\w*', '', s)

    # stop = set(stopwords.words('english'))
    sub_question = set(['aita','wibta'])

    post = sp(s)
    s = ' '.join([word.lemma_ for word in post if word.text not in sub_question])
    return s



class Proctologist:
    def __init__(self):
        self.cv = pickle.load(open('models/cv_fit_train.sav', 'rb'))
        self.pca = pickle.load(open('models/pca_combo.sav', 'rb'))
        self.model = pickle.load(open('models/lr.sav', 'rb'))
        
    def text_convert(self, arr):
        text_df = pd.DataFrame(arr, columns = ['text'])
        # clean up the text
        text_df['text_lil_clean'] = text_df['text'].apply(lambda x : clean_lil_bit(x))
        # add sentiment analysis score to df
        text_df['polarity'] = text_df['text_lil_clean'].apply(lambda x : TextBlob(x).sentiment.polarity)
        text_df['subjectivity'] = text_df['text_lil_clean'].apply(lambda x : TextBlob(x).sentiment.subjectivity)
        
        # fincal clean up of text
        text_df['text_clean'] = text_df['text_lil_clean'].apply(lambda x: remove_punc(x))
        
        self.df = text_df[['text_clean','polarity','subjectivity']]
        
        # create dtm DF
        cv_dtm = self.cv.transform(self.df['text_clean'])
        cv_cols = self.cv.get_feature_names()
        self.dtm = pd.DataFrame(cv_dtm.toarray(), columns=cv_cols)
        
        # create pca DF
        dtm_pca = self.pca.transform(self.dtm)
        pca_cols = ['PC_' + str(i) for i in range(1, self.pca.get_params()['n_components']+1)]
        self.dtm_pca_df = pd.DataFrame(dtm_pca, columns=pca_cols)
        
        # combine PCA and sentiment analysis into X DF
        self.X = pd.concat([self.df[['polarity','subjectivity']], self.dtm_pca_df], axis=1)
        
    def judgement(self, arr):
        self.text_convert(arr)
        return self.model.predict(self.X)

