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

# base_model = pickle.load(open('models/rfc_sen.sav', 'rb'))
# base_cv = pickle.load(open('models/cv_fit_train_min10.sav', 'rb'))
# pca_2 = pickle.load(open('models/pca_2.sav', 'rb'))
class Proctologist:
    def __init__(self, cv, model, sentiment=True):
        self.cv = cv
        self.model = model
        self.sentiment = sentiment
        self.pca = pca
        
    def text_convert(self, arr):
        text_df = pd.DataFrame(arr, columns = ['text'])
        # clean up the text
        text_df['text_lil_clean'] = text_df['text'].apply(lambda x : clean_lil_bit(x))
        # add sentiment analysis score to df
        text_df['polarity'] = text_df['text_lil_clean'].apply(lambda x : TextBlob(x).sentiment.polarity)
        text_df['subjectivity'] = text_df['text_lil_clean'].apply(lambda x : TextBlob(x).sentiment.subjectivity)
        # fincal clean up of text
        text_df['text_clean'] = text_df['text_lil_clean'].apply(lambda x: remove_punc(x))
        
        # create dtm DF
        cv_dtm = self.cv.transform(text_df['text_clean'])
        cv_cols = self.cv.get_feature_names()
        self.dtm = pd.DataFrame(cv_dtm.toarray(), columns=cv_cols)
        
        if self.sentiment:
        # combine DTM and sentiment analysis into X
            self.X = pd.concat([text_df[['polarity','subjectivity']], self.dtm], axis=1)
        else:
            self.X = self.dtm
        
    def diagnosis(self, arr):
        self.text_convert(arr)        
        verdict = self.model.predict(self.X)
        verdict_df = pd.DataFrame(zip(arr,verdict), columns=['text', 'asshole'])
        return verdict_df
