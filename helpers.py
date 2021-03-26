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
        self.pca = pickle.load(open('models/pca.sav', 'rb'))

    

    def text_convert(s):
        s_clean = clean_lil_bit(s)
        s_clean = remove_punc(s_clean)

        dtm_s = cv.transform(s_clean)
        dtm_pca = pca.transform(dtm_s)
        return dtm_pca


