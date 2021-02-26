
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string
import operator
from sklearn import feature_extraction,model_selection,naive_bayes,pipeline,manifold,preprocessing


# In[2]:


# import sys
# print(sys.getrecursionlimit()))


# In[3]:


# sys.setrecursionlimit(5000)


# In[4]:


df = pd.read_csv('../Input/train.csv')


# In[5]:


### data cleaner function
def clean(input_str):
    input_str = input_str.lower()
    input_str = re.sub(r'\d+', '', input_str)
    input_str = re.sub(r"n't", " not ", input_str)
    input_str = re.sub(r"can't", "cannot ", input_str)
    input_str = re.sub(r"what's", "what is ", input_str)
    input_str = re.sub(r"\'s", " ", input_str)
    input_str = re.sub(r"\'ve", " have ", input_str)
    input_str = re.sub(r"\'re", " are ", input_str)
    input_str = re.sub(r"\'d", " would ", input_str)
    input_str = re.sub(r"\'ll", " will ", input_str)
    input_str = re.sub(r"\'scuse", " excuse ", input_str)
    input_str = re.sub(r"i'm", "i am", input_str)
    input_str = re.sub(r" m ", " am ", input_str)
    input_str = re.sub('\s+', ' ', input_str)
    input_str = re.sub('\W', ' ', input_str)
    input_str = input_str.translate(str.maketrans('','', string.punctuation))
    input_str = input_str.strip()
    return input_str


# In[6]:


### Frequently used words in the obscene comments
def make_dict(d, stemm = False,lemm = True):
    #all_word = []
        ### Clean input data
    processed_text = clean(d)
        ### Tokenization
    processed_text = word_tokenize(processed_text)
     ### remove stop words
    processed_text = [word for word in processed_text if word not in stopwords.words('english')]
    #all_word.append(processed_text)

    ### Stemming
    if stemm == True:
      ps = nltk.stem.porter.PorterStemmer()
      processed_text = [ps.stem(word) for word in processed_text]

    ### Lemmatization
    if lemm == True:
      lem = nltk.stem.wordnet.WordNetLemmatizer()
      processed_text = [lem.lemmatize(word) for word in processed_text]

    text = " ".join(processed_text)
    
    return text

    


# In[7]:


input_str = df.comment_text[2]

make_dict(input_str, stemm= True)


# ## Append the clean comments in the dataset

# In[8]:


# df['clean_comment'] = df['comment_text'].apply(lambda x:make_dict(x, stemm= True))


# In[28]:


#df.head(20)

