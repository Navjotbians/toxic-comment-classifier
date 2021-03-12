
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import feature_extraction,model_selection,preprocessing, naive_bayes,pipeline, manifold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.externals import joblib
import pickle
import os
import sys  
sys.path.append('F:/AI/Toxic-comment-classifier/src')
from word_embeddings import w_embeddings
from clean_comments import clean
from processing import process_txt


# In[53]:


### load model
pkl_file = r'F:\AI\Toxic-comment-classifier\model\MultinomialNB'
open_file = open(pkl_file, "rb")
model = pickle.load(open_file)
open_file.close()


# In[54]:


### load vectorizer
pkl_file = r'F:\AI\Toxic-comment-classifier\model\bw_vectorizer1000.pkl'
open_file = open(pkl_file, "rb")
bw_vectorizer = pickle.load(open_file)
open_file.close()


# In[55]:


input_str = ["i'm going to kill you nigga, you are you sick or mad, i don't like you at all"]
i1 = ["that is so good, i am so happy bitch"]


# In[56]:


input_str = clean(i1[0])
input_str = process_txt(input_str, stemm= True)
input_str = bw_vectorizer.transform([input_str])


# In[57]:


model.predict(input_str)

