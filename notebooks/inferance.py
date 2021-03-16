
# coding: utf-8

# # Inference

# In[43]:


os.getcwd()


# In[42]:


import os
dir_path = os.path.dirname(os.getcwd())


# In[2]:


import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import feature_extraction,model_selection,preprocessing, naive_bayes,pipeline, manifold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score
import pickle
import os
import sys  
sys.path.append(os.path.join(dir_path, "src"))
from word_embeddings import get_embeddings
from clean_comments import clean
from processing import process_txt


# In[3]:


### load model
pkl_file = os.path.join(dir_path, 'model', 'final_model.pkl')
open_file = open(pkl_file, "rb")
model = pickle.load(open_file)
open_file.close()


# In[4]:


### load vectorizer
pkl_file = os.path.join(dir_path, 'model', 'final_vectorizer.pkl')
open_file = open(pkl_file, "rb")
bw_vectorizer = pickle.load(open_file)
open_file.close()


# In[38]:


i1 = ["that is so good, i am so happy bitch!"]
i2 = ['This project is quite interesting to work on']
i3 = ["i'm going to kill you nigga, you are you sick or mad, i don't like you at all"]


# In[39]:


input_str = clean(i2[0])
input_str = process_txt(input_str, stemm= True)
input_str = bw_vectorizer.transform([input_str])


# In[40]:


prediction = model.predict(input_str)
prediction


# In[41]:


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
predc = []
for i,j in zip(prediction[0], labels):
    if i == 1:
        predc.append(j)
print(predc)

if len(predc)== 0:
    i ='comment in not toxic'
    print(i)
else:
    i = str(predc)
    print(i)


# In[ ]:


1

