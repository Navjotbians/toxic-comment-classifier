
# coding: utf-8

# # Inference

# In[1]:


import pickle
import os
dir_path = os.path.dirname(os.getcwd())
import sys  
sys.path.append(os.path.join(dir_path, "src"))
from clean_comments import clean
from processing import process_txt


# In[2]:


### load model
pkl_file = os.path.join(dir_path, 'model', 'final_model.pkl')
open_file = open(pkl_file, "rb")
model = pickle.load(open_file)
open_file.close()


# In[3]:


### load vectorizer
pkl_file = os.path.join(dir_path, 'model', 'final_vectorizer.pkl')
open_file = open(pkl_file, "rb")
bw_vectorizer = pickle.load(open_file)
open_file.close()


# In[4]:


i1 = ["that is so good, i am so happy bitch!"]
i2 = ['This project is quite interesting to work on']
i3 = ["i'm going to kill you nigga, you are you sick or mad, i don't like you at all"]
i4 = ["D'aww! He matches this background colour I'm seemingly stuck with. Thanks.  (talk) 21:51, January 11, 2016 (UTC)"]


# In[5]:


input_str = clean(i1[0])
input_str = process_txt(input_str, stemm= True)
input_str = bw_vectorizer.transform([input_str])


# In[6]:


prediction = model.predict(input_str)
prediction


# In[7]:


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

predc = [labels[i] for i in range (0,len(prediction[0])) if prediction[0][i] == 1]

if len(predc)== 0:
    i ='comment in not toxic'
    print(i)
else:
    print("Prediction : {}".format(" | ".join(predc)))

